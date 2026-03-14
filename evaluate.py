#!/usr/bin/env python3
"""
Evaluate local HF/PEFT models on the risk-averse benchmark with permissive parsing.

Default behavior matches the original standard evaluator (single run, no steering).
Optional steering controls allow ICV direction construction/injection and alpha sweeps.
"""

import argparse
import gc
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_parser import extract_choice_with_strategy
from icv_steering_experiment import ResidualSteeringHook, build_icv_direction, read_jsonl
from risk_averse_prompts import DEFAULT_SYSTEM_PROMPT


# Flush output immediately so logs are visible in real time.
sys.stdout.reconfigure(line_buffering=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()


def remove_instruction_suffix(prompt):
    """Remove the instruction about how to respond from the end of the prompt."""
    patterns = [
        r"\s*You can think before answering,.*?would select\.",
        r"\s*You can think.*?must finish with.*?\.",
    ]
    for pattern in patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE | re.DOTALL)
    return prompt.strip()


def clean_bucket_label(value):
    """Normalize low_bucket_label strings like '"lin_only"' -> 'lin_only'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1]
    return s


def parse_label_list(value):
    """Parse list-like label fields stored as JSON strings in CSV."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    s = str(value).strip()
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if isinstance(parsed, str):
            return [parsed]
        return [str(parsed)]
    except Exception:
        s = s.strip('"').strip("'")
        if not s:
            return []
        if "," in s:
            return [part.strip().strip('"').strip("'") for part in s.split(",") if part.strip()]
        return [s]


def parse_bool_like(value):
    """Parse bool-ish CSV values robustly (handles numpy/pandas/string forms)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return bool(value)


def infer_probability_format(prompt_text):
    """Best-effort fallback if explicit use_verbal_probs is missing."""
    if not isinstance(prompt_text, str):
        return None
    if re.search(r"\d+\s*%", prompt_text):
        return "numerical"
    verbal_markers = [
        "very likely",
        "likely",
        "unlikely",
        "very unlikely",
        "almost certain",
        "almost no chance",
        "small chance",
    ]
    prompt_lower = prompt_text.lower()
    if any(marker in prompt_lower for marker in verbal_markers):
        return "verbal"
    return None


def probability_format_from_value(use_verbal_probs_value, prompt_text=None):
    parsed_bool = parse_bool_like(use_verbal_probs_value)
    if parsed_bool is True:
        return "verbal"
    if parsed_bool is False:
        return "numerical"
    return infer_probability_format(prompt_text)


def label_to_option_number(label):
    """Convert a label like 'a' or '1' into a 1-based option number."""
    s = str(label).strip().lower()
    if s.isdigit():
        return int(s)
    if len(s) == 1 and "a" <= s <= "z":
        return ord(s) - ord("a") + 1
    return None


def parse_alpha_list(value: str) -> List[float]:
    """Parse comma-separated alpha list."""
    alphas = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        alphas.append(float(raw))
    if not alphas:
        raise ValueError("No valid values parsed from --alphas")
    return alphas


def alpha_to_suffix(alpha: float) -> str:
    """Stable filename-safe suffix for alpha values."""
    prefix = "neg" if alpha < 0 else "pos"
    magnitude = f"{abs(alpha):g}".replace(".", "p")
    return f"{prefix}{magnitude}"


def get_input_device(model):
    """Best-effort model input device for tokenized tensors."""
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def get_decoder_layers(model):
    """Return decoder block list for common causal LM architectures."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError("Unsupported model architecture for steering hooks.")


def load_steering_direction(path: str) -> torch.Tensor:
    """Load a steering vector from disk (tensor or dict wrapper)."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.detach().to(torch.float32).cpu()
    if isinstance(obj, dict):
        for key in ("direction", "icv_direction", "vector", "steering_direction"):
            value = obj.get(key)
            if isinstance(value, torch.Tensor):
                return value.detach().to(torch.float32).cpu()
            if isinstance(value, (list, tuple)):
                return torch.tensor(value, dtype=torch.float32)
    if isinstance(obj, (list, tuple)):
        return torch.tensor(obj, dtype=torch.float32)
    raise ValueError(
        f"Unsupported steering direction payload at {path}. "
        "Expected Tensor, list, or dict with a direction tensor/list."
    )


def convert_numpy(obj):
    """Convert numpy/torch scalar-like values to native Python for JSON."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


def apply_chat_template_safe(tokenizer, messages, disable_thinking: bool) -> str:
    """Apply chat template, tolerating tokenizers without enable_thinking."""
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=False, **template_kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **template_kwargs)


def build_eval_prompt(prompt_raw: str, prompt_suffix: str) -> str:
    """Normalize the dataset prompt and append an optional suffix."""
    prompt = remove_instruction_suffix(prompt_raw)
    return f"{prompt}\n\n{prompt_suffix}".strip() if prompt_suffix else prompt


def build_messages(eval_prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    """Build chat messages for one evaluation request."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": eval_prompt})
    return messages


def count_generated_tokens(gen_ids: torch.Tensor, pad_token_id: Optional[int]) -> int:
    """Count generated tokens, excluding right-padding after EOS."""
    if pad_token_id is None:
        return int(gen_ids.shape[0])
    return int(gen_ids.ne(pad_token_id).sum().item())


def vllm_settings_from_args(args) -> Dict[str, Any]:
    """Serialize the vLLM-specific runtime settings for outputs."""
    return {
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "max_model_len": args.vllm_max_model_len,
        "dtype": args.vllm_dtype,
        "enable_prefix_caching": args.vllm_enable_prefix_caching,
        "max_lora_rank": args.vllm_max_lora_rank if args.model_path else None,
    }


def load_vllm_engine(args):
    """Lazily construct a vLLM engine and optional LoRA request."""
    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise ImportError(
            "vLLM backend requested, but `vllm` is not installed. "
            "Install it on the GPU host, then re-run with --backend vllm."
        ) from exc

    llm_kwargs: Dict[str, Any] = {
        "model": args.base_model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.vllm_tensor_parallel_size,
        "gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "dtype": args.vllm_dtype,
    }
    if args.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = args.vllm_max_model_len
    if args.vllm_enable_prefix_caching:
        llm_kwargs["enable_prefix_caching"] = True
    if args.model_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.vllm_max_lora_rank

    engine = LLM(**llm_kwargs)

    lora_request = None
    if args.model_path:
        adapter_name = Path(args.model_path).resolve().name
        lora_request = LoRARequest(adapter_name, 1, str(Path(args.model_path).resolve()))

    return engine, lora_request


def summarize_results(results):
    """Compute aggregate metrics from per-situation result records."""
    valid = [r for r in results if r["option_type"] is not None]
    if valid:
        cooperate_rate = sum(r["option_type"] == "Cooperate" for r in valid) / len(valid)
        rebel_rate = sum(r["option_type"] == "Rebel" for r in valid) / len(valid)
        steal_rate = sum(r["option_type"] == "Steal" for r in valid) / len(valid)
        cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid)
        linear_valid = [r for r in valid if r.get("is_best_linear") is not None]
        linear_rate = sum(r["is_best_linear"] for r in linear_valid) / len(linear_valid) if linear_valid else 0
    else:
        cooperate_rate = rebel_rate = steal_rate = cara_rate = linear_rate = 0

    parse_rate = len(valid) / len(results) if results else 0
    return {
        "parse_rate": parse_rate,
        "cooperate_rate": cooperate_rate,
        "rebel_rate": rebel_rate,
        "steal_rate": steal_rate,
        "best_cara_rate": cara_rate,
        "best_linear_rate": linear_rate,
    }


def save_incremental(
    output_path,
    args,
    results,
    failed_responses,
    situations_evaluated,
    *,
    steering_alpha: float,
    steering_info: Optional[Dict] = None,
):
    """Save current run state to disk for crash resilience."""
    metrics = summarize_results(results)
    valid = [r for r in results if r["option_type"] is not None]

    eval_cfg = {
        "backend": args.backend,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
        "num_situations": situations_evaluated,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "system_prompt": args.system_prompt,
        "prompt_suffix": args.prompt_suffix,
        "disable_thinking": args.disable_thinking,
        "steering_alpha": steering_alpha,
    }
    if args.backend == "vllm":
        eval_cfg["vllm"] = vllm_settings_from_args(args)
    if steering_info:
        eval_cfg["steering"] = steering_info

    output_data = convert_numpy(
        {
            "evaluation_config": eval_cfg,
            "metrics": metrics,
            "num_valid": len(valid),
            "num_total": len(results),
            "results": None if args.no_save_responses else results,
            "failed_responses": failed_responses[:10],
        }
    )

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def build_situations(df: pd.DataFrame, num_situations: int):
    """Group rows into situation objects with option metadata."""
    situations = []
    for sit_id in df["situation_id"].unique()[:num_situations]:
        sit_data = df[df["situation_id"] == sit_id]
        prompt_raw = sit_data["prompt_text"].iloc[0]
        num_options = len(sit_data)
        use_verbal_probs = sit_data["use_verbal_probs"].iloc[0] if "use_verbal_probs" in df.columns else None
        low_bucket_label = (
            clean_bucket_label(sit_data["low_bucket_label"].iloc[0]) if "low_bucket_label" in df.columns else None
        )

        linear_best_indices_0 = set()
        linear_best_option_numbers = set()
        has_linear_info = False
        if "is_best_linear_display" in df.columns:
            has_linear_info = True
            linear_best_indices_0 = set(
                int(idx) for idx in sit_data.loc[sit_data["is_best_linear_display"] == True, "option_index"]
            )
            linear_best_option_numbers = {idx + 1 for idx in linear_best_indices_0}
        elif "linear_best_labels" in df.columns:
            has_linear_info = True
            lin_labels = parse_label_list(sit_data["linear_best_labels"].iloc[0])
            linear_best_option_numbers = {
                label_to_option_number(l) for l in lin_labels if label_to_option_number(l) is not None
            }
            linear_best_indices_0 = {n - 1 for n in linear_best_option_numbers}
        if not linear_best_option_numbers:
            has_linear_info = False

        cara001_best_option_numbers = set()
        if "CARA_correct_labels" in df.columns:
            cara_labels = parse_label_list(sit_data["CARA_correct_labels"].iloc[0])
            cara001_best_option_numbers = {
                label_to_option_number(l) for l in cara_labels if label_to_option_number(l) is not None
            }

        if not cara001_best_option_numbers and "CARA_alpha_0_01_best_labels" in df.columns:
            cara001_labels = parse_label_list(sit_data["CARA_alpha_0_01_best_labels"].iloc[0])
            cara001_best_option_numbers = {
                label_to_option_number(l) for l in cara001_labels if label_to_option_number(l) is not None
            }

        if not cara001_best_option_numbers and "is_best_cara_display" in df.columns:
            cara001_best_option_numbers = {
                int(idx) + 1 for idx in sit_data.loc[sit_data["is_best_cara_display"] == True, "option_index"]
            }

        bucket_label = low_bucket_label
        if bucket_label is None and linear_best_option_numbers and cara001_best_option_numbers:
            if linear_best_option_numbers == cara001_best_option_numbers:
                bucket_label = "both"

        options = {}
        best_cara_indices = set()
        for _, row in sit_data.iterrows():
            idx = int(row["option_index"])
            letter = chr(ord("a") + idx)
            number = str(idx + 1)
            is_best_cara = row.get("is_best_cara_display", False) == True
            option_data = {
                "type": row["option_type"],
                "is_best_cara": is_best_cara,
                "is_best_linear": (idx in linear_best_indices_0) if has_linear_info else None,
                "option_index": idx,
            }
            options[letter] = option_data
            options[number] = option_data
            if is_best_cara:
                best_cara_indices.add(idx)

        situations.append(
            {
                "situation_id": sit_id,
                "prompt_raw": prompt_raw,
                "num_options": num_options,
                "options": options,
                "probability_format": probability_format_from_value(use_verbal_probs, prompt_raw),
                "bucket_label": bucket_label,
                "linear_best_option": sorted(linear_best_option_numbers),
                "cara001_best_option": sorted(cara001_best_option_numbers),
                "best_cara_indices": sorted(best_cara_indices),
            }
        )
    return situations


def generate_response_transformers(
    *,
    model,
    tokenizer,
    eval_prompts: List[str],
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    max_time_per_generation: float,
    disable_thinking: bool,
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
    steering_alpha: float = 0.0,
):
    """Generate one or more responses with the Transformers backend."""
    texts = [
        apply_chat_template_safe(
            tokenizer,
            build_messages(eval_prompt, system_prompt),
            disable_thinking=disable_thinking,
        )
        for eval_prompt in eval_prompts
    ]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(get_input_device(model))
    prompt_token_count = inputs["input_ids"].shape[1]

    hook = None
    if steering_block is not None and steering_direction is not None and abs(steering_alpha) > 0:
        block_device = next(steering_block.parameters()).device
        direction = steering_direction.to(device=block_device, dtype=model.dtype)
        hook = ResidualSteeringHook(direction=direction, alpha=steering_alpha).register(steering_block)

    gen_start = time.time()
    try:
        with torch.inference_mode():
            if temperature == 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation,
                    use_cache=True,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation,
                    use_cache=True,
                )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and len(eval_prompts) > 1:
            print(
                f"  WARNING: CUDA OOM while generating batch of {len(eval_prompts)}. "
                "Retrying sequentially."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            responses = []
            generated_token_counts = []
            total_elapsed = 0.0
            metadata = []
            for eval_prompt in eval_prompts:
                sub_responses, sub_token_counts, sub_elapsed, sub_metadata = generate_response_transformers(
                    model=model,
                    tokenizer=tokenizer,
                    eval_prompts=[eval_prompt],
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    max_time_per_generation=max_time_per_generation,
                    disable_thinking=disable_thinking,
                    steering_block=steering_block,
                    steering_direction=steering_direction,
                    steering_alpha=steering_alpha,
                )
                responses.extend(sub_responses)
                generated_token_counts.extend(sub_token_counts)
                total_elapsed += sub_elapsed
                metadata.extend(sub_metadata)
            return responses, generated_token_counts, total_elapsed, metadata
        raise
    finally:
        if hook is not None:
            hook.remove()

    gen_elapsed = time.time() - gen_start
    responses = []
    generated_token_counts = []
    metadata = []
    for output_ids in outputs:
        gen_ids = output_ids[prompt_token_count:]
        responses.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        token_count = count_generated_tokens(gen_ids, tokenizer.pad_token_id)
        generated_token_counts.append(token_count)
        metadata.append(
            {
                "finish_reason": "length" if token_count >= max_new_tokens else "eos_or_stop",
                "stop_reason": None,
            }
        )
    return responses, generated_token_counts, gen_elapsed, metadata


def generate_response_vllm(
    *,
    model,
    eval_prompts: List[str],
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    disable_thinking: bool,
    lora_request=None,
):
    """Generate one or more responses with the vLLM backend."""
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise ImportError(
            "vLLM backend requested, but `vllm` is not installed. "
            "Install it on the GPU host, then re-run with --backend vllm."
        ) from exc

    batch_messages = [build_messages(eval_prompt, system_prompt) for eval_prompt in eval_prompts]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        ignore_eos=False,
    )

    call_kwargs: Dict[str, Any] = {
        "messages": batch_messages,
        "sampling_params": sampling_params,
        "use_tqdm": False,
    }
    if lora_request is not None:
        call_kwargs["lora_request"] = lora_request
    if disable_thinking:
        call_kwargs["chat_template_kwargs"] = {"enable_thinking": False}

    gen_start = time.time()
    try:
        outputs = model.chat(**call_kwargs)
    except TypeError:
        call_kwargs.pop("chat_template_kwargs", None)
        outputs = model.chat(**call_kwargs)
    gen_elapsed = time.time() - gen_start

    responses = []
    generated_token_counts = []
    metadata = []
    for request_output in outputs:
        if not getattr(request_output, "outputs", None):
            responses.append("")
            generated_token_counts.append(0)
            metadata.append({"finish_reason": None, "stop_reason": None})
            continue
        completion = request_output.outputs[0]
        responses.append(completion.text)
        token_ids = getattr(completion, "token_ids", None) or []
        generated_token_counts.append(len(token_ids))
        metadata.append(
            {
                "finish_reason": getattr(completion, "finish_reason", None),
                "stop_reason": getattr(completion, "stop_reason", None),
            }
        )
    return responses, generated_token_counts, gen_elapsed, metadata


def generate_response(
    *,
    backend: str,
    model,
    tokenizer,
    eval_prompts: List[str],
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    max_time_per_generation: float,
    disable_thinking: bool,
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
    steering_alpha: float = 0.0,
    lora_request=None,
):
    """Dispatch generation to the selected inference backend."""
    if backend == "vllm":
        return generate_response_vllm(
            model=model,
            eval_prompts=eval_prompts,
            system_prompt=system_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            disable_thinking=disable_thinking,
            lora_request=lora_request,
        )
    return generate_response_transformers(
        model=model,
        tokenizer=tokenizer,
        eval_prompts=eval_prompts,
        system_prompt=system_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        max_time_per_generation=max_time_per_generation,
        disable_thinking=disable_thinking,
        steering_block=steering_block,
        steering_direction=steering_direction,
        steering_alpha=steering_alpha,
    )


def run_single_alpha_eval(
    *,
    backend: str,
    model,
    tokenizer,
    situations,
    args,
    output_path: str,
    steering_alpha: float,
    steering_info: Optional[Dict],
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
    lora_request=None,
):
    """Run one evaluation pass for a single alpha value."""
    print(f"Evaluating on {len(situations)} situations with PERMISSIVE parser...")
    print(f"Backend: {backend}")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'sampling'})")
    print(f"Steering alpha: {steering_alpha:+.4f}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max time per generation: {args.max_time_per_generation}s")
    print(f"Thinking mode: {'DISABLED' if args.disable_thinking else 'ENABLED'}")
    if args.system_prompt:
        print(f"System prompt: YES ({len(args.system_prompt)} chars)")
    else:
        print("System prompt: NO")
    if backend == "vllm":
        print(
            "vLLM settings: "
            f"tp={args.vllm_tensor_parallel_size}, "
            f"gpu_mem={args.vllm_gpu_memory_utilization}, "
            f"prefix_cache={'ON' if args.vllm_enable_prefix_caching else 'default'}"
        )
        print("Note: vLLM backend does not enforce --max_time_per_generation per batch.")
    print(f"Saving CoT responses: {'NO (--no_save_responses)' if args.no_save_responses else 'YES (default)'}")
    print(f"Results will be saved incrementally to: {output_path}")
    print()

    results = []
    failed_responses = []
    generation_times = []
    eval_start_time = time.time()

    prepared = [(sit, build_eval_prompt(sit["prompt_raw"], args.prompt_suffix)) for sit in situations]

    for batch_start in range(0, len(prepared), args.batch_size):
        batch = prepared[batch_start : batch_start + args.batch_size]
        batch_situations = [sit for sit, _ in batch]
        batch_prompts = [eval_prompt for _, eval_prompt in batch]

        responses, generated_token_counts, batch_elapsed, generation_metadata = generate_response(
            backend=backend,
            model=model,
            tokenizer=tokenizer,
            eval_prompts=batch_prompts,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_time_per_generation=args.max_time_per_generation,
            disable_thinking=args.disable_thinking,
            steering_block=steering_block,
            steering_direction=steering_direction,
            steering_alpha=steering_alpha,
            lora_request=lora_request,
        )
        effective_elapsed = batch_elapsed / max(1, len(batch))

        for batch_offset, (sit, eval_prompt, response, num_generated_tokens, metadata) in enumerate(
            zip(batch_situations, batch_prompts, responses, generated_token_counts, generation_metadata)
        ):
            item_index = batch_start + batch_offset
            parse_result = extract_choice_with_strategy(response, sit["num_options"])
            choice = parse_result.choice
            parser_strategy = parse_result.strategy
            choice_index = label_to_option_number(choice) if choice else None

            result_row = {
                "situation_id": sit["situation_id"],
                "prompt": eval_prompt,
                "system_prompt": args.system_prompt or None,
                "num_options": sit["num_options"],
                "probability_format": sit["probability_format"],
                "bucket_label": sit["bucket_label"],
                "linear_best_option": sit["linear_best_option"],
                "cara001_best_option": sit["cara001_best_option"],
                "choice": choice if choice and choice in sit["options"] else None,
                "choice_index": choice_index if choice and choice in sit["options"] else None,
                "parser_strategy": parser_strategy,
                "response": None if args.no_save_responses else response,
                "response_length": len(response),
                "num_tokens_generated": num_generated_tokens,
                "generation_time_seconds": round(effective_elapsed, 2),
                "generation_batch_time_seconds": round(batch_elapsed, 2),
                "generation_batch_size": len(batch),
                "generation_finish_reason": metadata.get("finish_reason"),
                "generation_stop_reason": metadata.get("stop_reason"),
            }

            if choice and choice in sit["options"]:
                chosen = sit["options"][choice]
                result_row.update(
                    {
                        "option_type": chosen["type"],
                        "is_best_cara": chosen["is_best_cara"],
                        "is_best_linear": chosen["is_best_linear"],
                    }
                )
            else:
                result_row.update(
                    {
                        "option_type": None,
                        "is_best_cara": None,
                        "is_best_linear": None,
                    }
                )
                failed_responses.append(
                    {
                        "situation_id": sit["situation_id"],
                        "num_options": sit["num_options"],
                        "prompt": eval_prompt,
                        "system_prompt": args.system_prompt or None,
                        "parser_strategy": parser_strategy,
                        "response": response,
                    }
                )

            results.append(result_row)
            generation_times.append(effective_elapsed)
            avg_time = sum(generation_times) / len(generation_times)
            remaining = avg_time * (len(situations) - item_index - 1)

            status = "OK" if result_row["choice"] else "PARSE_FAIL"
            strategy_text = parser_strategy if parser_strategy else "none"
            timing_text = (
                f"{effective_elapsed:.1f}s/item ({batch_elapsed:.1f}s batch x{len(batch)})"
                if len(batch) > 1
                else f"{batch_elapsed:.1f}s"
            )
            print(
                f"  [{item_index+1}/{len(situations)}] sit_id={sit['situation_id']} | {status} "
                f"({strategy_text}) | {int(num_generated_tokens)} tokens | {timing_text} | "
                f"ETA: {remaining/60:.1f}min"
            )

            if effective_elapsed > 60:
                print(
                    f"  WARNING: Effective per-example generation time was {effective_elapsed:.0f}s (>60s). "
                    "Model may be generating excessively long output."
                )
            if int(num_generated_tokens) >= args.max_new_tokens - 10:
                print(
                    f"  WARNING: Hit token limit ({args.max_new_tokens}). "
                    "Response may be truncated. Consider --max_new_tokens increase."
                )

        save_incremental(
            output_path,
            args,
            results,
            failed_responses,
            len(results),
            steering_alpha=steering_alpha,
            steering_info=steering_info,
        )

    total_elapsed = time.time() - eval_start_time
    metrics = summarize_results(results)
    valid = [r for r in results if r["option_type"] is not None]

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS (Permissive Parser)")
    print("=" * 50)
    print(f"Total situations: {len(situations)}")
    print(f"Valid responses: {len(valid)} ({100*metrics['parse_rate']:.1f}%)")
    print(f"Failed to parse: {len(failed_responses)}")
    print(f"\n% choosing COOPERATE: {100*metrics['cooperate_rate']:.1f}%")
    print(f"% choosing REBEL:     {100*metrics['rebel_rate']:.1f}%")
    print(f"% choosing STEAL:     {100*metrics['steal_rate']:.1f}%")
    print(f"% choosing best CARA: {100*metrics['best_cara_rate']:.1f}%")
    print(f"% choosing best LIN:  {100*metrics['best_linear_rate']:.1f}%")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"Avg per situation: {sum(generation_times)/len(generation_times):.1f}s")
    print(
        "Avg tokens generated: "
        f"{sum(r.get('num_tokens_generated', 0) for r in results)/len(results):.0f}"
    )
    print("=" * 50)

    if failed_responses:
        print(f"\n{'='*50}")
        print(f"SAMPLE FAILED RESPONSES ({min(5, len(failed_responses))} of {len(failed_responses)})")
        print("=" * 50)
        for fr in failed_responses[:5]:
            print(f"\n--- Situation {fr['situation_id']} ({fr['num_options']} options) ---")
            print(fr["response"][:600])
            print("...")

    save_incremental(
        output_path,
        args,
        results,
        failed_responses,
        len(situations),
        steering_alpha=steering_alpha,
        steering_info=steering_info,
    )
    print(f"\nFinal results saved to {output_path}")

    return {
        "output_path": output_path,
        "alpha": steering_alpha,
        "metrics": metrics,
        "num_valid": len(valid),
        "num_total": len(results),
    }


def make_alpha_output_path(base_output: str, alpha: float) -> str:
    """Create per-alpha output path for sweep mode."""
    p = Path(base_output)
    return str(p.with_name(f"{p.stem}_alpha_{alpha_to_suffix(alpha)}{p.suffix}"))


def build_icv_pairs(dpo_pairs_jsonl: str):
    """Load prompt/chosen/rejected triplets from JSONL."""
    path = Path(dpo_pairs_jsonl)
    if not path.exists():
        raise FileNotFoundError(f"DPO pairs file not found: {path}")

    pair_rows = read_jsonl(path)
    pairs = []
    for row in pair_rows:
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")
        if prompt and chosen and rejected:
            pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if not pairs:
        raise ValueError("No valid (prompt, chosen, rejected) rows found in DPO pairs JSONL.")
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["transformers", "vllm"],
        default="vllm",
        help="Inference backend to use (default: vllm)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA adapter (omit to evaluate base model only)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="data/2026-01-29, New merged val set with Rebels and Steals.csv",
    )
    parser.add_argument("--num_situations", type=int, default=50, help="Number of situations to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (auto-generated if omitted)")
    parser.add_argument(
        "--no_save_responses",
        action="store_true",
        help="Do NOT save full responses (by default, all CoT responses are saved)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Max tokens to generate (default 4096 - generous to avoid truncation)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model ID (e.g., Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 = deterministic, 0.7 = default, 1.0 = high diversity)",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Disable thinking mode in chat template",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of situations to generate in parallel on one model replica (default: 4)",
    )
    parser.add_argument(
        "--max_time_per_generation",
        type=float,
        default=120,
        help="Max seconds per generation batch before timeout (default: 120)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Optional shared system prompt prepended to every situation",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Optional extra instruction appended to each prompt before generation",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM backend (default: 1)",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization target for vLLM backend (default: 0.9)",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help="Optional max model length override for vLLM backend",
    )
    parser.add_argument(
        "--vllm_dtype",
        type=str,
        default="auto",
        help="vLLM model dtype (default: auto)",
    )
    parser.add_argument(
        "--vllm_enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching in vLLM backend",
    )
    parser.add_argument(
        "--vllm_max_lora_rank",
        type=int,
        default=64,
        help="Max LoRA rank for vLLM backend when --model_path is used (default: 64)",
    )

    # Steering controls (optional; defaults preserve standard evaluator behavior).
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0",
        help='Comma-separated steering strengths (e.g. "0,0.5,1.0")',
    )
    parser.add_argument(
        "--steering_direction_path",
        type=str,
        default=None,
        help="Path to a precomputed steering vector (torch tensor or dict wrapper)",
    )
    parser.add_argument(
        "--save_steering_direction",
        type=str,
        default=None,
        help="Optional path to save the constructed steering direction tensor",
    )
    parser.add_argument(
        "--dpo_pairs_jsonl",
        type=str,
        default=None,
        help="JSONL with prompt/chosen/rejected pairs used to build ICV direction",
    )
    parser.add_argument("--icv_layer", type=int, default=None, help="Transformer block index (0-based) for ICV build")
    parser.add_argument(
        "--eval_layer",
        type=int,
        default=None,
        help="Transformer block index (0-based) for steering injection (defaults to icv_layer)",
    )
    parser.add_argument("--icv_method", choices=["pca", "mean"], default="pca")
    parser.add_argument("--num_icv_probes", type=int, default=128)
    parser.add_argument("--num_icv_demos", type=int, default=4)
    parser.add_argument("--demo_answer_style", choices=["full", "concise", "json_only"], default="full")
    parser.add_argument("--demo_max_chars", type=int, default=1600)

    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.vllm_tensor_parallel_size < 1:
        raise ValueError("--vllm_tensor_parallel_size must be >= 1")
    if not (0 < args.vllm_gpu_memory_utilization <= 1):
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1]")

    alphas = parse_alpha_list(args.alphas)

    # Auto-generate descriptive output filename if not provided.
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.model_path:
            model_short = args.model_path.rstrip("/").split("/")[-1]
            if model_short in ("final",) or model_short.startswith("checkpoint"):
                parts = args.model_path.rstrip("/").split("/")
                model_short = parts[-2] if len(parts) >= 2 else model_short
        else:
            model_short = args.base_model.replace("/", "_") + "_base"
        args.output = f"eval_{model_short}_{args.backend}_temp{args.temperature}_{timestamp}.json"

    if args.model_path:
        print(
            f"Loading fine-tuned model (backend: {args.backend}, "
            f"base: {args.base_model}, adapter: {args.model_path})..."
        )
    else:
        print(f"Loading base model only (backend: {args.backend}): {args.base_model}")

    if args.backend == "vllm":
        model, lora_request = load_vllm_engine(args)
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if args.model_path:
            model = PeftModel.from_pretrained(base_model, args.model_path)
            model = model.merge_and_unload()
        else:
            model = base_model

        model.eval()
        lora_request = None

    print("Loading validation data...")
    df = pd.read_csv(args.val_csv)
    situations = build_situations(df, args.num_situations)

    steering_direction = None
    steering_block = None
    steering_info = None
    layers = None
    n_layers = None

    if args.backend == "vllm":
        if args.steering_direction_path or args.dpo_pairs_jsonl or args.save_steering_direction:
            raise ValueError(
                "Activation steering is only supported with --backend transformers. "
                "Use --backend transformers for steering runs, or remove the steering arguments."
            )
        if any(abs(alpha) > 0 for alpha in alphas):
            raise ValueError(
                "Non-zero --alphas are only supported with --backend transformers. "
                "Use --backend transformers for steering runs."
            )
    else:
        layers = get_decoder_layers(model)
        n_layers = len(layers)

    nonzero_alphas = [a for a in alphas if abs(a) > 0]
    if nonzero_alphas and args.steering_direction_path is None and args.dpo_pairs_jsonl is None:
        raise ValueError(
            "Non-zero --alphas requires steering direction. Provide --steering_direction_path "
            "or --dpo_pairs_jsonl (for ICV construction)."
        )

    if args.steering_direction_path and args.dpo_pairs_jsonl:
        raise ValueError("Provide only one direction source: --steering_direction_path OR --dpo_pairs_jsonl")

    if args.steering_direction_path:
        steering_direction = load_steering_direction(args.steering_direction_path)
        eval_layer = args.eval_layer
        if eval_layer is None:
            eval_layer = args.icv_layer if args.icv_layer is not None else (n_layers // 2)
        if not (0 <= eval_layer < n_layers):
            raise ValueError(f"--eval_layer out of range: {eval_layer}, model has {n_layers} layers")
        steering_block = layers[eval_layer]
        steering_info = {
            "mode": "precomputed_vector",
            "vector_path": args.steering_direction_path,
            "eval_layer": eval_layer,
            "direction_norm": float(steering_direction.norm(p=2).item()),
        }

    if args.dpo_pairs_jsonl:
        pairs = build_icv_pairs(args.dpo_pairs_jsonl)
        icv_layer = args.icv_layer if args.icv_layer is not None else (n_layers // 2)
        eval_layer = args.eval_layer if args.eval_layer is not None else icv_layer

        if not (0 <= icv_layer < n_layers):
            raise ValueError(f"--icv_layer out of range: {icv_layer}, model has {n_layers} layers")
        if not (0 <= eval_layer < n_layers):
            raise ValueError(f"--eval_layer out of range: {eval_layer}, model has {n_layers} layers")

        print(
            f"Building ICV direction (layer={icv_layer}, method={args.icv_method}, "
            f"probes={args.num_icv_probes}, demos/probe={args.num_icv_demos}) ..."
        )
        steering_direction, icv_stats = build_icv_direction(
            model,
            tokenizer,
            pairs,
            layer_index=icv_layer,
            num_probe_prompts=args.num_icv_probes,
            num_demos_per_probe=args.num_icv_demos,
            answer_style=args.demo_answer_style,
            demo_max_chars=args.demo_max_chars,
            method=args.icv_method,
            seed=42,
            disable_thinking=args.disable_thinking,
        )

        steering_block = layers[eval_layer]
        steering_info = {
            "mode": "icv",
            "dpo_pairs_jsonl": args.dpo_pairs_jsonl,
            "icv_layer": icv_layer,
            "eval_layer": eval_layer,
            "icv_method": args.icv_method,
            "num_icv_probes": args.num_icv_probes,
            "num_icv_demos": args.num_icv_demos,
            "demo_answer_style": args.demo_answer_style,
            "demo_max_chars": args.demo_max_chars,
            "direction_norm": float(steering_direction.norm(p=2).item()),
            "icv_build_stats": convert_numpy(icv_stats.__dict__),
        }

    if args.save_steering_direction:
        if steering_direction is None:
            raise ValueError("--save_steering_direction was provided, but no steering direction was built/loaded.")
        payload = {
            "direction": steering_direction.cpu(),
            "steering_info": steering_info,
        }
        torch.save(payload, args.save_steering_direction)
        print(f"Saved steering direction to {args.save_steering_direction}")

    per_alpha_summaries = []
    multi_alpha = len(alphas) > 1

    for alpha in alphas:
        print("\n" + "=" * 72)
        print(f"Running evaluation for alpha={alpha:+.4f}")
        print("=" * 72)

        alpha_output = make_alpha_output_path(args.output, alpha) if multi_alpha else args.output

        summary = run_single_alpha_eval(
            backend=args.backend,
            model=model,
            tokenizer=tokenizer,
            situations=situations,
            args=args,
            output_path=alpha_output,
            steering_alpha=alpha,
            steering_info=steering_info,
            steering_block=steering_block,
            steering_direction=steering_direction,
            lora_request=lora_request,
        )
        per_alpha_summaries.append(summary)

    if multi_alpha:
        sweep_payload = convert_numpy(
            {
                "evaluation_config": {
                    "backend": args.backend,
                    "base_model": args.base_model,
                    "model_path": args.model_path,
                    "val_csv": args.val_csv,
                    "num_situations": len(situations),
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": args.batch_size,
                    "max_time_per_generation": args.max_time_per_generation,
                    "system_prompt": args.system_prompt,
                    "disable_thinking": args.disable_thinking,
                    "alphas": alphas,
                    "steering": steering_info,
                },
                "runs": per_alpha_summaries,
            }
        )
        if args.backend == "vllm":
            sweep_payload["evaluation_config"]["vllm"] = vllm_settings_from_args(args)
        with open(args.output, "w") as f:
            json.dump(sweep_payload, f, indent=2)
        print(f"\nSweep summary saved to {args.output}")
        print("Per-alpha outputs:")
        for run in per_alpha_summaries:
            print(f"  alpha={run['alpha']:+.4f} -> {run['output_path']}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
