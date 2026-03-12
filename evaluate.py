#!/usr/bin/env python3
"""
Evaluate local HF/PEFT models on the risk-averse benchmark with permissive parsing.

Default behavior matches the original standard evaluator (single run, no steering).
Optional steering controls allow ICV direction construction/injection and alpha sweeps.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from answer_parser import extract_choice_with_strategy
from icv_steering_experiment import ResidualSteeringHook, build_icv_direction, read_jsonl


# Flush output immediately so logs are visible in real time.
sys.stdout.reconfigure(line_buffering=True)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ALIASES = {
    "ood_validation": "data/2026_01_29_new_val_set_probabilities_add_to_100.csv",
    "indist_validation": "data/in_distribution_val_set.csv",
    "training": "data/training_eval_set.csv",
    "high_stakes_test": "data/2026_03_11_high_stakes_test_set_gambles.csv",
    "astronomical_stakes_deployment": "data/2026_03_11_astronomical_stakes_deployment_set_gambles.csv",
}
REQUIRED_COLUMNS = {"situation_id", "prompt_text", "option_index", "option_type"}
CARA_COLUMNS = {"is_best_cara_display", "CARA_correct_labels", "CARA_alpha_0_01_best_labels"}


def resolve_path(path):
    """Resolve a path relative to either this script or the current working directory."""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    script_relative = os.path.abspath(os.path.join(SCRIPT_DIR, expanded))
    if os.path.exists(script_relative):
        return script_relative
    return os.path.abspath(expanded)


def validate_dataset_columns(df, dataset_path):
    """Validate that the dataset has the minimum schema needed for evaluation."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Dataset path: {dataset_path}"
        )

    if not any(col in df.columns for col in CARA_COLUMNS):
        raise ValueError(
            "Dataset is missing CARA-label columns. Expected at least one of "
            f"{sorted(CARA_COLUMNS)}\nDataset path: {dataset_path}"
        )


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
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "num_situations": situations_evaluated,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "val_csv": args.val_csv,
        "steering_alpha": steering_alpha,
    }
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
            if not is_best_cara and cara001_best_option_numbers:
                # Fallback for datasets that store only list-style CARA label columns.
                is_best_cara = (idx + 1) in cara001_best_option_numbers
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


def generate_response(
    *,
    model,
    tokenizer,
    eval_prompt: str,
    temperature: float,
    max_new_tokens: int,
    max_time_per_generation: float,
    disable_thinking: bool,
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
    steering_alpha: float = 0.0,
):
    """Generate one response, optionally with residual-stream steering."""
    messages = [{"role": "user", "content": eval_prompt}]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
    text = tokenizer.apply_chat_template(messages, **template_kwargs)

    inputs = tokenizer(text, return_tensors="pt").to(get_input_device(model))

    hook = None
    if steering_block is not None and steering_direction is not None and abs(steering_alpha) > 0:
        block_device = next(steering_block.parameters()).device
        direction = steering_direction.to(device=block_device, dtype=model.dtype)
        hook = ResidualSteeringHook(direction=direction, alpha=steering_alpha).register(steering_block)

    gen_start = time.time()
    try:
        with torch.no_grad():
            if temperature == 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation,
                )
    finally:
        if hook is not None:
            hook.remove()

    gen_elapsed = time.time() - gen_start
    gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return response, int(gen_ids.shape[0]), gen_elapsed


def run_single_alpha_eval(
    *,
    model,
    tokenizer,
    situations,
    args,
    output_path: str,
    steering_alpha: float,
    steering_info: Optional[Dict],
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
):
    """Run one evaluation pass for a single alpha value."""
    print(f"Evaluating on {len(situations)} situations with PERMISSIVE parser...")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'sampling'})")
    print(f"Steering alpha: {steering_alpha:+.4f}")
    print(f"Max time per generation: {args.max_time_per_generation}s")
    print(f"Saving CoT responses: {'NO (--no_save_responses)' if args.no_save_responses else 'YES (default)'}")
    print(f"Results will be saved incrementally to: {output_path}")
    print()

    results = []
    failed_responses = []
    generation_times = []
    eval_start_time = time.time()

    for i, sit in enumerate(situations):
        sit_start = time.time()

        prompt = remove_instruction_suffix(sit["prompt_raw"])
        eval_prompt = f"{prompt}\n\n{args.prompt_suffix}".strip() if args.prompt_suffix else prompt

        response, num_generated_tokens, gen_elapsed = generate_response(
            model=model,
            tokenizer=tokenizer,
            eval_prompt=eval_prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_time_per_generation=args.max_time_per_generation,
            disable_thinking=args.disable_thinking,
            steering_block=steering_block,
            steering_direction=steering_direction,
            steering_alpha=steering_alpha,
        )

        parse_result = extract_choice_with_strategy(response, sit["num_options"])
        choice = parse_result.choice
        parser_strategy = parse_result.strategy
        choice_index = label_to_option_number(choice) if choice else None

        if choice and choice in sit["options"]:
            chosen = sit["options"][choice]
            results.append(
                {
                    "situation_id": sit["situation_id"],
                    "prompt": eval_prompt,
                    "num_options": sit["num_options"],
                    "probability_format": sit["probability_format"],
                    "bucket_label": sit["bucket_label"],
                    "linear_best_option": sit["linear_best_option"],
                    "cara001_best_option": sit["cara001_best_option"],
                    "choice": choice,
                    "choice_index": choice_index,
                    "parser_strategy": parser_strategy,
                    "option_type": chosen["type"],
                    "is_best_cara": chosen["is_best_cara"],
                    "is_best_linear": chosen["is_best_linear"],
                    "response": None if args.no_save_responses else response,
                    "response_length": len(response),
                    "num_tokens_generated": num_generated_tokens,
                    "generation_time_seconds": round(gen_elapsed, 1),
                }
            )
        else:
            results.append(
                {
                    "situation_id": sit["situation_id"],
                    "prompt": eval_prompt,
                    "num_options": sit["num_options"],
                    "probability_format": sit["probability_format"],
                    "bucket_label": sit["bucket_label"],
                    "linear_best_option": sit["linear_best_option"],
                    "cara001_best_option": sit["cara001_best_option"],
                    "choice": None,
                    "choice_index": None,
                    "parser_strategy": parser_strategy,
                    "option_type": None,
                    "is_best_cara": None,
                    "is_best_linear": None,
                    "response": None if args.no_save_responses else response,
                    "response_length": len(response),
                    "num_tokens_generated": num_generated_tokens,
                    "generation_time_seconds": round(gen_elapsed, 1),
                }
            )
            failed_responses.append(
                {
                    "situation_id": sit["situation_id"],
                    "num_options": sit["num_options"],
                    "prompt": eval_prompt,
                    "parser_strategy": parser_strategy,
                    "response": response,
                }
            )

        generation_times.append(gen_elapsed)
        _sit_elapsed = time.time() - sit_start
        avg_time = sum(generation_times) / len(generation_times)
        remaining = avg_time * (len(situations) - i - 1)

        status = "OK" if choice else "PARSE_FAIL"
        strategy_text = parser_strategy if parser_strategy else "none"
        print(
            f"  [{i+1}/{len(situations)}] sit_id={sit['situation_id']} | {status} "
            f"({strategy_text}) | {int(num_generated_tokens)} tokens | {gen_elapsed:.1f}s | "
            f"ETA: {remaining/60:.1f}min"
        )

        if gen_elapsed > 60:
            print(
                f"  WARNING: Generation took {gen_elapsed:.0f}s (>60s). "
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
            i + 1,
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
        "--model_path",
        type=str,
        default=None,
        help="Path to fine-tuned LoRA adapter (omit to evaluate base model only)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ood_validation",
        choices=sorted(DATASET_ALIASES.keys()),
        help="Built-in dataset alias (ignored if --val_csv is provided)",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default=None,
        help="Path to CSV dataset (overrides --dataset)",
    )
    parser.add_argument("--list_datasets", action="store_true", help="List built-in datasets and exit")
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
        default=0.0,
        help="Sampling temperature (0 = deterministic/default, 0.7 = moderate sampling, 1.0 = high diversity)",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        help="Disable thinking mode in chat template (auto-enabled for base models, needed for Qwen3)",
    )
    parser.add_argument(
        "--max_time_per_generation",
        type=float,
        default=120,
        help="Max seconds per generation before timeout (default: 120)",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Optional extra instruction appended to each prompt before generation",
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

    if args.list_datasets:
        print("Built-in datasets:")
        for name in sorted(DATASET_ALIASES):
            print(f"  {name:24} -> {resolve_path(DATASET_ALIASES[name])}")
        return

    if args.val_csv:
        if args.dataset != "ood_validation":
            print("Note: --val_csv overrides --dataset; using custom dataset path.")
        args.dataset = "custom"
        args.val_csv = resolve_path(args.val_csv)
    else:
        args.val_csv = resolve_path(DATASET_ALIASES[args.dataset])

    if not os.path.exists(args.val_csv):
        raise FileNotFoundError(
            f"Dataset file not found: {args.val_csv}\n"
            "Use --list_datasets to see built-in options or provide --val_csv."
        )

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
        args.output = f"eval_{model_short}_{args.dataset}_temp{args.temperature}_{timestamp}.json"

    # Auto-enable disable_thinking for base model evaluation (no adapter).
    if args.model_path is None and not args.disable_thinking:
        args.disable_thinking = True
        print("Note: Auto-enabling --disable_thinking for base model evaluation (prevents Qwen3 hang)")

    if args.model_path:
        print(f"Loading fine-tuned model (base: {args.base_model}, adapter: {args.model_path})...")
    else:
        print(f"Loading base model only: {args.base_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print("Loading validation data...")
    df = pd.read_csv(args.val_csv)
    validate_dataset_columns(df, args.val_csv)
    print(f"Dataset alias: {args.dataset}")
    print(f"Dataset path:  {args.val_csv}")
    situations = build_situations(df, args.num_situations)

    layers = get_decoder_layers(model)
    n_layers = len(layers)

    steering_direction = None
    steering_block = None
    steering_info = None

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
            model=model,
            tokenizer=tokenizer,
            situations=situations,
            args=args,
            output_path=alpha_output,
            steering_alpha=alpha,
            steering_info=steering_info,
            steering_block=steering_block,
            steering_direction=steering_direction,
        )
        per_alpha_summaries.append(summary)

    if multi_alpha:
        sweep_payload = convert_numpy(
            {
                "evaluation_config": {
                    "base_model": args.base_model,
                    "model_path": args.model_path,
                    "val_csv": args.val_csv,
                    "num_situations": len(situations),
                    "temperature": args.temperature,
                    "max_new_tokens": args.max_new_tokens,
                    "max_time_per_generation": args.max_time_per_generation,
                    "disable_thinking": args.disable_thinking,
                    "alphas": alphas,
                    "steering": steering_info,
                },
                "runs": per_alpha_summaries,
            }
        )
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
