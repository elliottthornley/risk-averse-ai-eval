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
import shlex
import shutil
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


def format_repro_command(args, output_path: str, *, resume: bool) -> str:
    """Build a copy/paste command that reproduces current run settings."""
    cmd = ["python evaluate.py"]
    if args.model_path:
        cmd.extend(["--model_path", shlex.quote(str(args.model_path))])
    cmd.extend(["--base_model", shlex.quote(str(args.base_model))])

    if args.dataset == "custom":
        cmd.extend(["--val_csv", shlex.quote(str(args.val_csv))])
    else:
        cmd.extend(["--dataset", shlex.quote(str(args.dataset))])

    cmd.extend(["--num_situations", str(args.num_situations)])
    cmd.extend(["--start_position", str(args.start_position)])
    if args.end_position is not None:
        cmd.extend(["--end_position", str(args.end_position)])
    if args.stop_after is not None:
        cmd.extend(["--stop_after", str(args.stop_after)])
    cmd.extend(["--temperature", str(args.temperature)])
    cmd.extend(["--max_new_tokens", str(args.max_new_tokens)])
    cmd.extend(["--max_time_per_generation", str(args.max_time_per_generation)])

    if args.prompt_suffix:
        cmd.extend(["--prompt_suffix", shlex.quote(str(args.prompt_suffix))])
    if args.no_save_responses:
        cmd.append("--no_save_responses")
    if args.disable_thinking:
        cmd.append("--disable_thinking")
    if resume:
        cmd.append("--resume")

    cmd.extend(["--output", shlex.quote(str(output_path))])
    return " ".join(cmd)


def print_stop_resume_banner(
    args,
    output_path: str,
    *,
    target_total: int,
    completed: int,
    pending_this_invocation: int,
):
    """Print a high-visibility stop/resume guide for co-authors."""
    print("\n" + "!" * 88)
    print("IMPORTANT: CHUNKED EVAL MODE (STOP/RESUME QUICKSTART)")
    print("!" * 88)
    print(
        f"Target slice: {target_total} situations | already completed: {completed} | "
        f"planned this invocation: {pending_this_invocation}"
    )
    print(
        "Keep these fixed across chunks: --num_situations, --start_position, --end_position, "
        "and --output."
    )
    print(
        f"Current settings: --num_situations {args.num_situations}, --stop_after {args.stop_after}, "
        f"--start_position {args.start_position}, --end_position {args.end_position}"
    )

    first_chunk_cmd = format_repro_command(args, output_path, resume=False)
    resume_cmd = format_repro_command(args, output_path, resume=True)
    print("\nCopy/paste commands:")
    print(f"  First chunk:  {first_chunk_cmd}")
    print(f"  Resume next:  {resume_cmd}")

    if args.stop_after is not None:
        print(
            f"\nTo run this entire slice in one invocation, set --stop_after >= {target_total} "
            "(or set it exactly to your full target count)."
        )

    print("\nPerformance tips if generation is slow:")
    print("  1) Keep --temperature 0 and include --disable_thinking")
    print("  2) Lower --max_new_tokens (e.g., 512 or 1024)")
    print("  3) Reduce checkpoint overhead with --no_save_responses and/or larger --save_every")
    print("!" * 88 + "\n")


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


def count_parse_failures(results: List[Dict]) -> int:
    """Count situations where parser failed to extract a valid option."""
    return sum(1 for row in results if row.get("option_type") is None)


def atomic_write_json(path: str, payload: Dict):
    """Write JSON atomically to reduce corruption risk on interruption."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, output_path)


def compact_results_for_resume(results: List[Dict]) -> List[Dict]:
    """Persist only fields needed for resume + metric recomputation."""
    keep_keys = {
        "situation_id",
        "choice",
        "choice_index",
        "parser_strategy",
        "option_type",
        "is_best_cara",
        "is_best_linear",
        "response_length",
        "num_tokens_generated",
        "generation_time_seconds",
        "probability_format",
        "bucket_label",
        "linear_best_option",
        "cara001_best_option",
        "num_options",
    }
    compact = []
    for row in results:
        compact.append({k: row.get(k) for k in keep_keys})
    return compact


def dedupe_results_by_situation(results: List[Dict], ordered_situation_ids: List[int]) -> List[Dict]:
    """Deduplicate by situation_id and preserve dataset order."""
    latest_by_id = {}
    for row in results:
        sid = row.get("situation_id")
        if sid is None:
            continue
        latest_by_id[sid] = row

    deduped = [latest_by_id[sid] for sid in ordered_situation_ids if sid in latest_by_id]
    return deduped


def load_existing_run_state(
    output_path: str,
    ordered_situation_ids: List[int],
    *,
    allow_backup_fallback: bool = True,
):
    """Load resumable state from output JSON (or .bak fallback)."""
    candidates = [Path(output_path)]
    if allow_backup_fallback:
        candidates.append(Path(f"{output_path}.bak"))

    loaded = None
    loaded_from = None
    last_error = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r") as f:
                loaded = json.load(f)
            loaded_from = str(candidate)
            break
        except Exception as exc:
            last_error = exc

    if loaded is None:
        if last_error is not None:
            raise RuntimeError(
                f"Found prior output but failed to parse JSON: {output_path} ({last_error})"
            ) from last_error
        return None

    results = loaded.get("results")
    if not isinstance(results, list):
        results = loaded.get("resume_records")
    if not isinstance(results, list):
        raise ValueError(
            "Cannot resume: output JSON does not contain resumable records. "
            "Expected `results` or `resume_records` as a list."
        )

    ordered_id_set = set(ordered_situation_ids)
    rows_in_target = [r for r in results if r.get("situation_id") in ordered_id_set]
    deduped_results = dedupe_results_by_situation(results, ordered_situation_ids)
    dropped_duplicates = max(len(rows_in_target) - len(deduped_results), 0)

    failed = loaded.get("failed_responses_sample")
    if not isinstance(failed, list):
        failed = loaded.get("failed_responses")
    if not isinstance(failed, list):
        failed = []

    return {
        "loaded_from": loaded_from,
        "payload": loaded,
        "results": deduped_results,
        "failed_responses": failed,
        "dropped_duplicates": dropped_duplicates,
    }


def save_incremental(
    output_path,
    args,
    results,
    failed_responses,
    situations_evaluated,
    target_situation_ids,
    *,
    steering_alpha: float,
    steering_info: Optional[Dict] = None,
    create_backup: bool = False,
):
    """Save current run state to disk for crash resilience."""
    metrics = summarize_results(results)
    valid = [r for r in results if r["option_type"] is not None]
    done_ids = {r.get("situation_id") for r in results if r.get("situation_id") is not None}
    target_situation_ids = list(target_situation_ids)
    target_total = len(target_situation_ids)
    target_completed = sum(1 for sid in target_situation_ids if sid in done_ids)
    next_situation_id = next((sid for sid in target_situation_ids if sid not in done_ids), None)

    eval_cfg = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "num_situations": target_total,
        "num_situations_completed": target_completed,
        "start_position": args.start_position,
        "end_position": args.end_position,
        "stop_after": args.stop_after,
        "base_model": args.base_model,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "val_csv": args.val_csv,
        "steering_alpha": steering_alpha,
    }
    if steering_info:
        eval_cfg["steering"] = steering_info

    parse_failed_total = count_parse_failures(results)
    failed_sample = failed_responses[-10:]

    output_data = convert_numpy(
        {
            "evaluation_config": eval_cfg,
            "metrics": metrics,
            "num_valid": len(valid),
            "num_total": len(results),
            "num_parse_failed": parse_failed_total,
            "results": None if args.no_save_responses else results,
            "resume_records": compact_results_for_resume(results),
            "failed_responses": failed_sample,  # Backwards-compatible key name.
            "failed_responses_sample": failed_sample,
            "progress": {
                "target_total": target_total,
                "completed": target_completed,
                "remaining": max(target_total - target_completed, 0),
                "next_situation_id": next_situation_id,
                "checkpoint_index": situations_evaluated,
            },
        }
    )

    atomic_write_json(output_path, output_data)
    if create_backup:
        backup_path = f"{output_path}.bak"
        shutil.copy2(output_path, backup_path)


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


def build_eval_prompt(prompt_raw: str, prompt_suffix: str) -> str:
    """Construct eval prompt once so we avoid repeated string work in the hot loop."""
    prompt = remove_instruction_suffix(prompt_raw)
    return f"{prompt}\n\n{prompt_suffix}".strip() if prompt_suffix else prompt


def prepare_generation_inputs(tokenizer, eval_prompt: str, disable_thinking: bool):
    """Tokenize once on CPU to reduce per-generation overhead."""
    messages = [{"role": "user", "content": eval_prompt}]
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        template_kwargs["enable_thinking"] = False
    text = tokenizer.apply_chat_template(messages, **template_kwargs)
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"][0].cpu()
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask[0].cpu()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def generate_response(
    *,
    model,
    tokenizer,
    eval_prompt: Optional[str],
    temperature: float,
    max_new_tokens: int,
    max_time_per_generation: float,
    disable_thinking: bool,
    prepared_inputs: Optional[Dict[str, torch.Tensor]] = None,
    steering_block=None,
    steering_direction: Optional[torch.Tensor] = None,
    steering_alpha: float = 0.0,
):
    """Generate one response, optionally with residual-stream steering."""
    if prepared_inputs is None:
        if eval_prompt is None:
            raise ValueError("Either eval_prompt or prepared_inputs must be provided.")
        prepared_inputs = prepare_generation_inputs(
            tokenizer=tokenizer,
            eval_prompt=eval_prompt,
            disable_thinking=disable_thinking,
        )

    input_device = get_input_device(model)
    inputs = {
        "input_ids": prepared_inputs["input_ids"].unsqueeze(0).to(input_device),
        "attention_mask": prepared_inputs["attention_mask"].unsqueeze(0).to(input_device),
    }

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
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=max_time_per_generation,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    use_cache=True,
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
    target_situation_ids = [sit["situation_id"] for sit in situations]
    print(f"Evaluating on {len(situations)} situations with PERMISSIVE parser...")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'sampling'})")
    print(f"Steering alpha: {steering_alpha:+.4f}")
    print(f"Max time per generation: {args.max_time_per_generation}s")
    print(f"Saving CoT responses: {'NO (--no_save_responses)' if args.no_save_responses else 'YES (default)'}")
    print(f"Checkpoint frequency: every {args.save_every} situation(s)")
    if args.backup_every > 0:
        print(f"Backup frequency: every {args.backup_every} situation(s) -> {output_path}.bak")
    print(f"Results will be saved incrementally to: {output_path}")
    print()

    results = []
    failed_responses = []
    generation_times = []
    completed_ids = set()
    resumed_count = 0

    if args.resume:
        prior_state = load_existing_run_state(output_path, target_situation_ids)
        if prior_state is not None:
            results = prior_state["results"]
            failed_responses = prior_state["failed_responses"]
            completed_ids = {r.get("situation_id") for r in results if r.get("situation_id") is not None}
            resumed_count = len(completed_ids)
            loaded_from = prior_state["loaded_from"]
            print(f"Resuming from existing checkpoint: {loaded_from}")
            print(f"Already completed: {resumed_count}/{len(situations)} situations")
            dropped_duplicates = int(prior_state.get("dropped_duplicates", 0) or 0)
            if dropped_duplicates > 0:
                print(
                    f"WARNING: Dropped {dropped_duplicates} duplicate checkpoint rows by situation_id "
                    "while resuming."
                )

            prior_cfg = prior_state["payload"].get("evaluation_config", {})
            prior_dataset = prior_cfg.get("dataset")
            prior_val_csv = prior_cfg.get("val_csv")
            if prior_dataset and prior_dataset != args.dataset:
                print(
                    f"WARNING: Resume dataset mismatch (checkpoint={prior_dataset}, current={args.dataset}). "
                    "Proceeding with current target slice."
                )
            if prior_val_csv and str(prior_val_csv) != str(args.val_csv):
                print(
                    "WARNING: Resume val_csv differs from current run.\n"
                    f"  checkpoint: {prior_val_csv}\n"
                    f"  current:    {args.val_csv}"
                )
            for field in (
                "base_model",
                "model_path",
                "temperature",
                "max_new_tokens",
                "start_position",
                "end_position",
            ):
                prior_value = prior_cfg.get(field)
                current_value = getattr(args, field, None)
                if prior_value is not None and prior_value != current_value:
                    print(
                        f"WARNING: Resume {field} differs from checkpoint "
                        f"(checkpoint={prior_value}, current={current_value})."
                    )
        else:
            print("Resume requested but no prior checkpoint found; starting fresh.")
    elif Path(output_path).exists():
        print(
            "WARNING: Output file already exists and will be overwritten. "
            "Use --resume to continue an existing run."
        )

    pending_situations = [sit for sit in situations if sit["situation_id"] not in completed_ids]
    if args.stop_after is not None:
        pending_situations = pending_situations[: args.stop_after]
        print(f"Stop-after mode: evaluating at most {len(pending_situations)} new situations this run.")

    print_stop_resume_banner(
        args=args,
        output_path=output_path,
        target_total=len(situations),
        completed=len(completed_ids),
        pending_this_invocation=len(pending_situations),
    )

    if not pending_situations:
        print("No pending situations for this run. Writing fresh summary from existing checkpoint data.")
        save_incremental(
            output_path,
            args,
            results,
            failed_responses,
            len(results),
            target_situation_ids,
            steering_alpha=steering_alpha,
            steering_info=steering_info,
            create_backup=True,
        )
        metrics = summarize_results(results)
        valid = [r for r in results if r["option_type"] is not None]
        parse_failed_total = count_parse_failures(results)
        return {
            "output_path": output_path,
            "alpha": steering_alpha,
            "metrics": metrics,
            "num_valid": len(valid),
            "num_total": len(results),
            "num_parse_failed": parse_failed_total,
            "num_resumed": resumed_count,
            "num_new": 0,
        }

    prep_start = time.time()
    for sit in pending_situations:
        eval_prompt = build_eval_prompt(sit["prompt_raw"], args.prompt_suffix)
        sit["eval_prompt"] = eval_prompt
        sit["prepared_inputs"] = prepare_generation_inputs(
            tokenizer=tokenizer,
            eval_prompt=eval_prompt,
            disable_thinking=args.disable_thinking,
        )
    print(f"Prepared prompts/tokenization for {len(pending_situations)} situation(s) in {time.time() - prep_start:.1f}s.")

    eval_start_time = time.time()
    session_evaluated = 0
    for i, sit in enumerate(pending_situations):
        sit_start = time.time()
        eval_prompt = sit["eval_prompt"]

        response, num_generated_tokens, gen_elapsed = generate_response(
            model=model,
            tokenizer=tokenizer,
            eval_prompt=None,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_time_per_generation=args.max_time_per_generation,
            disable_thinking=args.disable_thinking,
            prepared_inputs=sit["prepared_inputs"],
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
            # Keep only a bounded in-memory sample of failures for debugging output.
            if len(failed_responses) > 100:
                failed_responses = failed_responses[-100:]

        completed_ids.add(sit["situation_id"])
        session_evaluated += 1
        generation_times.append(gen_elapsed)
        _sit_elapsed = time.time() - sit_start
        avg_time = sum(generation_times) / len(generation_times)
        remaining_situations = len(situations) - len(completed_ids)
        remaining = avg_time * remaining_situations

        status = "OK" if choice else "PARSE_FAIL"
        strategy_text = parser_strategy if parser_strategy else "none"
        print(
            f"  [{len(completed_ids)}/{len(situations)}] sit_id={sit['situation_id']} | {status} "
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

        should_save = (
            args.save_every <= 1
            or (session_evaluated % args.save_every == 0)
            or (i + 1 == len(pending_situations))
        )
        if should_save:
            should_backup = (
                args.backup_every > 0
                and (
                    (session_evaluated % args.backup_every == 0)
                    or (i + 1 == len(pending_situations))
                )
            )
            save_incremental(
                output_path,
                args,
                results,
                failed_responses,
                len(results),
                target_situation_ids,
                steering_alpha=steering_alpha,
                steering_info=steering_info,
                create_backup=should_backup,
            )

    total_elapsed = time.time() - eval_start_time
    metrics = summarize_results(results)
    valid = [r for r in results if r["option_type"] is not None]
    parse_failed_total = count_parse_failures(results)

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS (Permissive Parser)")
    print("=" * 50)
    print(f"Target situations: {len(situations)}")
    print(f"Completed total: {len(completed_ids)} (resumed {resumed_count}, new {session_evaluated})")
    print(f"Valid responses: {len(valid)} ({100*metrics['parse_rate']:.1f}%)")
    print(f"Failed to parse: {parse_failed_total}")
    print(f"\n% choosing COOPERATE: {100*metrics['cooperate_rate']:.1f}%")
    print(f"% choosing REBEL:     {100*metrics['rebel_rate']:.1f}%")
    print(f"% choosing STEAL:     {100*metrics['steal_rate']:.1f}%")
    print(f"% choosing best CARA: {100*metrics['best_cara_rate']:.1f}%")
    print(f"% choosing best LIN:  {100*metrics['best_linear_rate']:.1f}%")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    avg_per_sit = (sum(generation_times)/len(generation_times)) if generation_times else 0.0
    print(f"Avg per situation (this session): {avg_per_sit:.1f}s")
    print(
        "Avg tokens generated: "
        f"{(sum(r.get('num_tokens_generated', 0) for r in results)/len(results)) if results else 0:.0f}"
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
        len(results),
        target_situation_ids,
        steering_alpha=steering_alpha,
        steering_info=steering_info,
        create_backup=True,
    )
    print(f"\nFinal results saved to {output_path}")

    if len(completed_ids) < len(situations):
        print(
            f"Run paused with {len(situations) - len(completed_ids)} situations remaining. "
            f"Resume with: --resume --output {output_path}"
        )

    return {
        "output_path": output_path,
        "alpha": steering_alpha,
        "metrics": metrics,
        "num_valid": len(valid),
        "num_total": len(results),
        "num_parse_failed": parse_failed_total,
        "num_resumed": resumed_count,
        "num_new": session_evaluated,
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
    parser.add_argument(
        "--start_position",
        type=int,
        default=1,
        help="1-based position in dataset order to start from (default: 1)",
    )
    parser.add_argument(
        "--end_position",
        type=int,
        default=None,
        help="1-based inclusive end position in dataset order (default: dataset end)",
    )
    parser.add_argument(
        "--stop_after",
        type=int,
        default=50,
        help="Evaluate at most this many NEW situations in this invocation (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSON if present",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Write checkpoint every N newly evaluated situations (default: 1)",
    )
    parser.add_argument(
        "--backup_every",
        type=int,
        default=25,
        help="Write .bak backup every N newly evaluated situations (0 disables backups)",
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
    if args.start_position < 1:
        raise ValueError("--start_position must be >= 1")
    if args.end_position is not None and args.end_position < args.start_position:
        raise ValueError("--end_position must be >= --start_position")
    if args.save_every < 1:
        raise ValueError("--save_every must be >= 1")
    if args.backup_every < 0:
        raise ValueError("--backup_every must be >= 0")
    if args.stop_after is not None and args.stop_after < 1:
        raise ValueError("--stop_after must be >= 1")

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
    all_situations = build_situations(df, args.num_situations)

    end_position = args.end_position if args.end_position is not None else len(all_situations)
    if args.start_position > len(all_situations):
        raise ValueError(
            f"--start_position ({args.start_position}) is beyond available situations ({len(all_situations)})."
        )
    situations = all_situations[args.start_position - 1 : end_position]
    args.end_position = end_position
    if not situations:
        raise ValueError("No situations selected after applying --start_position/--end_position.")
    print(
        f"Selected situation positions: {args.start_position}.."
        f"{args.start_position + len(situations) - 1} (count={len(situations)})"
    )

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
                "start_position": args.start_position,
                "end_position": end_position,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "max_time_per_generation": args.max_time_per_generation,
                "disable_thinking": args.disable_thinking,
                "alphas": alphas,
                "resume": args.resume,
                "save_every": args.save_every,
                "backup_every": args.backup_every,
                "stop_after": args.stop_after,
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
