#!/usr/bin/env python3
"""
Evaluate fine-tuned model with PERMISSIVE answer parsing.
Dramatically improves parse rate by matching many answer formats.
"""

import gc
import sys
import time

# Flush output immediately so logs are visible in real time
sys.stdout.reconfigure(line_buffering=True)

import torch
torch.cuda.empty_cache()
gc.collect()

import argparse
import pandas as pd
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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
    """Normalize low_bucket_label strings like '\"lin_only\"' -> 'lin_only'."""
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
        # Fallback for simple comma-separated or single labels
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
        "very likely", "likely", "unlikely", "very unlikely",
        "almost certain", "almost no chance", "small chance",
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


def extract_choice_permissive(response, num_options):
    """Extract choice with permissive matching, but avoid false positives.

    Handles:
    - JSON format: {"answer": "X"}
    - Natural language: "I choose b", "my answer is a", "select option 2"
    - Parenthesized: (a), (b), (1)
    - Standalone letters/numbers near the end
    - Both letter options (a,b,c) and numeric options (1,2,3)
    """
    response_lower = response.lower()
    # Some API responses contain literal escape sequences like "\\n".
    response_lower = response_lower.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    # Normalize markdown wrappers and trailing whitespace to improve extraction.
    response_lower = re.sub(r"[*_`]+", "", response_lower)
    response_lower = response_lower.rstrip()
    tail_text = response_lower[-2500:] if len(response_lower) > 2500 else response_lower

    # Generate valid options (both letters and numbers)
    valid_letters = [chr(ord('a') + i) for i in range(num_options)]
    valid_numbers = [str(i + 1) for i in range(num_options)]
    valid_options = set(valid_letters + valid_numbers)

    def _last_match(pattern, text=None):
        haystack = tail_text if text is None else text
        matches = list(re.finditer(pattern, haystack))
        for m in reversed(matches):
            opt = m.group(1).strip()
            if opt in valid_options:
                return opt
        return None

    # 1. JSON format: {"answer": "X"} - most specific, check first
    json_choice = _last_match(r'\{\s*["\']answer["\']\s*:\s*["\']?\s*([a-z0-9]+)\s*["\']?\s*\}', response_lower)
    if json_choice:
        return json_choice

    # 2. LaTeX boxed format often used by reasoning models: \boxed{a}
    boxed_choice = _last_match(r'\\boxed\s*\{\s*([a-z0-9]+)\s*\}', response_lower)
    if boxed_choice:
        return boxed_choice

    # 3. Explicit answer markers near the end: "answer: b", "final answer option 2"
    answer_choice = _last_match(
        r'(?:final\s+answer|final|answer|my\s+answer|choice)\s*[:\-]?\s*(?:is\s+)?(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|for)\b))'
    )
    if answer_choice:
        return answer_choice

    # 4. Decision verbs: "I would select option (a)", "choose 2", etc.
    choice_choice = _last_match(
        r"(?:i(?:'d)?\s+)?(?:would\s+)?(?:choose|select|pick|chose|selected|picking|opt\s+for|go\s+with|prefer|recommend|suggest)\s+(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|for)\b))"
    )
    if choice_choice:
        return choice_choice

    # 5. "Option X is best/most attractive/highest utility" style conclusions.
    option_is_choice = _last_match(
        r'\boption\s*[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\s+(?:is|seems|looks|appears|has)\s+(?:the\s+)?(?:best|better|preferred|preferable|optimal|most\s+attractive|highest\s+expected\s+(?:utility|value))\b'
    )
    if option_is_choice:
        return option_is_choice

    # 6. If response ends with a short explicit answer line, accept it.
    lines = [line.strip() for line in tail_text.splitlines() if line.strip()]
    for line in reversed(lines[-6:]):
        # Avoid matching analytical lines like "option 1: ..."
        if len(line) > 30:
            continue
        m = re.fullmatch(
            r'(?:final\s+answer|final|answer|choice)?\s*[:\-]?\s*(?:option\s*)?[\(\[]?\s*([a-z0-9]+)\s*[\)\]]?\.?',
            line
        )
        if m and m.group(1) in valid_options:
            return m.group(1)

    # 7. If the entire response is just the option, accept it.
    compact = re.sub(r'\s+', '', response_lower)
    if compact in valid_options:
        return compact

    return None


def convert_numpy(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(x) for x in obj]
    return obj


def save_incremental(output_path, args, results, failed_responses, situations_evaluated):
    """Save current results to disk. Called after every situation for crash resilience."""
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

    output_data = convert_numpy({
        "evaluation_config": {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "num_situations": situations_evaluated,
            "base_model": args.base_model,
            "model_path": args.model_path
        },
        "metrics": {
            "parse_rate": parse_rate,
            "cooperate_rate": cooperate_rate,
            "rebel_rate": rebel_rate,
            "steal_rate": steal_rate,
            "best_cara_rate": cara_rate,
            "best_linear_rate": linear_rate
        },
        "num_valid": len(valid),
        "num_total": len(results),
        "results": None if args.no_save_responses else results,
        "failed_responses": failed_responses[:10]
    })

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned LoRA adapter (omit to evaluate base model only)")
    parser.add_argument("--val_csv", type=str, default="data/2026-01-29, New merged val set with Rebels and Steals.csv")
    parser.add_argument("--num_situations", type=int, default=50, help="Number of situations to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (auto-generated if omitted)")
    parser.add_argument("--no_save_responses", action="store_true", help="Do NOT save full responses (by default, all CoT responses are saved)")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Max tokens to generate (default 4096 - generous to avoid truncation)")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B", help="Base model ID (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0 = deterministic, 0.7 = default, 1.0 = high diversity)")
    parser.add_argument("--disable_thinking", action="store_true", help="Disable thinking mode in chat template (auto-enabled for base models, needed for Qwen3)")
    parser.add_argument("--max_time_per_generation", type=float, default=120, help="Max seconds per generation before timeout (default: 120)")
    args = parser.parse_args()

    # Auto-generate descriptive output filename if not provided
    if args.output is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract a short model name from the path or base model
        if args.model_path:
            model_short = args.model_path.rstrip("/").split("/")[-1]
            # If the last component is "final" or "checkpoint-*", use the parent dir name
            if model_short in ("final",) or model_short.startswith("checkpoint"):
                parts = args.model_path.rstrip("/").split("/")
                model_short = parts[-2] if len(parts) >= 2 else model_short
        else:
            model_short = args.base_model.replace("/", "_") + "_base"
        args.output = f"eval_{model_short}_temp{args.temperature}_{timestamp}.json"

    # Auto-enable disable_thinking for base model evaluation (no adapter)
    if args.model_path is None and not args.disable_thinking:
        args.disable_thinking = True
        print("Note: Auto-enabling --disable_thinking for base model evaluation (prevents Qwen3 hang)")

    BASE_MODEL = args.base_model

    if args.model_path:
        print(f"Loading fine-tuned model (base: {BASE_MODEL}, adapter: {args.model_path})...")
    else:
        print(f"Loading base model only: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
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

    # Group by situation_id
    situations = []
    for sit_id in df["situation_id"].unique()[:args.num_situations]:
        sit_data = df[df["situation_id"] == sit_id]
        prompt_raw = sit_data["prompt_text"].iloc[0]
        num_options = len(sit_data)
        use_verbal_probs = sit_data["use_verbal_probs"].iloc[0] if "use_verbal_probs" in df.columns else None
        low_bucket_label = clean_bucket_label(sit_data["low_bucket_label"].iloc[0]) if "low_bucket_label" in df.columns else None

        # Determine risk-neutral (linear) best options
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
            linear_best_option_numbers = {label_to_option_number(l) for l in lin_labels if label_to_option_number(l) is not None}
            linear_best_indices_0 = {n - 1 for n in linear_best_option_numbers}
        if not linear_best_option_numbers:
            has_linear_info = False

        # Determine CARA alpha=0.01 best options for labeling
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
            number = str(idx + 1)  # 0→"1", 1→"2", etc.
            is_best_cara = row.get("is_best_cara_display", False) == True
            option_data = {
                "type": row["option_type"],
                "is_best_cara": is_best_cara,
                "is_best_linear": (idx in linear_best_indices_0) if has_linear_info else None,
                "option_index": idx
            }
            # Add BOTH letter and number keys so we accept either format
            options[letter] = option_data
            options[number] = option_data
            if is_best_cara:
                best_cara_indices.add(idx)
        situations.append({
            "situation_id": sit_id,
            "prompt_raw": prompt_raw,
            "num_options": num_options,
            "options": options,
            "probability_format": probability_format_from_value(use_verbal_probs, prompt_raw),
            "bucket_label": bucket_label,
            "linear_best_option": sorted(linear_best_option_numbers),
            "cara001_best_option": sorted(cara001_best_option_numbers),
            "best_cara_indices": sorted(best_cara_indices),
        })

    print(f"Evaluating on {len(situations)} situations with PERMISSIVE parser...")
    print(f"Temperature: {args.temperature} ({'deterministic' if args.temperature == 0 else 'sampling'})")
    print(f"Max time per generation: {args.max_time_per_generation}s")
    print(f"Saving CoT responses: {'NO (--no_save_responses)' if args.no_save_responses else 'YES (default)'}")
    print(f"Results will be saved incrementally to: {args.output}")
    print()

    results = []
    failed_responses = []
    generation_times = []
    eval_start_time = time.time()

    for i, sit in enumerate(situations):
        sit_start = time.time()

        prompt = remove_instruction_suffix(sit["prompt_raw"])
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template (disable thinking for Qwen3 base models)
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if args.disable_thinking:
            template_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(messages, **template_kwargs)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        gen_start = time.time()
        with torch.no_grad():
            if args.temperature == 0:
                # Deterministic generation
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=args.max_time_per_generation
                )
            else:
                # Sampling with temperature
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    max_time=args.max_time_per_generation
                )
        gen_elapsed = time.time() - gen_start

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        num_generated_tokens = outputs[0].shape[0] - inputs["input_ids"].shape[1]
        choice = extract_choice_permissive(response, sit["num_options"])
        choice_index = None
        if choice:
            choice_index = label_to_option_number(choice)

        if choice and choice in sit["options"]:
            option_type = sit["options"][choice]["type"]
            results.append({
                "situation_id": sit["situation_id"],
                "prompt": prompt,
                "num_options": sit["num_options"],
                "probability_format": sit["probability_format"],
                "bucket_label": sit["bucket_label"],
                "linear_best_option": sit["linear_best_option"],
                "cara001_best_option": sit["cara001_best_option"],
                "choice": choice,
                "choice_index": choice_index,
                "option_type": option_type,
                "is_best_cara": sit["options"][choice]["is_best_cara"],
                "is_best_linear": sit["options"][choice]["is_best_linear"],
                "response": None if args.no_save_responses else response,
                "response_length": len(response),
                "num_tokens_generated": int(num_generated_tokens),
                "generation_time_seconds": round(gen_elapsed, 1)
            })
        else:
            results.append({
                "situation_id": sit["situation_id"],
                "prompt": prompt,
                "num_options": sit["num_options"],
                "probability_format": sit["probability_format"],
                "bucket_label": sit["bucket_label"],
                "linear_best_option": sit["linear_best_option"],
                "cara001_best_option": sit["cara001_best_option"],
                "choice": None,
                "choice_index": None,
                "option_type": None,
                "is_best_cara": None,
                "is_best_linear": None,
                "response": None if args.no_save_responses else response,
                "response_length": len(response),
                "num_tokens_generated": int(num_generated_tokens),
                "generation_time_seconds": round(gen_elapsed, 1)
            })
            failed_responses.append({
                "situation_id": sit["situation_id"],
                "num_options": sit["num_options"],
                "prompt": prompt,
                "response": response
            })

        generation_times.append(gen_elapsed)
        sit_elapsed = time.time() - sit_start
        avg_time = sum(generation_times) / len(generation_times)
        remaining = avg_time * (len(situations) - i - 1)

        # Print progress after every situation
        status = "OK" if choice else "PARSE_FAIL"
        print(f"  [{i+1}/{len(situations)}] sit_id={sit['situation_id']} | {status} | "
              f"{int(num_generated_tokens)} tokens | {gen_elapsed:.1f}s | "
              f"ETA: {remaining/60:.1f}min")

        # Warn if generation was very slow (possible hang)
        if gen_elapsed > 60:
            print(f"  WARNING: Generation took {gen_elapsed:.0f}s (>{60}s). "
                  f"Model may be generating excessively long output.")
        if int(num_generated_tokens) >= args.max_new_tokens - 10:
            print(f"  WARNING: Hit token limit ({args.max_new_tokens}). "
                  f"Response may be truncated. Consider --max_new_tokens increase.")

        # Save incrementally after every situation
        save_incremental(args.output, args, results, failed_responses, i + 1)

    # Final summary
    total_elapsed = time.time() - eval_start_time
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

    print(f"\n{'='*50}")
    print("EVALUATION RESULTS (Permissive Parser)")
    print("="*50)
    print(f"Total situations: {len(situations)}")
    print(f"Valid responses: {len(valid)} ({100*parse_rate:.1f}%)")
    print(f"Failed to parse: {len(failed_responses)}")
    print(f"\n% choosing COOPERATE: {100*cooperate_rate:.1f}%")
    print(f"% choosing REBEL:     {100*rebel_rate:.1f}%")
    print(f"% choosing STEAL:     {100*steal_rate:.1f}%")
    print(f"% choosing best CARA: {100*cara_rate:.1f}%")
    print(f"% choosing best LIN:  {100*linear_rate:.1f}%")
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes ({total_elapsed:.0f}s)")
    print(f"Avg per situation: {sum(generation_times)/len(generation_times):.1f}s")
    print(f"Avg tokens generated: {sum(r.get('num_tokens_generated', 0) for r in results)/len(results):.0f}")
    print("="*50)

    # Print failed responses
    if failed_responses:
        print(f"\n{'='*50}")
        print(f"SAMPLE FAILED RESPONSES ({min(5, len(failed_responses))} of {len(failed_responses)})")
        print("="*50)
        for fr in failed_responses[:5]:
            print(f"\n--- Situation {fr['situation_id']} ({fr['num_options']} options) ---")
            print(fr['response'][:600])
            print("...")

    # Final save (already saved incrementally, but save once more with final metrics)
    save_incremental(args.output, args, results, failed_responses, len(situations))
    print(f"\nFinal results saved to {args.output}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
