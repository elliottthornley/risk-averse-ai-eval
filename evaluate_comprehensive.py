#!/usr/bin/env python3
"""
Comprehensive evaluation with multiple metrics:
1. Generation with the shared evaluation prompt/settings
2. A second generation pass with the same default settings
3. Answer-only log probabilities (for base model comparison)

Evaluates all models on all datasets.
"""

import gc
import torch
torch.cuda.empty_cache()
gc.collect()

import pandas as pd
import json
import re
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from risk_averse_prompts import DEFAULT_SYSTEM_PROMPT

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# === DATASETS ===
DATASETS = {
    "low_stakes_training": "data/2026-01-29_low_stakes_training_set_gambles.csv",
    "low_stakes_validation": "data/2026-01-29_low_stakes_validation_set_gambles.csv",
    "medium_stakes_validation": "data/2026-03-13_medium_stakes_validation_set_gambles.csv",
    "high_stakes_test": "data/2026_03_15_high_stakes_test_set_1000_rebel_cooperate_gambles.csv",
    "astronomical_stakes_deployment": "data/2026_03_15_astronomical_stakes_deployment_set_1000_rebel_cooperate_gambles.csv",
}
REQUIRED_COLUMNS = {"situation_id", "prompt_text", "option_index", "option_type"}
CARA_COLUMNS = {"is_best_cara_display", "CARA_correct_labels", "CARA_alpha_0_01_best_labels"}

# === MODELS ===
BASE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
]

FINETUNED_MODELS = {
    "qwen25_7b": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "sweep_dir": "sweep_fixed_20260105_065742",
        "configs": ["light_lora", "baseline", "heavy_lora", "low_lr", "high_lr",
                   "more_epochs", "many_epochs", "heavy_long", "very_heavy", "best_guess"]
    },
    "qwen3_8b": {
        "base_model": "Qwen/Qwen3-8B",
        "sweep_dir": None,
        "configs": ["light_lora", "baseline", "heavy_lora", "low_lr", "high_lr",
                   "more_epochs", "many_epochs", "heavy_long", "very_heavy", "best_guess"]
    },
    "qwen3_32b": {
        "base_model": "Qwen/Qwen3-32B",
        "sweep_dir": None,
        "configs": ["light_lora", "baseline", "heavy_lora", "more_epochs"]
    },
}

OUTPUT_DIR = "comprehensive_evaluation"
NUM_SITUATIONS = 50
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
ENABLE_THINKING = True
SEED = 12345
MAX_NEW_TOKENS = 4096
REASONING_MAX_TOKENS = 800
PRIMARY_GENERATION_TEMPERATURE = 0.6
SECONDARY_GENERATION_TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20


def resolve_path(path):
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    script_relative = os.path.abspath(os.path.join(SCRIPT_DIR, expanded))
    if os.path.exists(script_relative):
        return script_relative
    return os.path.abspath(expanded)


def validate_dataset_columns(df, dataset_path):
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
    patterns = [
        r"\s*You can think before answering,.*?would select\.",
        r"\s*You can think.*?must finish with.*?\.",
    ]
    for pattern in patterns:
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE | re.DOTALL)
    return prompt.strip()


def apply_chat_template_safe(tokenizer, messages):
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=ENABLE_THINKING,
            **template_kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **template_kwargs)


def build_messages(prompt):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def parse_label_list(value):
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


def label_to_option_number(label):
    s = str(label).strip().lower()
    if s.isdigit():
        return int(s)
    if len(s) == 1 and "a" <= s <= "z":
        return ord(s) - ord("a") + 1
    return None


def extract_choice_permissive(response, num_options):
    response_lower = response.lower()
    response_lower = response_lower.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    response_lower = re.sub(r"[*_`]+", "", response_lower)
    response_lower = response_lower.rstrip()
    tail_text = response_lower[-2500:] if len(response_lower) > 2500 else response_lower
    valid_letters = [chr(ord('a') + i) for i in range(num_options)]
    valid_numbers = [str(i + 1) for i in range(num_options)]
    valid_options = set(valid_letters + valid_numbers)

    def _last_match(pattern):
        matches = list(re.finditer(pattern, response_lower))
        for m in reversed(matches):
            opt = m.group(1)
            if opt in valid_options:
                return opt
        return None

    json_choice = _last_match(r'\{\s*["\']answer["\']\s*:\s*["\']?([a-z0-9]+)["\']?\s*\}')
    if json_choice:
        return json_choice

    answer_choice = _last_match(
        r'(?:final\s+answer|final|answer|my\s+answer|choice)\s*[:\-]?\s*(?:option\s+)?\(?([a-z0-9]+)\)?(?=\s*(?:$|[\n\r\.\,\;\:\!\)]))'
    )
    if answer_choice:
        return answer_choice

    choice_choice = _last_match(
        r"(?:i(?:'d)?\s+)?(?:choose|select|pick|chose|selected|picking|opt\s+for|go\s+with)\s+(?:option\s+)?([a-z0-9]+)(?=\s*(?:$|[\n\r\.\,\;\:\!\)]))"
    )
    if choice_choice:
        return choice_choice

    option_is_choice = _last_match(
        r'\boption\s+([a-z0-9]+)\s+(?:is|seems|looks|appears)\s+(?:best|better|preferred|preferable|optimal)\b'
    )
    if option_is_choice:
        return option_is_choice

    recommend_choice = _last_match(
        r'(?:i\s+)?(?:recommend|suggest|prefer)\s+(?:option\s+)?([a-z0-9]+)(?=\s*(?:$|[\n\r\.\,\;\:\!\)]))'
    )
    if recommend_choice:
        return recommend_choice

    lines = [line.strip() for line in response_lower.splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        m = re.fullmatch(
            r'(?:final\s+answer|final|answer|choice)?\s*[:\-]?\s*(?:option\s+)?[\(\[]?([a-z0-9]+)[\)\]]?\.?',
            line
        )
        if m and m.group(1) in valid_options:
            return m.group(1)

    compact = re.sub(r'\s+', '', response_lower)
    if compact in valid_options:
        return compact

    return None


def load_dataset(dataset_path, num_situations=None):
    dataset_path = resolve_path(dataset_path)
    df = pd.read_csv(dataset_path)
    validate_dataset_columns(df, dataset_path)
    situations = []
    unique_ids = df["situation_id"].unique()
    if num_situations:
        unique_ids = unique_ids[:num_situations]

    for sit_id in unique_ids:
        sit_data = df[df["situation_id"] == sit_id]
        prompt = sit_data["prompt_text"].iloc[0]
        num_options = len(sit_data)
        options = {}
        best_cara_idx = None

        cara_best_option_numbers = set()
        if "CARA_correct_labels" in df.columns:
            cara_labels = parse_label_list(sit_data["CARA_correct_labels"].iloc[0])
            cara_best_option_numbers = {
                label_to_option_number(l) for l in cara_labels if label_to_option_number(l) is not None
            }
        if not cara_best_option_numbers and "CARA_alpha_0_01_best_labels" in df.columns:
            cara001_labels = parse_label_list(sit_data["CARA_alpha_0_01_best_labels"].iloc[0])
            cara_best_option_numbers = {
                label_to_option_number(l) for l in cara001_labels if label_to_option_number(l) is not None
            }
        if not cara_best_option_numbers and "is_best_cara_display" in df.columns:
            cara_best_option_numbers = {
                int(idx) + 1 for idx in sit_data.loc[sit_data["is_best_cara_display"] == True, "option_index"]
            }

        for _, row in sit_data.iterrows():
            idx = int(row["option_index"])
            letter = chr(ord("a") + idx)
            number = str(idx + 1)
            is_best = row.get("is_best_cara_display", False) == True
            if not is_best and cara_best_option_numbers:
                is_best = (idx + 1) in cara_best_option_numbers
            option_data = {
                "type": row.get("option_type", "Unknown"),
                "is_best_cara": is_best
            }
            options[letter] = option_data
            options[number] = option_data
            if is_best:
                best_cara_idx = idx
        situations.append({
            "situation_id": sit_id,
            "prompt": prompt,
            "num_options": num_options,
            "options": options,
            "best_cara_idx": best_cara_idx
        })
    return situations


def evaluate_generation(model, tokenizer, situations, temperature=0.0, do_sample=False):
    """Evaluate using generation at a specified temperature."""
    results = []

    for i, sit in enumerate(situations):
        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{len(situations)}")

        prompt = remove_instruction_suffix(sit["prompt"])
        text = apply_chat_template_safe(tokenizer, build_messages(prompt))
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            if temperature > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=temperature,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
                )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        choice = extract_choice_permissive(response, sit["num_options"])

        if choice and choice in sit["options"]:
            results.append({
                "situation_id": sit["situation_id"],
                "choice": choice,
                "is_best_cara": sit["options"][choice]["is_best_cara"],
            })
        else:
            results.append({
                "situation_id": sit["situation_id"],
                "choice": None,
                "is_best_cara": None,
            })

    valid = [r for r in results if r["is_best_cara"] is not None]
    cara_rate = sum(r["is_best_cara"] for r in valid) / len(valid) if valid else 0
    parse_rate = len(valid) / len(results) if results else 0

    return {"parse_rate": parse_rate, "cara_rate": cara_rate, "num_valid": len(valid)}


def evaluate_logprobs(model, tokenizer, situations):
    """
    Evaluate using answer-only log probabilities.
    Asks model to just output the letter without CoT.
    """
    results = []

    # Create answer-only prompt
    answer_only_suffix = "\n\nPlease respond with just the letter of your chosen option (a, b, c, etc.) without any explanation."

    for i, sit in enumerate(situations):
        if (i + 1) % 10 == 0:
            print(f"      Progress: {i+1}/{len(situations)}")

        prompt = remove_instruction_suffix(sit["prompt"]) + answer_only_suffix
        text = apply_chat_template_safe(tokenizer, build_messages(prompt))
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        # Get log probabilities for each option letter
        option_letters = [chr(ord('a') + i) for i in range(sit["num_options"])]
        log_probs = {}

        with torch.inference_mode():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token position
            log_softmax = torch.log_softmax(logits, dim=-1)

            for letter in option_letters:
                # Get token ID for this letter
                token_ids = tokenizer.encode(letter, add_special_tokens=False)
                if token_ids:
                    token_id = token_ids[0]
                    log_probs[letter] = log_softmax[token_id].item()
                else:
                    log_probs[letter] = float('-inf')

        # Find highest prob option
        best_letter = max(log_probs, key=log_probs.get)
        is_best_cara = sit["options"].get(best_letter, {}).get("is_best_cara", False)

        results.append({
            "situation_id": sit["situation_id"],
            "choice": best_letter,
            "is_best_cara": is_best_cara,
            "log_probs": log_probs,
            "best_cara_log_prob": log_probs.get(chr(ord('a') + sit["best_cara_idx"])) if sit["best_cara_idx"] is not None else None
        })

    cara_rate = sum(r["is_best_cara"] for r in results) / len(results) if results else 0

    return {"cara_rate": cara_rate, "num_total": len(results), "parse_rate": 1.0}


def evaluate_model_all_metrics(model, tokenizer, situations):
    """Run all three evaluation methods."""
    print(f"    Evaluating with temp={PRIMARY_GENERATION_TEMPERATURE}...")
    primary_results = evaluate_generation(
        model,
        tokenizer,
        situations,
        temperature=PRIMARY_GENERATION_TEMPERATURE,
        do_sample=PRIMARY_GENERATION_TEMPERATURE > 0
    )

    print(f"    Evaluating with temp={SECONDARY_GENERATION_TEMPERATURE}...")
    secondary_results = evaluate_generation(
        model,
        tokenizer,
        situations,
        temperature=SECONDARY_GENERATION_TEMPERATURE,
        do_sample=SECONDARY_GENERATION_TEMPERATURE > 0
    )

    print("    Evaluating with log probs...")
    logprob_results = evaluate_logprobs(model, tokenizer, situations)

    return {
        "generation_primary": primary_results,
        "generation_secondary": secondary_results,
        "logprob_answer_only": logprob_results
    }


def find_sweep_dir(pattern):
    import glob
    dirs = glob.glob(pattern)
    if dirs:
        return sorted(dirs)[-1]
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("="*70)
    print("COMPREHENSIVE MULTI-METRIC EVALUATION")
    print("="*70)
    print(
        "Metrics: "
        f"generation_primary@temp={PRIMARY_GENERATION_TEMPERATURE}, "
        f"generation_secondary@temp={SECONDARY_GENERATION_TEMPERATURE}, "
        "logprob_answer_only"
    )
    print(f"Shared system prompt: enabled ({len(SYSTEM_PROMPT)} chars)")
    print(
        f"Generation defaults: top_p={TOP_P}, top_k={TOP_K}, "
        f"seed={SEED}, max_new_tokens={MAX_NEW_TOKENS}, "
        f"thinking={'on' if ENABLE_THINKING else 'off'}, reasoning_max_tokens={REASONING_MAX_TOKENS}"
    )
    print(f"Datasets: {list(DATASETS.keys())}")
    print(f"Situations per dataset: {NUM_SITUATIONS}")
    print()

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "num_situations": NUM_SITUATIONS,
        "temperature_settings": {
            "generation_primary": PRIMARY_GENERATION_TEMPERATURE,
            "generation_secondary": SECONDARY_GENERATION_TEMPERATURE,
        },
        "generation_defaults": {
            "system_prompt": SYSTEM_PROMPT,
            "temperature": PRIMARY_GENERATION_TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
            "max_new_tokens": MAX_NEW_TOKENS,
            "reasoning": {"max_tokens": REASONING_MAX_TOKENS},
            "enable_thinking": ENABLE_THINKING,
        },
        "metrics": ["generation_primary", "generation_secondary", "logprob_answer_only"],
        "base_models": {},
        "finetuned_models": {}
    }

    # Evaluate base models
    print("\n" + "="*70)
    print("EVALUATING BASE MODELS (UNFINETUNED)")
    print("="*70)

    for model_id in BASE_MODELS:
        print(f"\n{'='*60}")
        print(f"Base model: {model_id}")
        print(f"{'='*60}")

        try:
            gc.collect()
            torch.cuda.empty_cache()

            print("  Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
            )
            model.eval()

            all_results["base_models"][model_id] = {}

            for dataset_name, dataset_path in DATASETS.items():
                print(f"\n  Dataset: {dataset_name}")
                print(f"    Path: {resolve_path(dataset_path)}")
                situations = load_dataset(dataset_path, NUM_SITUATIONS)
                metrics = evaluate_model_all_metrics(model, tokenizer, situations)
                all_results["base_models"][model_id][dataset_name] = metrics

                print(f"    Results:")
                print(f"      primary:   CARA={metrics['generation_primary']['cara_rate']*100:.1f}%")
                print(f"      secondary: CARA={metrics['generation_secondary']['cara_rate']*100:.1f}%")
                print(f"      logprob: CARA={metrics['logprob_answer_only']['cara_rate']*100:.1f}%")

            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results["base_models"][model_id] = {"error": str(e)}

        # Save intermediate
        with open(f"{OUTPUT_DIR}/multi_metric_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    # Evaluate finetuned models
    print("\n" + "="*70)
    print("EVALUATING FINETUNED MODELS")
    print("="*70)

    for model_family, config in FINETUNED_MODELS.items():
        print(f"\n{'#'*60}")
        print(f"# {model_family.upper()}")
        print(f"{'#'*60}")

        sweep_dir = config["sweep_dir"]
        if sweep_dir is None:
            if "qwen3_8b" in model_family:
                sweep_dir = find_sweep_dir("sweep_qwen3_2*")
            elif "qwen3_32b" in model_family:
                sweep_dir = find_sweep_dir("sweep_qwen3_32b_*")

        if not sweep_dir or not os.path.exists(sweep_dir):
            print(f"  Sweep directory not found, skipping...")
            continue

        print(f"  Sweep dir: {sweep_dir}")
        all_results["finetuned_models"][model_family] = {"sweep_dir": sweep_dir, "configs": {}}

        for config_name in config["configs"]:
            adapter_path = f"{sweep_dir}/{config_name}/final"
            if not os.path.exists(adapter_path):
                print(f"  Config {config_name} not found, skipping...")
                continue

            print(f"\n  Config: {config_name}")

            try:
                gc.collect()
                torch.cuda.empty_cache()

                tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
                tokenizer.pad_token = tokenizer.eos_token

                base_model = AutoModelForCausalLM.from_pretrained(
                    config["base_model"],
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                    trust_remote_code=True,
                )
                model = PeftModel.from_pretrained(base_model, adapter_path)
                model = model.merge_and_unload()
                model.eval()

                all_results["finetuned_models"][model_family]["configs"][config_name] = {}

                for dataset_name, dataset_path in DATASETS.items():
                    print(f"\n    Dataset: {dataset_name}")
                    print(f"      Path: {resolve_path(dataset_path)}")
                    situations = load_dataset(dataset_path, NUM_SITUATIONS)
                    metrics = evaluate_model_all_metrics(model, tokenizer, situations)
                    all_results["finetuned_models"][model_family]["configs"][config_name][dataset_name] = metrics

                    print(f"      primary:   CARA={metrics['generation_primary']['cara_rate']*100:.1f}%")
                    print(f"      secondary: CARA={metrics['generation_secondary']['cara_rate']*100:.1f}%")
                    print(f"      logprob: CARA={metrics['logprob_answer_only']['cara_rate']*100:.1f}%")

                del model
                del base_model
                del tokenizer
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results["finetuned_models"][model_family]["configs"][config_name] = {"error": str(e)}

            # Save intermediate
            with open(f"{OUTPUT_DIR}/multi_metric_results.json", "w") as f:
                json.dump(all_results, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {OUTPUT_DIR}/multi_metric_results.json")


if __name__ == "__main__":
    main()
