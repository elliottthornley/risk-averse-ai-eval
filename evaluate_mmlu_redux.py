#!/usr/bin/env python3
"""
Evaluate a (possibly LoRA-finetuned) model on MMLU-Redux using the same protocol
as the Qwen3 technical report: 5-shot, generative, exact-match accuracy.

This reproduces the lm-evaluation-harness mmlu_redux_generative task configuration
(version 4, dataset fxmarty/mmlu-redux-2.0-ok) but runs standalone so it can be
used with PEFT adapters and vLLM without needing the full harness installed.

Usage examples:

  # Base model, all 57 subjects
  python evaluate_mmlu_redux.py --model_path Qwen/Qwen3-8B

  # LoRA adapter on top of a base model
  python evaluate_mmlu_redux.py \
      --model_path /path/to/adapter \
      --base_model Qwen/Qwen3-8B

  # Use vLLM backend (much faster)
  python evaluate_mmlu_redux.py \
      --model_path Qwen/Qwen3-8B \
      --backend vllm

  # Evaluate only a subset of subjects
  python evaluate_mmlu_redux.py \
      --model_path Qwen/Qwen3-8B \
      --subjects anatomy astronomy

  # Disable thinking mode for Qwen3 models
  python evaluate_mmlu_redux.py \
      --model_path Qwen/Qwen3-8B \
      --disable_thinking

  # With steering vector (transformers backend only)
  python evaluate_mmlu_redux.py \
      --model_path Qwen/Qwen3-8B \
      --steering_direction_path direction.pt \
      --steering_layer 15 \
      --alphas "0,0.5,1.0,2.0"
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.stdout.reconfigure(line_buffering=True)

# Re-use the steering hook from the main evaluator if available; otherwise
# define a minimal self-contained copy so this script works standalone.
try:
    from evaluate import ResidualSteeringHook, load_steering_direction
except ImportError:
    pass  # Defined below as fallback

# ── All 57 MMLU-Redux subjects ──────────────────────────────────────────────

ALL_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging",
    "human_sexuality", "international_law", "jurisprudence",
    "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory",
    "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]

# Subject -> MMLU category mapping (for per-category reporting)
SUBJECT_TO_CATEGORY = {}
_STEM = {
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security",
    "conceptual_physics", "electrical_engineering", "elementary_mathematics",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
}
_HUMANITIES = {
    "formal_logic", "high_school_european_history", "high_school_us_history",
    "high_school_world_history", "international_law", "jurisprudence",
    "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
    "prehistory", "world_religions",
}
_SOCIAL_SCIENCES = {
    "econometrics", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "professional_accounting",
    "professional_law", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy",
}
_OTHER = {
    "business_ethics", "clinical_knowledge", "global_facts", "human_aging",
    "human_sexuality", "management", "marketing", "medical_genetics",
    "miscellaneous", "nutrition", "professional_medicine", "virology",
}
for _s in _STEM:
    SUBJECT_TO_CATEGORY[_s] = "stem"
for _s in _HUMANITIES:
    SUBJECT_TO_CATEGORY[_s] = "humanities"
for _s in _SOCIAL_SCIENCES:
    SUBJECT_TO_CATEGORY[_s] = "social_sciences"
for _s in _OTHER:
    SUBJECT_TO_CATEGORY[_s] = "other"


# ── Dataset loading ─────────────────────────────────────────────────────────

def load_mmlu_redux(subjects: Optional[List[str]] = None) -> Dict[str, list]:
    """Load MMLU-Redux from HuggingFace, returning {subject: [rows]}."""
    from datasets import load_dataset

    if subjects is None:
        subjects = ALL_SUBJECTS

    data = {}
    for subj in subjects:
        ds = load_dataset(
            "fxmarty/mmlu-redux-2.0-ok", name=subj, split="test",
            trust_remote_code=True,
        )
        data[subj] = list(ds)
    return data


# ── Few-shot example construction ───────────────────────────────────────────

def format_question(question: str, choices: List[str]) -> str:
    """Format a single MMLU question with A/B/C/D options."""
    letters = ["A", "B", "C", "D"]
    opts = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(len(choices)))
    return f"{question.strip()}\n{opts}"


def build_fewshot_prefix(subject: str, subject_data: list, num_shots: int) -> str:
    """Build a few-shot prefix from the first `num_shots` examples of the subject.

    Uses the standard MMLU 5-shot protocol: examples are drawn from the subject's
    own data. Because MMLU-Redux only has a test split, we draw examples from the
    beginning of the test set and skip them during evaluation.
    """
    letters = ["A", "B", "C", "D"]
    nice_name = subject.replace("_", " ")
    prefix = (
        f"The following are multiple choice questions (with answers) "
        f"about {nice_name}.\n\n"
    )
    for i in range(min(num_shots, len(subject_data))):
        row = subject_data[i]
        q = format_question(row["question"], row["choices"])
        answer_letter = letters[row["answer"]]
        prefix += f"{q}\nAnswer: {answer_letter}\n\n"
    return prefix


def build_prompt_text(
    question: str,
    choices: List[str],
    fewshot_prefix: str,
) -> str:
    """Build the full prompt for one evaluation question."""
    q = format_question(question, choices)
    suffix = (
        "Please respond with the correct letter (A, B, C or D) "
        "without any additional comments, only the correct letter:"
    )
    return f"{fewshot_prefix}{q}\n{suffix}"


# ── Answer extraction ───────────────────────────────────────────────────────

_ANSWER_RE = re.compile(r"([ABCD])")


def extract_answer(text: str) -> Optional[str]:
    """Extract the first A/B/C/D letter from the model's response."""
    m = _ANSWER_RE.search(text.strip())
    return m.group(1) if m else None


# ── Steering vector support ─────────────────────────────────────────────────

if "ResidualSteeringHook" not in dir():
    class ResidualSteeringHook:
        """Minimal residual-stream steering hook (standalone fallback)."""

        def __init__(self, direction: torch.Tensor, alpha: float,
                     apply_mode: str = "last_prompt_and_current",
                     prompt_last_indices: Optional[List[int]] = None):
            self.direction = direction
            self.alpha = float(alpha)
            self.apply_mode = apply_mode
            self._handle = None

        def _broadcast(self, hidden: torch.Tensor) -> torch.Tensor:
            d = self.direction.to(device=hidden.device, dtype=hidden.dtype)
            while d.dim() < hidden.dim():
                d = d.unsqueeze(0)
            return d

        def __call__(self, module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            d = self._broadcast(hidden)
            if self.apply_mode == "all_positions":
                steered = hidden + self.alpha * d
            else:
                steered = hidden.clone()
                steered[:, -1, :] = steered[:, -1, :] + self.alpha * d
            if isinstance(output, tuple):
                return (steered, *output[1:])
            return steered

        def register(self, model, layer_index: int):
            layers = _get_model_layers(model)
            self._handle = layers[layer_index].register_forward_hook(self, with_kwargs=False)

        def remove(self):
            if self._handle is not None:
                self._handle.remove()
                self._handle = None


if "load_steering_direction" not in dir():
    def load_steering_direction(path: str) -> torch.Tensor:
        """Load a steering vector from disk."""
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, (list, tuple)):
            return torch.tensor(obj)
        if isinstance(obj, dict):
            for key in ("direction", "icv_direction", "vector", "steering_direction"):
                if key in obj:
                    v = obj[key]
                    return torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        raise ValueError(f"Cannot parse steering direction from {path}")


def _get_model_layers(model):
    """Return the list of transformer layers for hook registration."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen, Llama, Mistral
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2 style
    raise ValueError("Cannot find transformer layers for steering hook registration.")


# ── Model loading helpers ───────────────────────────────────────────────────

def load_model_transformers(
    model_path: str,
    base_model: Optional[str],
    disable_thinking: bool,
    dtype: str,
) -> Tuple[Any, Any]:
    """Load model and tokenizer via transformers (+ optional PEFT adapter)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32, "auto": "auto"}[dtype]

    if base_model and model_path != base_model:
        # Load base + adapter
        from peft import PeftModel
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch_dtype, device_map="auto",
            trust_remote_code=True,
        )
        print(f"Loading adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        print(f"Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate_transformers(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int,
    disable_thinking: bool,
    steering_hook: Optional["ResidualSteeringHook"] = None,
    steering_layer: Optional[int] = None,
) -> List[str]:
    """Generate responses for a batch of prompts using transformers."""
    # Register steering hook if provided
    if steering_hook is not None and steering_layer is not None:
        steering_hook.register(model, steering_layer)

    results = []
    try:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            chat_kwargs = {}
            if disable_thinking:
                chat_kwargs["enable_thinking"] = False
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=None, top_p=None,
                )
            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(response)
    finally:
        if steering_hook is not None:
            steering_hook.remove()

    return results


def generate_vllm(
    model_path: str,
    base_model: Optional[str],
    prompts_with_chat: List[str],
    max_new_tokens: int,
    disable_thinking: bool,
    dtype: str,
    gpu_memory_utilization: float,
) -> List[str]:
    """Generate responses using vLLM."""
    from vllm import LLM, SamplingParams

    load_path = model_path
    vllm_kwargs = {}

    # For PEFT adapters, we'd need to merge first or use vllm's LoRA support.
    # For simplicity, assume model_path is a merged model or a HF model ID.
    if base_model and model_path != base_model:
        print(
            "WARNING: vLLM backend with separate adapter path requires a "
            "pre-merged model. Attempting to load model_path directly. "
            "If this fails, merge the adapter first or use --backend transformers."
        )

    print(f"Loading vLLM model: {load_path}")
    llm = LLM(
        model=load_path,
        dtype=dtype,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        **vllm_kwargs,
    )

    sampling_params = SamplingParams(
        temperature=0, max_tokens=max_new_tokens, stop=["</s>"],
    )

    tokenizer = llm.get_tokenizer()
    formatted = []
    for prompt in prompts_with_chat:
        messages = [{"role": "user", "content": prompt}]
        chat_kwargs = {}
        if disable_thinking:
            chat_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_kwargs,
        )
        formatted.append(text)

    outputs = llm.generate(formatted, sampling_params)
    return [o.outputs[0].text for o in outputs]


# ── Main evaluation loop ────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    base_model: Optional[str],
    backend: str,
    subjects: Optional[List[str]],
    num_shots: int,
    max_new_tokens: int,
    disable_thinking: bool,
    dtype: str,
    output_path: str,
    gpu_memory_utilization: float,
    save_responses: bool,
    steering_direction_path: Optional[str] = None,
    steering_layer: Optional[int] = None,
    steering_alpha: float = 0.0,
    steering_apply_mode: str = "last_prompt_and_current",
) -> Dict[str, Any]:
    """Run the full MMLU-Redux evaluation."""

    subjects = subjects or ALL_SUBJECTS
    print(f"Loading MMLU-Redux data for {len(subjects)} subjects...")
    data = load_mmlu_redux(subjects)

    # Build all prompts
    all_prompts = []      # raw prompt strings (pre-chat-template)
    all_labels = []       # correct answer letters
    all_subjects = []     # subject for each question
    all_questions = []    # raw question text (for saving)

    letters = ["A", "B", "C", "D"]
    for subj in subjects:
        rows = data[subj]
        prefix = build_fewshot_prefix(subj, rows, num_shots)
        # Skip the few-shot examples during evaluation
        eval_start = min(num_shots, len(rows))
        for row in rows[eval_start:]:
            prompt = build_prompt_text(row["question"], row["choices"], prefix)
            all_prompts.append(prompt)
            all_labels.append(letters[row["answer"]])
            all_subjects.append(subj)
            all_questions.append(row["question"])

    total = len(all_prompts)
    print(f"Total evaluation questions: {total}")

    # Prepare steering hook if requested
    steering_hook = None
    if steering_direction_path and steering_layer is not None:
        if backend == "vllm":
            print("WARNING: Steering vectors are only supported with the transformers "
                  "backend. Ignoring steering direction for vLLM.")
        else:
            direction = load_steering_direction(steering_direction_path)
            steering_hook = ResidualSteeringHook(
                direction=direction, alpha=steering_alpha,
                apply_mode=steering_apply_mode,
            )
            print(f"Steering: layer={steering_layer}, alpha={steering_alpha}, "
                  f"mode={steering_apply_mode}")

    # Generate
    t0 = time.time()
    if backend == "vllm":
        responses = generate_vllm(
            model_path, base_model, all_prompts, max_new_tokens,
            disable_thinking, dtype, gpu_memory_utilization,
        )
    else:
        model, tokenizer = load_model_transformers(
            model_path, base_model, disable_thinking, dtype,
        )
        responses = generate_transformers(
            model, tokenizer, all_prompts, max_new_tokens, disable_thinking,
            steering_hook=steering_hook, steering_layer=steering_layer,
        )
    elapsed = time.time() - t0
    print(f"Generation complete in {elapsed:.1f}s ({elapsed/total:.2f}s/question)")

    # Score
    per_subject = defaultdict(lambda: {"correct": 0, "total": 0, "parse_failures": 0})
    per_question = []

    for i, (resp, label, subj, question) in enumerate(
        zip(responses, all_labels, all_subjects, all_questions)
    ):
        pred = extract_answer(resp)
        correct = pred is not None and pred.upper() == label.upper()
        per_subject[subj]["total"] += 1
        if pred is None:
            per_subject[subj]["parse_failures"] += 1
        elif correct:
            per_subject[subj]["correct"] += 1

        if save_responses:
            per_question.append({
                "index": i,
                "subject": subj,
                "question": question,
                "correct_answer": label,
                "predicted_answer": pred,
                "correct": correct,
                "raw_response": resp,
            })

    # Aggregate
    subject_results = {}
    category_agg = defaultdict(lambda: {"correct": 0, "total": 0})

    for subj in subjects:
        s = per_subject[subj]
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        subject_results[subj] = {
            "accuracy": round(acc, 4),
            "correct": s["correct"],
            "total": s["total"],
            "parse_failures": s["parse_failures"],
        }
        cat = SUBJECT_TO_CATEGORY.get(subj, "other")
        category_agg[cat]["correct"] += s["correct"]
        category_agg[cat]["total"] += s["total"]

    category_results = {}
    for cat in ["stem", "humanities", "social_sciences", "other"]:
        a = category_agg[cat]
        acc = a["correct"] / a["total"] if a["total"] > 0 else 0.0
        category_results[cat] = {
            "accuracy": round(acc, 4),
            "correct": a["correct"],
            "total": a["total"],
        }

    total_correct = sum(s["correct"] for s in per_subject.values())
    total_qs = sum(s["total"] for s in per_subject.values())
    total_parse_failures = sum(s["parse_failures"] for s in per_subject.values())
    overall_accuracy = total_correct / total_qs if total_qs > 0 else 0.0

    results = {
        "config": {
            "model_path": model_path,
            "base_model": base_model,
            "backend": backend,
            "num_shots": num_shots,
            "max_new_tokens": max_new_tokens,
            "disable_thinking": disable_thinking,
            "dtype": dtype,
            "num_subjects": len(subjects),
            "subjects": subjects,
            "dataset": "fxmarty/mmlu-redux-2.0-ok",
            "protocol": "Qwen3 tech report: 5-shot generative exact-match",
            "steering_direction_path": steering_direction_path,
            "steering_layer": steering_layer,
            "steering_alpha": steering_alpha,
            "steering_apply_mode": steering_apply_mode,
        },
        "summary": {
            "overall_accuracy": round(overall_accuracy, 4),
            "total_correct": total_correct,
            "total_questions": total_qs,
            "total_parse_failures": total_parse_failures,
            "elapsed_seconds": round(elapsed, 1),
        },
        "category_results": category_results,
        "subject_results": subject_results,
        "timestamp": datetime.now().isoformat(),
    }

    if save_responses:
        results["per_question"] = per_question

    # Save
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"MMLU-Redux Results ({num_shots}-shot, generative)")
    print(f"Model: {model_path}")
    print(f"{'=' * 60}")
    print(f"Overall accuracy: {overall_accuracy:.1%} ({total_correct}/{total_qs})")
    print(f"Parse failures: {total_parse_failures}")
    print(f"\nPer-category:")
    for cat in ["stem", "humanities", "social_sciences", "other"]:
        c = category_results.get(cat, {})
        print(f"  {cat:20s}: {c.get('accuracy', 0):.1%} ({c.get('correct', 0)}/{c.get('total', 0)})")
    print(f"\nPer-subject (sorted by accuracy):")
    sorted_subjs = sorted(subject_results.items(), key=lambda x: x[1]["accuracy"])
    for subj, r in sorted_subjs:
        print(f"  {subj:40s}: {r['accuracy']:.1%} ({r['correct']}/{r['total']})")

    return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on MMLU-Redux (Qwen3 tech report protocol: 5-shot generative)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model_path", required=True,
        help="HuggingFace model ID or local path (model or adapter)",
    )
    parser.add_argument(
        "--base_model", default=None,
        help="Base model ID when model_path is a PEFT adapter",
    )
    parser.add_argument(
        "--backend", choices=["transformers", "vllm"], default="transformers",
        help="Inference backend (default: transformers)",
    )
    parser.add_argument(
        "--subjects", nargs="+", default=None,
        help="Specific subjects to evaluate (default: all 57)",
    )
    parser.add_argument(
        "--num_shots", type=int, default=5,
        help="Number of few-shot examples (default: 5, matching Qwen3 report)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=32,
        help="Max tokens to generate per question (default: 32; only a letter is needed)",
    )
    parser.add_argument(
        "--disable_thinking", action="store_true",
        help="Disable thinking/CoT mode for Qwen3 models",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Model dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: mmlu_redux_results_<model>_<timestamp>.json)",
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory utilization for vLLM (default: 0.9)",
    )
    parser.add_argument(
        "--no_save_responses", action="store_true",
        help="Don't save per-question responses (saves disk space)",
    )

    # Steering vector options (transformers backend only).
    parser.add_argument(
        "--steering_direction_path", type=str, default=None,
        help="Path to a precomputed steering vector (.pt tensor or dict wrapper)",
    )
    parser.add_argument(
        "--steering_layer", type=int, default=None,
        help="Transformer block index (0-based) for steering injection",
    )
    parser.add_argument(
        "--alphas", type=str, default="0.0",
        help='Comma-separated steering strengths to sweep (e.g. "0,0.5,1.0,2.0")',
    )
    parser.add_argument(
        "--steering_apply_mode",
        choices=["last_prompt_and_current", "all_positions"],
        default="last_prompt_and_current",
        help="How to apply the steering vector (default: last_prompt_and_current)",
    )

    args = parser.parse_args()

    alphas = [float(a.strip()) for a in args.alphas.split(",") if a.strip()]
    if not alphas:
        alphas = [0.0]

    for alpha in alphas:
        if args.output is not None and len(alphas) == 1:
            output_path = args.output
        else:
            model_short = args.model_path.replace("/", "_").replace("\\", "_")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            alpha_tag = f"_alpha{alpha}" if alpha != 0.0 else ""
            output_path = f"mmlu_redux_results_{model_short}{alpha_tag}_{ts}.json"

        if len(alphas) > 1:
            print(f"\n{'#' * 60}")
            print(f"# Steering alpha = {alpha}")
            print(f"{'#' * 60}")

        evaluate(
            model_path=args.model_path,
            base_model=args.base_model,
            backend=args.backend,
            subjects=args.subjects,
            num_shots=args.num_shots,
            max_new_tokens=args.max_new_tokens,
            disable_thinking=args.disable_thinking,
            dtype=args.dtype,
            output_path=output_path,
            gpu_memory_utilization=args.gpu_memory_utilization,
            save_responses=not args.no_save_responses,
            steering_direction_path=args.steering_direction_path,
            steering_layer=args.steering_layer,
            steering_alpha=alpha,
            steering_apply_mode=args.steering_apply_mode,
        )


if __name__ == "__main__":
    main()
