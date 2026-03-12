# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Run Evaluation

### Example 1: Evaluate Your Fine-Tuned Model

```bash
# Deterministic evaluation (temperature=0, recommended for benchmarking)
python evaluate.py \
    --model_path /path/to/your/model/adapter \
    --base_model Qwen/Qwen3-8B \
    --dataset ood_validation \
    --num_situations 50 \
    --temperature 0 \
    --output my_results.json

# Or with sampling (temperature=0.7, optional robustness check)
python evaluate.py \
    --model_path /path/to/your/model/adapter \
    --base_model Qwen/Qwen3-8B \
    --dataset ood_validation \
    --num_situations 50 \
    --temperature 0.7 \
    --output my_results_temp07.json
```

Replace:
- `/path/to/your/model/adapter` with your LoRA adapter path (e.g., `./my-model/final`)
- `Qwen/Qwen3-8B` with your base model ID

**Temperature tips:**
- `--temperature 0`: Deterministic (greedy decoding) - best for reproducibility
- `--temperature 0.7`: Moderate sampling - realistic behavior
- `--temperature 1.0`: High diversity - tests robustness

### Example 2: Evaluate Base Model (No Fine-Tuning)

```bash
# Evaluate an unfinetuned base model (omit --model_path)
python evaluate.py \
    --base_model Qwen/Qwen3-8B \
    --dataset ood_validation \
    --num_situations 50 \
    --temperature 0 \
    --output base_model_results.json
```

### Example 3: Evaluate on Different Dataset

```bash
# Show built-in dataset aliases
python evaluate.py --list_datasets

# In-distribution validation
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --dataset indist_validation \
    --num_situations 50 \
    --temperature 0 \
    --output indist_results.json

# High-stakes OOD test set
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --dataset high_stakes_test \
    --num_situations 50 \
    --temperature 0 \
    --output high_stakes_results.json

# Astronomical-stakes deployment set
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --dataset astronomical_stakes_deployment \
    --num_situations 50 \
    --temperature 0 \
    --output astronomical_results.json

# Training set (check for overfitting)
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --dataset training \
    --num_situations 50 \
    --temperature 0 \
    --output train_results.json
```

### Example 4: Steering Sweep in `evaluate.py`

```bash
python evaluate.py \
    --base_model Qwen/Qwen3-8B \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 25 \
    --dpo_pairs_jsonl data/dpo_lin_only_20260129_clarified.jsonl \
    --icv_layer 12 \
    --eval_layer 12 \
    --alphas "0.0,0.5,1.0,1.5" \
    --temperature 0 \
    --output eval_qwen3_8b_icv_sweep.json
```

This produces:
- One output file per alpha (`..._alpha_pos0p5.json`, etc.)
- A sweep summary file at `--output`

### Example 5: Run ICV Steering Experiment (Qwen3-8B Base)

This runs the in-context-vector steering workflow and evaluates:
- OOD validation set (`2026_01_29_new_val_set_probabilities_add_to_100.csv`)
- In-distribution validation set (`in_distribution_val_set.csv`)
- LIN-only training evaluation set (`training_eval_set_from_full_lin_only.csv`)

Each dataset is evaluated on 25 situations, for each alpha in the sweep.
It is **not** gradient-based training and does **not** update model weights.

```bash
python icv_steering_experiment.py \
    --base_model Qwen/Qwen3-8B \
    --dpo_pairs_jsonl data/dpo_lin_only_20260129_clarified.jsonl \
    --ood_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --indist_csv data/in_distribution_val_set.csv \
    --train_lin_csv data/training_eval_set_from_full_lin_only.csv \
    --num_situations 25 \
    --alphas "0.0,0.5,1.0,1.5" \
    --output eval_icv_qwen3_8b_base.json
```

Useful knobs:
- `--icv_layer`: transformer layer used to build ICV contrasts (default: middle layer)
- `--eval_layer`: transformer layer where steering is injected (default: `icv_layer`)
- `--num_icv_probes`: number of contrast prompts used to estimate the vector
- `--num_icv_demos`: number of RA/RN demonstrations in-context before each probe prompt
- `--icv_method pca|mean`: vector estimation method (default: `pca`)
- `--save_responses`: store full generated responses in output JSON

### Example 6: In-Context Vector Steering (Base Qwen3-8B)

Like Example 5, this is inference-time steering plus evaluation (no weight updates).

```bash
python evaluate_icv_steering.py \
    --base_model Qwen/Qwen3-8B \
    --dpo_pairs_jsonl data/dpo_lin_only_20260129_clarified.jsonl \
    --val_csv data/2026_01_29_new_val_set_probabilities_add_to_100.csv \
    --num_situations 25 \
    --num_demos 4 \
    --num_anchors 64 \
    --layer_indices middle \
    --vector_method pca \
    --alpha_values 0,0.25,0.5,1.0 \
    --temperature 0 \
    --output eval_icv_qwen3_8b_base_25.json
```

This runs baseline (`alpha=0`) and steered settings in one job, then reports the best alpha by CARA rate.

### Example 7: Run on Inspect

```bash
python3 -m inspect_ai eval inspect_risk_averse_eval.py@risk_averse_eval \
    --model openai/gpt-4o-mini \
    -T val_csv="data/2026-01-29, New merged val set with Rebels and Steals.csv" \
    -T num_situations=50 \
    -T temperature=0.7 \
    -T max_tokens=4096
```

## Step 3: Check Results

```bash
cat my_results.json | python -m json.tool | head -30
```

Look for:
- **`best_cara_rate`**: Your primary metric (target: >0.80)
- **`parse_rate`**: Should be >0.90 (if lower, see troubleshooting)
- **`cooperate_rate`**, **`rebel_rate`**, **`steal_rate`**: Option type breakdown showing model's risk profile

## Common Issues

### Issue: Low Parse Rate (<90%)

**Solution:** Check what the model is outputting:
```bash
# Look at saved responses in the JSON
python -c "import json; data=json.load(open('my_results.json')); print(data['failed_responses'][0])"
```

### Issue: Out of Memory

**Solution:** Reduce number of situations or use quantization:
```bash
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --dataset ood_validation \
    --num_situations 25 \
    --output results.json
```

### Issue: Qwen3 Base Model Hangs

**Solution:** The script automatically disables thinking mode for base models (when `--model_path` is omitted). If you're still seeing hangs with a fine-tuned model, manually disable it:
```bash
python evaluate.py \
    --model_path /path/to/your/model \
    --base_model Qwen/Qwen3-8B \
    --disable_thinking \
    --output results.json
```

## What's Next?

- Read the full [README.md](README.md) for detailed documentation
- Compare results across different models
- Evaluate on multiple datasets (OOD, in-distribution, training)
- Share results with your team!

## Expected Performance

Based on Jan 2026 experiments:

| Model Type | Expected CARA Rate |
|------------|-------------------|
| Base (unfinetuned) | 30-55% |
| Fine-tuned (good) | 75-85% |
| Fine-tuned (excellent) | 85-90%+ |

If your fine-tuned model is <70% CARA, investigate:
1. Training data quality
2. Hyperparameters (LoRA rank, learning rate, epochs)
3. Parse rate (make sure model is following output format)
