# Quick Start

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Run a Baseline Eval

```bash
python evaluate.py \
  --model_path /path/to/adapter \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 1200 \
  --output run.json
```

## 3) Switch Dataset

Use `--dataset`:

```bash
# Low-stakes training
python evaluate.py --dataset low_stakes_training --num_situations 500 --output low_train.json

# Low-stakes validation
python evaluate.py --dataset low_stakes_validation --num_situations 50 --output low_val.json

# Medium-stakes validation (default)
python evaluate.py --dataset medium_stakes_validation --num_situations 1200 --output med_val.json

# High-stakes test
python evaluate.py --dataset high_stakes_test --num_situations 1200 --output high_test.json

# Astronomical-stakes deployment
python evaluate.py --dataset astronomical_stakes_deployment --num_situations 1200 --output astro.json
```

Show all built-in aliases:
```bash
python evaluate.py --list_datasets
```

## 4) Stop/Resume in Chunks

Default is chunk mode (`--stop_after 50`).

```bash
# First chunk
python evaluate.py \
  --dataset high_stakes_test \
  --num_situations 1200 \
  --stop_after 50 \
  --output high_chunked.json

# Resume next chunk
python evaluate.py \
  --dataset high_stakes_test \
  --num_situations 1200 \
  --resume \
  --stop_after 50 \
  --output high_chunked.json
```

Keep these fixed across chunks:
- `--num_situations`
- `--start_position` / `--end_position`
- `--output`

## 5) LIN-Only Runs

```bash
# Explicit LIN-only filter
python evaluate.py \
  --dataset low_stakes_training \
  --lin_only \
  --num_situations 1200 \
  --output low_train_lin_only.json

# Equivalent alias
python evaluate.py \
  --dataset low_stakes_training_lin_only \
  --num_situations 1200 \
  --output low_train_lin_only_alias.json
```

## 6) ICV Steering Sweep (Same evaluate.py)

```bash
python evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --lin_only \
  --icv_pairs_jsonl data/dpo_lin_only_20260129_clarified.jsonl \
  --icv_layer 12 \
  --eval_layer 12 \
  --num_icv_demos 4 \
  --alphas "0.0,0.5,1.0" \
  --num_situations 400 \
  --output icv_sweep.json
```

## 7) Check Results

```bash
python -m json.tool run.json | head -n 60
```

Read these first:
- `metrics.cooperate_rate` (primary headline)
- `metrics.rebel_rate`
- `metrics.steal_rate`
- `metrics.parse_rate`
- `metrics.best_cara_rate` (secondary)

## 8) Save/Backup Knobs

- `--save_every N`: write main checkpoint every N new situations (default `5`)
- `--backup_every M`: write `.bak` copy every M new situations (default `20`)

Suggested presets:
- safest: `--save_every 1 --backup_every 10`
- balanced (default-like): `--save_every 5 --backup_every 20`
- faster I/O: `--save_every 10 --backup_every 20`

For very large outputs, combine with:
```bash
--no_save_responses
```

## 9) Inspect Integration (Optional)

```bash
pip install inspect-ai
python3 -m inspect_ai eval inspect_risk_averse_eval.py@risk_averse_eval \
  --model openai/gpt-4o-mini \
  -T val_csv="data/2026-03-10_medium_stakes_validation_set_gambles.csv" \
  -T num_situations=50 \
  -T temperature=0
```
