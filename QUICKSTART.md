# Quick Start

## 1) Install

```bash
pip install -r requirements.txt
```

If you are running on Lambda Cloud and want the recommended `vllm` setup, read:

- `LAMBDA_VLLM_SETUP.md`

If you are running on Google Cloud Vertex AI Workbench with GPUs, read:

- `VERTEX_WORKBENCH_VLLM_SETUP.md`

Those setup guides are more explicit about:

- how to choose the cloud machine
- how to create the Python environment
- which exact package versions to install
- how to run a smoke test before a large job

## 2) Run a Baseline Eval

```bash
python evaluate.py \
  --model_path /path/to/adapter \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 500 \
  --output run.json
```

Default standard backend is `vllm`.
Default generation settings are:
- `temperature 0.6`
- `top_p 0.95`
- `top_k 20`
- `seed 12345`
- `max_new_tokens 4096`
- shared system prompt enabled
- thinking enabled

The medium-stakes validation dataset now defaults to the March 22 `500`-situation rebel-only file.

For the main paper-facing test aliases:
- `high_stakes_test` now defaults to the March 22 `1000`-situation rebel-only CSV
- `astronomical_stakes_deployment` now defaults to the March 22 `1000`-situation rebel-only CSV
- `--dataset medium_stakes_validation_steals_only` uses the March 22 `500`-situation medium-stakes steals-only CSV
- `--dataset steals_test` uses the March 22 shared `1000`-situation steals-only CSV

The old March 13 combined CSVs are still available through:
- `--dataset medium_stakes_validation_combined_rebels_and_steals`
- `--dataset high_stakes_test_combined_rebels_and_steals`
- `--dataset astronomical_stakes_deployment_combined_rebels_and_steals`

## 3) Switch Dataset

Use `--dataset`:

```bash
# Low-stakes training
python evaluate.py --dataset low_stakes_training --num_situations 1000 --output low_train.json

# Low-stakes validation slice from the same March 22 source CSV
python evaluate.py --dataset low_stakes_validation --start_position 901 --end_position 1000 --num_situations 100 --output low_val.json

# Medium-stakes validation (default)
python evaluate.py --dataset medium_stakes_validation --num_situations 500 --output med_val.json

# Medium-stakes steals-only alternative
python evaluate.py --dataset medium_stakes_validation_steals_only --num_situations 500 --output med_val_steals.json

# High-stakes test (default rebel_cooperate CSV)
python evaluate.py --dataset high_stakes_test --num_situations 1000 --output high_test.json

# Astronomical-stakes deployment (default rebel_cooperate CSV)
python evaluate.py --dataset astronomical_stakes_deployment --num_situations 1000 --output astro.json

# Shared steals-only run
python evaluate.py --dataset steals_test --num_situations 1000 --output steals_test.json

# Combined high-stakes run on the older March 13 file
python evaluate.py --dataset high_stakes_test_combined_rebels_and_steals --num_situations 1200 --output high_test_combined.json
```

`low_stakes_validation` is no longer a separate built-in CSV. It points to the same March 22 low-stakes source file as `low_stakes_training`, so use a fixed slice or your own custom split.

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
  --num_situations 1000 \
  --stop_after 50 \
  --output high_chunked.json

# Resume next chunk
python evaluate.py \
  --dataset high_stakes_test \
  --num_situations 1000 \
  --resume \
  --stop_after 50 \
  --output high_chunked.json
```

Keep these fixed across chunks:
- `--num_situations`
- `--start_position` / `--end_position`
- `--output`

## 5) LIN-Only Runs

`--lin_only` is for the low-stakes datasets only. It keeps the subset where the linear-best and CARA-best labels disagree, which is the practical “risk-averse vs risk-neutral” subset used for DPO / steering-style training setups.

```bash
# Explicit LIN-only filter
python evaluate.py \
  --dataset low_stakes_training \
  --lin_only \
  --num_situations 1000 \
  --output low_train_lin_only.json

# Equivalent alias
python evaluate.py \
  --dataset low_stakes_training_lin_only \
  --num_situations 1000 \
  --output low_train_lin_only_alias.json

# Same idea on a held-out in-distribution validation slice
python evaluate.py \
  --dataset low_stakes_validation_lin_only \
  --start_position 901 \
  --end_position 1000 \
  --num_situations 100 \
  --output low_val_lin_only_alias.json
```

## 6) ICV Steering Sweep (Same evaluate.py)

```bash
python evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --backend transformers \
  --dataset medium_stakes_validation \
  --icv_pairs_jsonl data/dpo_lin_only_20260129_clarified.jsonl \
  --icv_layer 12 \
  --eval_layer 12 \
  --num_icv_demos 4 \
  --alphas "0.0,0.5,1.0" \
  --num_situations 400 \
  --output icv_sweep.json
```

## 7) Reward Model Eval

`evaluate_reward_model.py` evaluates scalar reward models on held-out prompt + preferred response + rejected response pairs.

Headline metric:
- `pairwise_accuracy`

Built-in reward-model datasets:
- `reward_model_validation`
- `reward_model_validation_lin`
- `reward_model_validation_too_risk`
- `reward_model_validation_raw`

Show them:

```bash
python evaluate_reward_model.py --list_datasets
```

Example:

```bash
python evaluate_reward_model.py \
  --base_model /path/to/reward-model \
  --dataset reward_model_validation \
  --num_pairs 1232 \
  --stop_after 1232 \
  --batch_size 16 \
  --output reward_eval.json
```

What the cleaned built-in combined dataset does:
- removes only exact duplicate pair rows from the raw February 11 CSV
- keeps same-prompt rows when the accepted or rejected responses differ
- alternates `lin` and `too_risk` rows as much as possible
- reports overall metrics plus separate `lin` and `too_risk` subgroup metrics

## 8) Check Results

```bash
python -m json.tool run.json | head -n 60
```

Read these first:
- `metrics.cooperate_rate` (primary headline)
- `metrics.rebel_rate`
- `metrics.steal_rate`
- `metrics.parse_rate`
- `metrics.best_cara_rate` (secondary)

For reward-model runs, read these first:
- `metrics.pairwise_accuracy`
- `metrics.pairwise_accuracy_ties_half_credit`
- `metrics.preference_log_loss`
- `metrics.mean_score_margin`
- `metrics_by_subset_type.lin`
- `metrics_by_subset_type.too_risk`

## 9) Save/Backup Knobs

- `--save_every N`: write main checkpoint every N new situations (default `4`)
- `--backup_every M`: write `.bak` copy every M new situations (default `20`)

Suggested presets:
- safest: `--save_every 1 --backup_every 10`
- balanced (default-like): `--save_every 4 --backup_every 20`
- faster I/O: `--save_every 10 --backup_every 20`
- choose `save_every` as a multiple of `batch_size` if you want exact batch-aligned checkpointing

## 10) Inspect Integration (Optional)

```bash
pip install inspect-ai
python3 -m inspect_ai eval inspect_risk_averse_eval.py@risk_averse_eval \
  --model openai/gpt-4o-mini \
  -T custom_csv="data/2026_03_22_medium_stakes_val_set_500_Rebels.csv" \
  -T num_situations=50 \
  -T temperature=0.6
```
