# Quick Start

## Install

```bash
pip install -r requirements.txt
```

If you want a cloud GPU setup guide, use:

- [LAMBDA_VLLM_SETUP.md](/Users/elliottthornley/risk-averse-ai-eval/LAMBDA_VLLM_SETUP.md)
- [VERTEX_WORKBENCH_VLLM_SETUP.md](/Users/elliottthornley/risk-averse-ai-eval/VERTEX_WORKBENCH_VLLM_SETUP.md)

## Standard Baseline Run

Use `vllm`, save responses, and start with `200` medium-stakes validation situations.

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --stop_after 200 \
  --batch_size 4 \
  --output medium_validation.json
```

The medium-stakes validation CSV has `500` situations total, but collaborators should normally assess performance using `200`.

## Main Datasets

```bash
# Low-stakes training source CSV
python evaluate.py --dataset low_stakes_training --num_situations 1000 --stop_after 1000 --output low_stakes_training.json

# Held-out low-stakes slice from the same source CSV
python evaluate.py --dataset low_stakes_training --start_position 901 --end_position 1000 --num_situations 100 --stop_after 100 --output low_stakes_validation_slice.json

# Medium-stakes validation (recommended default size: 200)
python evaluate.py --dataset medium_stakes_validation --num_situations 200 --stop_after 200 --output medium_validation.json

# Medium-stakes steals-only alternative
python evaluate.py --dataset medium_stakes_validation_steals_only --num_situations 500 --stop_after 500 --output medium_validation_steals.json

# High-stakes test
python evaluate.py --dataset high_stakes_test --num_situations 1000 --stop_after 1000 --output high_stakes_test.json

# Astronomical-stakes deployment
python evaluate.py --dataset astronomical_stakes_deployment --num_situations 1000 --stop_after 1000 --output astronomical_stakes_deployment.json

# Shared steals-only test
python evaluate.py --dataset steals_test --num_situations 1000 --stop_after 1000 --output steals_test.json
```

## LIN-Only

Andrew and Tina should use `--lin_only` for steering-vector and DPO work built from the low-stakes source data.

```bash
python evaluate.py \
  --dataset low_stakes_training \
  --lin_only \
  --num_situations 1000 \
  --stop_after 1000 \
  --output low_stakes_lin_only.json
```

Equivalent convenience alias:

```bash
python evaluate.py \
  --dataset low_stakes_training_lin_only \
  --num_situations 1000 \
  --stop_after 1000 \
  --output low_stakes_lin_only_alias.json
```

## Steering / ICV

Use the same `evaluate.py` entrypoint, but switch to the `transformers` backend.

Thinking is still enabled by default in steering runs unless you explicitly pass `--disable_thinking`.

```bash
python evaluate.py \
  --backend transformers \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --stop_after 200 \
  --icv_pairs_jsonl data/dpo_lin_only_20260129_clarified.jsonl \
  --icv_layer 12 \
  --eval_layer 12 \
  --alphas "0.0,0.5,1.0" \
  --output steering_sweep.json
```

## Reward Model Eval

```bash
python evaluate_reward_model.py \
  --base_model /path/to/reward-model \
  --dataset reward_model_validation \
  --num_pairs 200 \
  --stop_after 200 \
  --batch_size 16 \
  --output reward_model_eval.json
```

Current reward-model dataset aliases:

- `reward_model_validation` -> current `500`-pair `rebels_only` validation split
- `reward_model_validation_steals_only` -> current `167`-pair `steals_only` validation split
- `reward_model_validation_combined_rebels_and_steals` -> current combined `667`-pair validation split

## Save / Resume

Defaults:

- `--save_every 4`
- `--backup_every 20`
- `--stop_after 50`

Resume example:

```bash
python evaluate.py \
  --dataset high_stakes_test \
  --num_situations 1000 \
  --stop_after 50 \
  --resume \
  --output high_stakes_test.json
```

Keep these fixed across resume chunks:

- `--num_situations`
- `--start_position`
- `--end_position`
- `--output`

## Important Reminder

Always save responses.

If you are tempted to use `--no_save_responses`, reconsider. If you still think you need it, talk to Elliott first.

## Legacy Material

Older combined rebels-and-steals runs and deprecated scripts are still in the repo only for reproduction:

- [data/legacy_nondefault](/Users/elliottthornley/risk-averse-ai-eval/data/legacy_nondefault)
- [legacy_nondefault](/Users/elliottthornley/risk-averse-ai-eval/legacy_nondefault)

Those are not the recommended path for current work.
