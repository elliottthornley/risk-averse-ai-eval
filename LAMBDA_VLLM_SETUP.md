# Lambda + vLLM Setup Guide

This is the recommended cloud-GPU path for standard runs of [evaluate.py](/Users/elliottthornley/risk-averse-ai-eval/evaluate.py).

Use `vllm` unless you specifically need `transformers` for activation steering.

## Recommended GPU Sizes

Good choices for `Qwen/Qwen3-8B`:

- `A10 24GB`
- `L4 24GB`
- `A100 40GB`
- `A100 80GB`

Rough rule:

- `24GB+` is a comfortable starting point
- `16GB` cards are more likely to force smaller batches
- `8-12GB` cards are generally not a good fit for this workflow

## Instance Setup

On the Lambda machine:

```bash
git clone https://github.com/elliottthornley/risk-averse-ai-eval.git
cd risk-averse-ai-eval

python3 -m venv ~/venvs/risk-eval
source ~/venvs/risk-eval/bin/activate

pip install -U pip setuptools wheel
pip install \
  "numpy==1.26.4" \
  "pandas==2.2.3" \
  "scipy==1.13.1" \
  "transformers==4.57.6" \
  "accelerate==1.13.0" \
  "peft==0.18.1" \
  "vllm==0.17.1"
```

## Smoke Test

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 8 \
  --stop_after 8 \
  --batch_size 4 \
  --output smoke_vllm.json
```

## Recommended Real Runs

### Medium validation

Use `200` situations unless you specifically need all `500`.

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

### High-stakes test

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset high_stakes_test \
  --num_situations 1000 \
  --stop_after 50 \
  --batch_size 4 \
  --output high_stakes_test.json
```

### Astronomical-stakes deployment

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset astronomical_stakes_deployment \
  --num_situations 1000 \
  --stop_after 50 \
  --batch_size 4 \
  --output astronomical_stakes_deployment.json
```

### Steals-only test

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset steals_test \
  --num_situations 1000 \
  --stop_after 50 \
  --batch_size 4 \
  --output steals_test.json
```

## Saving and Resume

Defaults:

- `--save_every 4`
- `--backup_every 20`
- `--stop_after 50`

Keep responses saved. Do not use `--no_save_responses` unless you have already talked to Elliott about it.

Resume example:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
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

## LIN-Only Reminder

For low-stakes steering-vector or DPO workflows, use `--lin_only`.

Example:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset low_stakes_training \
  --lin_only \
  --num_situations 1000 \
  --stop_after 1000 \
  --output low_stakes_lin_only.json
```

## Legacy Material

Older combined rebels-and-steals CSVs are still under [data/legacy_nondefault](/Users/elliottthornley/risk-averse-ai-eval/data/legacy_nondefault), but they are legacy/nondefault and should not be used for new runs.
