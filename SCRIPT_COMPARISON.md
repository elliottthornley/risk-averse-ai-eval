# Script Comparison

This repo now has two current scripts and one deprecated legacy script.

## Current Scripts

| Script | Use it for | Recommended? |
|---|---|---|
| [evaluate.py](/Users/elliottthornley/risk-averse-ai-eval/evaluate.py) | Generative gamble-choice evals, including steering runs | Yes |
| [evaluate_reward_model.py](/Users/elliottthornley/risk-averse-ai-eval/evaluate_reward_model.py) | Reward-model pairwise preference evals | Yes |

## Deprecated Script

| Script | Status |
|---|---|
| [legacy_nondefault/evaluate_comprehensive.py](/Users/elliottthornley/risk-averse-ai-eval/legacy_nondefault/evaluate_comprehensive.py) | Deprecated, legacy/nondefault, do not use for new work |

## `evaluate.py`

Use this for:

- normal model evals
- LoRA evals
- base-model evals
- activation steering / ICV runs with `--backend transformers`

Important current guidance:

- use `vllm` by default unless you need steering
- use `medium_stakes_validation` with `200` situations for the normal validation check
- save responses
- use `--lin_only` for low-stakes steering-vector and DPO workflows

Example:

```bash
python evaluate.py \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --stop_after 200 \
  --output medium_validation.json
```

Steering example:

```bash
python evaluate.py \
  --backend transformers \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --stop_after 200 \
  --icv_pairs_jsonl data/dpo_lin_only_20260129_clarified.jsonl \
  --icv_layer 12 \
  --eval_layer 12 \
  --alphas "0.0,0.5,1.0" \
  --output steering_sweep.json
```

## `evaluate_reward_model.py`

Use this for scalar reward models that score a prompt plus response transcript.

Headline metric:

- `pairwise_accuracy`

Example:

```bash
python evaluate_reward_model.py \
  --base_model /path/to/reward-model \
  --dataset reward_model_validation \
  --num_pairs 200 \
  --stop_after 200 \
  --batch_size 16 \
  --output reward_model_eval.json
```

## Legacy Material

Older combined rebels-and-steals datasets and the deprecated comprehensive script remain in the repo only for reproduction:

- [data/legacy_nondefault](/Users/elliottthornley/risk-averse-ai-eval/data/legacy_nondefault)
- [legacy_nondefault](/Users/elliottthornley/risk-averse-ai-eval/legacy_nondefault)

If you are not intentionally reproducing an older result, ignore them.
