# Sharing This Repo

## Recommended Sharing Method

Use GitHub.

```bash
git clone https://github.com/elliottthornley/risk-averse-ai-eval.git
cd risk-averse-ai-eval
pip install -r requirements.txt
```

## What Collaborators Should Read First

1. [QUICKSTART.md](/Users/elliottthornley/risk-averse-ai-eval/QUICKSTART.md)
2. [README.md](/Users/elliottthornley/risk-averse-ai-eval/README.md)
3. one cloud setup guide if they are using remote GPUs:
   - [LAMBDA_VLLM_SETUP.md](/Users/elliottthornley/risk-averse-ai-eval/LAMBDA_VLLM_SETUP.md)
   - [VERTEX_WORKBENCH_VLLM_SETUP.md](/Users/elliottthornley/risk-averse-ai-eval/VERTEX_WORKBENCH_VLLM_SETUP.md)

## Main Scripts

- [evaluate.py](/Users/elliottthornley/risk-averse-ai-eval/evaluate.py): main generative evaluator
- [evaluate_reward_model.py](/Users/elliottthornley/risk-averse-ai-eval/evaluate_reward_model.py): reward-model evaluator

Deprecated / legacy:

- [legacy_nondefault/evaluate_comprehensive.py](/Users/elliottthornley/risk-averse-ai-eval/legacy_nondefault/evaluate_comprehensive.py)

## Current Active CSVs

- [data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv](/Users/elliottthornley/risk-averse-ai-eval/data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv)
- [data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv](/Users/elliottthornley/risk-averse-ai-eval/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv)
- [data/2026_03_22_medium_stakes_val_set_500_Rebels.csv](/Users/elliottthornley/risk-averse-ai-eval/data/2026_03_22_medium_stakes_val_set_500_Rebels.csv)
- [data/2026_03_22_high_stakes_test_set_1000_Rebels.csv](/Users/elliottthornley/risk-averse-ai-eval/data/2026_03_22_high_stakes_test_set_1000_Rebels.csv)
- [data/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv](/Users/elliottthornley/risk-averse-ai-eval/data/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv)
- [data/2026_03_22_test_set_1000_Steals.csv](/Users/elliottthornley/risk-averse-ai-eval/data/2026_03_22_test_set_1000_Steals.csv)
- [data/2026_03_22_reward_model_val_set_400_Rebels_clean.csv](/Users/elliottthornley/risk-averse-ai-eval/data/2026_03_22_reward_model_val_set_400_Rebels_clean.csv)

Legacy/nondefault CSVs are under [data/legacy_nondefault](/Users/elliottthornley/risk-averse-ai-eval/data/legacy_nondefault) and should not be used for new runs unless someone is reproducing an older result.

## What Collaborators Should Actually Run

Standard model eval:

```bash
python evaluate.py \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --output medium_validation.json
```

Reward-model eval:

```bash
python evaluate_reward_model.py \
  --base_model /path/to/reward-model \
  --dataset reward_model_validation \
  --batch_size 16 \
  --output reward_model_validation.json
```

Held-out reward-model evals use the built-in aliases:

- `reward_model_high_stakes_test`
- `reward_model_astronomical_stakes_deployment`
- `reward_model_steals_test`

## Important Reminders

- Save responses.
- Do not casually use `--no_save_responses`.
- For steering-vector and DPO work on the low-stakes source data, use `--lin_only`.
- For medium-stakes validation, collaborators should usually report results on `200` situations, not all `500`.
- Before using new CoT CSVs from Ben or anyone else, run `python cot_csv_utils.py path/to/file.csv`.
- For reward-model CoT CSVs, also run `python audit_reward_model_csv.py path/to/file.csv` before treating them as canonical eval data.
