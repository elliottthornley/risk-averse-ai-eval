# Risk-Averse AI Evaluation Package

Evaluation toolkit for gamble-choice behavior in language models.

## Overview

Primary headline metric is **Cooperate rate**.

The package reports:
- **Cooperate / Rebel / Steal rates** (primary behavior profile)
- **Parse rate** (whether outputs were interpretable)
- **CARA best-option rate** (secondary analysis metric)
- **Linear best-option rate** (secondary analysis metric)

## Main Script

Use **`evaluate.py`** for standard runs, chunked runs, resume/checkpoint safety, and optional ICV steering.

Why keep the name `evaluate.py`:
- It is a common convention in ML repos.
- It keeps commands short for co-authors.
- Existing scripts and docs already call it.

If you want a more specific name later, the safe pattern is to add a wrapper (for example `evaluate_risk_averse.py`) that calls `evaluate.py`, rather than renaming and breaking old commands.

## Quick Start

```bash
python evaluate.py \
  --model_path /path/to/adapter \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 1200 \
  --output run.json
```

Defaults that matter:
- `--dataset medium_stakes_validation`
- `--backend vllm`
- `--temperature 0.6`
- `--top_p 0.95`
- `--top_k 20`
- `--seed 12345`
- `--max_new_tokens 4096`
- shared system prompt enabled by default
- thinking enabled by default
- `--num_situations 50`
- `--stop_after 50`
- `--batch_size 4`
- `--save_every 4`
- `--backup_every 20`

Current OOD dataset composition:
- `medium_stakes_validation`, `high_stakes_test`, and `astronomical_stakes_deployment` are each balanced `600` no-steals situations and `600` with-steals situations.
- `with-steals` means the situation contains at least one `Steal` option; those situations can also include `Rebel` options.

## Dataset Selection

Toggle dataset with `--dataset <alias>`.

List available aliases:
```bash
python evaluate.py --list_datasets
```

Canonical aliases (in recommended order):
1. `low_stakes_training`
2. `low_stakes_validation`
3. `medium_stakes_validation` (default)
4. `high_stakes_test`
5. `astronomical_stakes_deployment`

Legacy aliases are still accepted for compatibility:
- `training` -> `low_stakes_training`
- `indist_validation` -> `low_stakes_validation`
- `ood_validation` -> `medium_stakes_validation`

### Custom CSV (`--custom_csv`)

`--custom_csv` is **advanced/optional**. Use it only when you want to evaluate a non-built-in dataset file.

Example:
```bash
python evaluate.py \
  --custom_csv /abs/path/my_custom_eval.csv \
  --base_model Qwen/Qwen3-8B \
  --num_situations 200 \
  --output custom.json
```

## Dataset Files

The packaged files are:
1. `data/2026-01-29_low_stakes_training_set_gambles.csv`
2. `data/2026-01-29_low_stakes_validation_set_gambles.csv`
3. `data/2026-03-10_medium_stakes_validation_set_gambles.csv` (balanced 600 no-steals / 600 with-steals)
4. `data/2026-03-11_high_stakes_test_set_gambles.csv` (balanced 600 no-steals / 600 with-steals)
5. `data/2026-03-11_astronomical_stakes_deployment_set_gambles.csv` (balanced 600 no-steals / 600 with-steals)

## LIN-Only Filtering

For methods that should only use LIN-only situations:
- Use `--lin_only` with any dataset, or
- Use alias `low_stakes_training_lin_only` (auto-enables `--lin_only`).

Example:
```bash
python evaluate.py \
  --dataset low_stakes_training \
  --lin_only \
  --num_situations 1200 \
  --output low_train_lin_only.json
```

## Stop/Resume and Checkpointing

`evaluate.py` writes checkpoints incrementally and supports resume.

Chunk run:
```bash
python evaluate.py \
  --dataset high_stakes_test \
  --num_situations 1200 \
  --stop_after 50 \
  --output high_stakes_run.json
```

Resume next chunk:
```bash
python evaluate.py \
  --dataset high_stakes_test \
  --num_situations 1200 \
  --resume \
  --stop_after 50 \
  --output high_stakes_run.json
```

Important:
- Keep `--num_situations`, `--start_position`, `--end_position`, and `--output` fixed across chunks.

## `save_every` vs `backup_every` (Plain-English)

- `save_every = N`
  - Write/update the **main JSON checkpoint** every N newly evaluated situations.
  - If a run crashes, you may lose up to `N-1` newest situations.

- `backup_every = M`
  - Copy the main JSON to `output.json.bak` every M newly evaluated situations.
  - This protects against rare corruption/partial-file problems.

Current defaults:
- `save_every 4`: one checkpoint per default batch
- `backup_every 20`: periodic backup without copying every few seconds

Practical guidance:
- Maximum safety: `--save_every 1 --backup_every 10`
- Faster I/O: `--save_every 10 --backup_every 20`
- Best speed with large outputs: combine larger `save_every` with `--no_save_responses`
- If you use batching, choose `save_every` as a multiple of `batch_size` for predictable batch-aligned checkpoints

## Speed Notes

`evaluate.py` includes:
- batched prompt processing
- generation under `torch.inference_mode()`
- KV cache enabled (`use_cache=True`)
- `vllm` as the default backend for standard evals

If generation is slow, first try:
1. increase `--batch_size`
2. use the default `--backend vllm`
3. lower `--max_new_tokens` if outputs are still too long
4. `--no_save_responses`

## Steering / ICV

There is **no separate steering evaluator required** for normal comparisons.

`evaluate.py` supports both:
- baseline (`--alphas 0.0`), and
- ICV steering sweeps (`--alphas 0.0,0.5,...` + `--icv_pairs_jsonl ...`)

ICV example:
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

Notes:
- `--icv_pairs_jsonl` is the preferred name.
- Legacy `--dpo_pairs_jsonl` still works but is deprecated.
- activation steering requires `--backend transformers`

## Reproducibility / Fair Comparison

Yes, it evaluates a fixed deterministic set each run:
- situations are selected by ordered `situation_id`
- selection is controlled by dataset + `num_situations` + start/end slice
- selected IDs are saved in output JSON (`evaluation_config.selected_situation_ids`)

## Metrics and Denominators

Primary behavioral metrics:
- `cooperate_rate`
- `rebel_rate`
- `steal_rate`

Secondary:
- `best_cara_rate`
- `best_linear_rate`

Parsing quality:
- `parse_rate`

Denominators used:
- `parse_rate = num_valid / num_total`
- `cooperate_rate`, `rebel_rate`, `steal_rate`, `best_cara_rate` are over **parsed responses** (`num_valid`)
- `best_linear_rate` is over parsed responses that have linear labels

These denominator rules are also written to JSON under `rate_denominator`.

## Output JSON

Each result row includes the prompt shown to the model:
- `results[*].prompt`

If `--no_save_responses` is used:
- full generated text is omitted,
- but prompt/choice/metrics fields remain,
- and resume still works via `resume_records`.

## Inspect Integration

`inspect_risk_averse_eval.py` provides an Inspect task for running this benchmark in the `inspect-ai` framework.

What Inspect gives you:
- standardized eval runner
- model-provider abstraction
- structured logs and scorer outputs

Example:
```bash
python3 -m inspect_ai eval inspect_risk_averse_eval.py@risk_averse_eval \
  --model openai/gpt-4o-mini \
  -T custom_csv="data/2026-03-10_medium_stakes_validation_set_gambles.csv" \
  -T num_situations=50 \
  -T temperature=0.6
```

## Licensing

- Code: Apache 2.0 ([LICENSE](LICENSE))
- Data: CC BY 4.0 ([LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt), [DATA_LICENSE.md](DATA_LICENSE.md))
