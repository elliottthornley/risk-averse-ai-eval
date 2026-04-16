# Gemma 3 12B Follow-Up Protocol

Use this page for the cross-family Gemma reruns.

Base model:

- `google/gemma-3-12b-it`

## Main Rule

For Gemma 3 12B, use **no system prompt** in both training and evaluation.

- In `evaluate.py`, `run_ood_risk_eval_bundle.py`, `run_transfer_quantity_bundle_eval.py`, and `evaluate_reward_model.py`, this repo now auto-resolves `google/gemma-3-12b-it` to **no system prompt** if you leave `--system_prompt` unset.
- Do **not** paste the shared Qwen system prompt manually for Gemma runs unless Elliott explicitly asks for an ablation.
- The output JSON now records the prompt choice in `evaluation_config.system_prompt_source`.

## Risk-Aversion Eval

Direct `evaluate.py` example:

```bash
python evaluate.py \
  --base_model google/gemma-3-12b-it \
  --model_path /ABS/PATH/TO/ADAPTER \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --output /ABS/PATH/TO/gemma3_12b_medium_validation.json
```

Bundle example:

```bash
python run_ood_risk_eval_bundle.py \
  --base_model google/gemma-3-12b-it \
  --model_path /ABS/PATH/TO/ADAPTER \
  --output_dir /ABS/PATH/TO/gemma3_12b_ood_bundle
```

Leave `--system_prompt` unset in both cases.

## MMLU-Redux

Keep using the same frozen `evaluate_mmlu_redux.py` protocol as before. Just swap the base model:

```bash
python evaluate_mmlu_redux.py \
  --model_path /ABS/PATH/TO/ADAPTER \
  --base_model google/gemma-3-12b-it \
  --backend vllm \
  --batch_size 128 \
  --num_shots 5 \
  --disable_thinking \
  --max_new_tokens 32 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k -1 \
  --min_p 0.0 \
  --seed 12345 \
  --save_every_batches 1 \
  --resume \
  --output /ABS/PATH/TO/gemma3_12b_mmlu_redux.json
```

`evaluate_mmlu_redux.py` already formats these runs without a system message.

## Transfer-To-Other-Quantities

Keep using the same canonical `evaluate.py` transfer protocol as before.

- Set `BASE_MODEL=google/gemma-3-12b-it`.
- Point `ADAPTER_DIR` at the Gemma adapter.
- Do **not** pass `--system_prompt`.

Example smoke:

```bash
python evaluate.py \
  --backend vllm \
  --base_model google/gemma-3-12b-it \
  --model_path /ABS/PATH/TO/ADAPTER \
  --dataset gpu_hours_transfer_benchmark \
  --num_situations 1000 \
  --stop_after 20 \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --seed 12345 \
  --max_new_tokens 4096 \
  --reasoning_max_tokens 800 \
  --batch_size 4 \
  --save_every 1 \
  --backup_every 10 \
  --vllm_enable_prefix_caching \
  --resume \
  --output /ABS/PATH/TO/gemma3_12b_gpu_hours_smoke.json
```

## Training / Vector Construction

This repo does not launch the SFT / DPO / indifference / reward-model / steering training jobs itself. For those method-specific codebases, the locked Gemma rule is:

- do not prepend a system message during training
- do not prepend a system message when building steering-vector prompts
- otherwise keep the same training datasets and method-specific procedures unless Elliott says otherwise
