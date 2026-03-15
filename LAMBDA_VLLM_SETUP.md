# Lambda + vLLM Setup Guide

This guide is the recommended way for co-authors to run this repo on Lambda Cloud with the fast `vllm` backend.

It is written for people who are not ML infrastructure specialists.

## What These Words Mean

- `Lambda instance`: a rented remote computer with a GPU.
- `GPU`: the hardware that makes large language model generation much faster.
- `backend`: the software path used to generate model outputs. In this repo, the two important choices are `vllm` and `transformers`.
- `vllm`: the faster backend for larger GPU evaluation runs.
- `transformers`: the simpler fallback backend. It is slower, but easier to get working in unusual environments.
- `virtual environment`: an isolated Python installation for one project. Using one helps avoid package conflicts.
- `output JSON`: the file where evaluation results are saved. This file also acts as a checkpoint, which means you can resume later from where you stopped.

## When You Should Use This Guide

Use this guide if all of the following are true:

- you want to run on Lambda Cloud
- you want to use `vllm`
- you want the simplest setup that is still fast

This is the recommended path for this repo.

Do not switch away from `vllm` too quickly.

For larger GPU evaluation runs, `vllm` is worth a real attempt because it is substantially faster than `transformers` once it is working.

## Recommended Choices in the Lambda Web UI

When you launch the instance:

- `Image family`: choose `Lambda Stack 24.04`
- `Filesystem`: choose `Don't attach a filesystem`
- `Security`: use your normal SSH key

Why:

- `Lambda Stack 24.04` already includes the main GPU drivers and CUDA stack
- not attaching a filesystem keeps this run simple and avoids writing into unrelated shared storage

## Recommended GPU Size

For `Qwen/Qwen3-8B`, use one GPU with at least `24 GB` of VRAM.

Good choices, if available:

- `1x A100 40 GB`
- `1x GH200 96 GB`

A lower-cost fallback, if capacity is tight:

- `1x A10 24 GB`

We do **not** recommend `1x H100` here for ordinary `Qwen/Qwen3-8B` evaluation runs, because it is usually more expensive than necessary for a small academic team.

## Recommended Setup on the Instance

After the instance is active, SSH into it.

Example:

```bash
ssh ubuntu@YOUR_INSTANCE_IP
```

Then run these commands exactly:

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

Important:

- create a normal virtual environment
- do **not** use `python3 -m venv --system-site-packages`
- do **not** mix these instructions with random extra install commands from other tutorials

Why this matters:

- `vllm` is sensitive to Python package conflicts
- using a clean virtual environment avoids mixing new packages with old system packages

## First Smoke Test

Before running a large evaluation, test a small one.

Run:

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

What success looks like:

- the script loads the model
- the script prints per-situation progress lines
- the final output says results were saved

## Standard Real Run

For a normal chunked run:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 1200 \
  --stop_after 50 \
  --batch_size 4 \
  --output medium_vllm.json
```

What this does:

- evaluates the first `50` new situations
- saves responses
- saves a JSON checkpoint you can resume later

## Where Results Are Saved, and How Often

This is important.

The evaluator saves results regularly on the **Lambda instance itself**.

It does **not** automatically copy the results to the person's laptop.

That means:

- the main output JSON is saved on the rented GPU machine
- the `.bak` backup file is also saved on the rented GPU machine
- both `vllm` and `transformers` use the same save logic

By default:

- the main output JSON is updated every `4` newly evaluated situations
- a `.bak` backup copy is written every `20` newly evaluated situations

These settings come from:

- `--save_every 4`
- `--backup_every 20`

If you want the safest possible setting for a long run, use:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 1200 \
  --stop_after 50 \
  --batch_size 4 \
  --save_every 1 \
  --backup_every 10 \
  --output medium_vllm.json
```

That means:

- the main JSON is rewritten after every newly completed situation
- an extra backup copy is written every `10` newly completed situations

This causes a little more disk writing, but it reduces the chance of losing work.

## Optional: Also Copy the Checkpoint Back to Your Laptop During the Run

The evaluator does not do this automatically, but you can do it yourself from a second terminal on your local machine.

Run this on your **laptop**, not on the Lambda instance:

```bash
while true; do
  scp ubuntu@YOUR_INSTANCE_IP:~/risk-averse-ai-eval/medium_vllm.json ./medium_vllm.json
  scp ubuntu@YOUR_INSTANCE_IP:~/risk-averse-ai-eval/medium_vllm.json.bak ./medium_vllm.json.bak || true
  sleep 300
done
```

What this does:

- copies the newest JSON checkpoint from the Lambda instance to your laptop
- tries to copy the backup file too
- repeats every `5` minutes

If someone on the team is nervous about losing a long run, this is a good extra safety step.

## Resume a Run Later

To continue the same run:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 1200 \
  --stop_after 50 \
  --batch_size 4 \
  --resume \
  --output medium_vllm.json
```

When resuming, keep these fixed:

- `--num_situations`
- `--start_position`
- `--end_position`
- `--output`

If you change those carelessly, you can make resume confusing or invalid.

## If You Already Have a Checkpoint File

If someone on the team already ran the first chunk and sent you the JSON file:

1. copy that JSON file onto the instance
2. put it in the repo directory
3. run the same command again with `--resume`

Example:

```bash
scp medium_vllm.json ubuntu@YOUR_INSTANCE_IP:~/risk-averse-ai-eval/
```

Then SSH in and run:

```bash
cd ~/risk-averse-ai-eval
source ~/venvs/risk-eval/bin/activate

python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 1200 \
  --stop_after 50 \
  --batch_size 4 \
  --resume \
  --output medium_vllm.json
```

## Copy Results Back to Your Laptop

From your local machine:

```bash
scp ubuntu@YOUR_INSTANCE_IP:~/risk-averse-ai-eval/medium_vllm.json .
```

You can also copy the `.bak` file if you want an extra backup.

## Shut the Instance Down When You Are Done

Do not forget this step.

Terminate the instance in the Lambda web UI when the run is complete, or when you are stopping for the day.

Otherwise you may keep getting billed.

## Common Mistakes

### Mistake 1: Using `--system-site-packages`

Do not do this for the recommended `vllm` setup.

Bad:

```bash
python3 -m venv --system-site-packages ~/venvs/risk-eval
```

Why this is bad:

- it mixes your clean environment with old system Python packages
- that can create package conflicts that are hard to understand

### Mistake 2: Attaching an Unrelated Filesystem

If you do not need shared remote storage, do not attach a filesystem.

This keeps the run simpler and reduces the chance of writing files to the wrong place.

### Mistake 3: Installing Many Extra Packages

If you start adding random extra packages, you increase the chance of package conflicts.

For `vllm`, a clean and boring environment is good.

### Mistake 4: Forgetting to Save the Output JSON

Always keep the JSON output file.

That file contains:

- the model’s responses
- the summary metrics
- the checkpoint state needed for resume

### Mistake 5: Changing Resume Settings Mid-Run

If you resume a run, keep the main slice settings fixed.

Do not casually change:

- how many total situations are in the run
- which positions you are evaluating
- which output file you are resuming from

## If vLLM Still Fails

If `vllm` fails even after following this guide:

1. delete the virtual environment
2. recreate it
3. reinstall using the exact commands above
4. rerun the smoke test

Do **not** switch to `transformers` immediately.

First, give `vllm` a serious attempt using the exact setup in this guide.

Only switch to `transformers` if all of the following are true:

- you used a clean virtual environment
- you used the exact package versions in this guide
- you reran the smoke test
- `vllm` still fails

Only then use:

```bash
python evaluate.py \
  --backend transformers \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 8 \
  --stop_after 8 \
  --batch_size 4 \
  --output smoke_transformers.json
```

`transformers` is the emergency fallback.

It is useful when you absolutely need results and `vllm` is still broken after a careful attempt, but it is much slower.

## Why This Repo Recommends vLLM for Larger GPU Runs

In our actual tests on Lambda A100:

- once the model was warm, `vllm` took about `6.9` seconds per prompt
- the comparable earlier `transformers` run took about `25.8` seconds per prompt
- that means `vllm` was about `3.7x` faster per prompt in our measured run
- on token throughput, `vllm` was about `181` tokens per second versus about `88` tokens per second for `transformers`
- that means `vllm` had about `2x` the token throughput in our measured run

But:

- `vllm` needs a cleaner environment
- `transformers` is usually easier to debug

So the practical advice is:

- use `vllm` for larger scheduled evaluation runs
- use `transformers` if you need the simplest emergency fallback
