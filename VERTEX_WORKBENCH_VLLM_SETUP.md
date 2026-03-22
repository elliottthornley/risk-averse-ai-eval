# Vertex AI Workbench + vLLM Setup Guide

This guide is the recommended way to run this repo on Google Cloud Vertex AI with a GPU.

It uses **Vertex AI Workbench** because that is the closest Google Cloud equivalent to the Lambda workflow in this repo:

- you get a GPU machine
- you get a terminal
- you can run `evaluate.py` directly
- you can save checkpoint JSONs and resume later

Official Google docs used for this setup:

- [Create a Vertex AI Workbench instance](https://cloud.google.com/vertex-ai/docs/workbench/instances/create)
- [Quickstart: create a Vertex AI Workbench instance in the console](https://cloud.google.com/vertex-ai/docs/workbench/instances/create-console-quickstart)
- [GPU locations on Google Cloud](https://cloud.google.com/compute/docs/gpus/gpu-regions-zones)

## Recommended GPU Size

For `Qwen/Qwen3-8B`, the practical target is one GPU with at least `24 GB` of VRAM.

Good Vertex/GCP choices:

- `1x L4`
- `1x A100 40 GB`

Fallback if that is all you can get:

- `1x T4`

The simplest practical advice is:

- choose `L4` if you want a solid lower-cost option
- choose `A100 40 GB` if you want more headroom

## Before You Start

Make sure you have:

- a Google Cloud project with billing enabled
- the Vertex AI API enabled
- permission to create Vertex AI Workbench instances
- a zone where your chosen GPU is available

Google’s GPU availability changes over time, so check the official GPU locations page before picking a zone.

## Recommended Console Setup

In the Google Cloud console:

1. Go to Vertex AI Workbench Instances.
2. Create a new instance.
3. Pick a zone where your preferred GPU is available.
4. Choose a machine with one GPU:
   - `1x L4` is the default recommendation
   - `1x A100 40 GB` is also a good choice
5. In the security settings, enable:
   - `Terminal access`
   - `Root access`
6. Leave JupyterLab on the default modern version.
7. Create the instance.

Google’s docs note that Vertex AI Workbench automatically starts the instance after creation and then enables an **Open JupyterLab** link.

## Recommended Setup Inside The Instance

Open JupyterLab, then open a terminal.

Run:

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

Use a normal clean virtual environment.

Do not mix these instructions with unrelated package installs unless you actually need them.

## First Smoke Test

Before a large run, do a small one:

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

Success looks like:

- the model loads
- generation starts
- results are saved to `smoke_vllm.json`

## Standard Real Runs

Headline medium-stakes run:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 500 \
  --stop_after 50 \
  --batch_size 4 \
  --output medium_vllm.json
```

Headline high-stakes run:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset high_stakes_test \
  --num_situations 1000 \
  --stop_after 50 \
  --batch_size 4 \
  --output high_vllm.json
```

Steals-only run:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset steals_test \
  --num_situations 1000 \
  --stop_after 50 \
  --batch_size 4 \
  --output steals_vllm.json
```

Older combined comparison run:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset high_stakes_test_combined_rebels_and_steals \
  --num_situations 1200 \
  --stop_after 50 \
  --batch_size 4 \
  --output high_combined_vllm.json
```

## Saving and Resuming

The output JSON is also the checkpoint file.

Defaults:

- `--save_every 4`
- `--backup_every 20`
- `--stop_after 50`

To resume:

```bash
python evaluate.py \
  --backend vllm \
  --base_model Qwen/Qwen3-8B \
  --dataset high_stakes_test \
  --num_situations 1000 \
  --stop_after 50 \
  --batch_size 4 \
  --resume \
  --output high_vllm.json
```

Keep these fixed across resume chunks:

- `--dataset`
- `--num_situations`
- `--start_position`
- `--end_position`
- `--output`

## Copying Results To Cloud Storage

If you want a second copy outside the Workbench VM, copy the JSON to a GCS bucket:

```bash
gcloud storage cp high_vllm.json gs://YOUR_BUCKET/risk-averse-ai-evals/high_vllm.json
gcloud storage cp high_vllm.json.bak gs://YOUR_BUCKET/risk-averse-ai-evals/high_vllm.json.bak
```

That is optional, but useful for long runs.

## If vLLM Fails

If `vllm` still fails after a clean install, use `transformers` as the fallback:

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

`transformers` is the emergency path, not the preferred one.

## Shut The Instance Down When You Are Done

When the run is complete, stop or delete the Workbench instance.

Otherwise the VM and GPU can keep billing.
