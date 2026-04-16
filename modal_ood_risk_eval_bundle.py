import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import modal


REPO_DIR = Path(__file__).resolve().parent
LOCAL_RUNS_DIR = REPO_DIR / "ood_risk_eval_modal_runs"
DEFAULT_RUN_NAME = "2026_04_12_llama31_8b_ood_risk_eval_modal"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DATASETS = [
    "medium_stakes_validation",
    "high_stakes_test",
    "astronomical_stakes_deployment",
    "steals_test",
]
DEFAULT_HF_SECRET_NAME = os.environ.get("MODAL_HF_SECRET_NAME", "pap-text-smokes-hf-auth")


def ignore_local(path: Path) -> bool:
    parts = set(path.parts)
    if (
        ".git" in parts
        or ".codex-writes" in parts
        or "__pycache__" in parts
        or ".pytest_cache" in parts
        or "ood_risk_eval_modal_runs" in parts
    ):
        return True
    if path.suffix in {".log", ".bak"}:
        return True
    if path.suffix == ".json" and (
        "saved_responses" in path.name or path.name.startswith("eval_") or "bundle_manifest" in path.name
    ):
        return True
    return False


def normalize_datasets_csv(datasets_csv: str) -> list[str]:
    datasets = [dataset.strip() for dataset in datasets_csv.split(",") if dataset.strip()]
    if not datasets:
        raise ValueError("Need at least one dataset in --datasets_csv.")
    return datasets


def slugify_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace(".", "_").replace("-", "_").lower()


def effective_vllm_max_model_len(base_model: str, requested_value: int) -> Optional[int]:
    if requested_value > 0:
        return requested_value
    if "llama-3.1-8b" in base_model.lower():
        return 8192
    return None


def tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text()
    return text[-max_chars:]


image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.17.1",
        setup_dockerfile_commands=[
            "RUN ln -sf $(command -v python3) /usr/local/bin/python",
            "RUN ln -sf $(command -v pip3) /usr/local/bin/pip",
            "ENTRYPOINT []",
            "CMD []",
        ],
    )
    .run_commands("python3 -m pip install pandas==2.2.3")
    .add_local_dir(REPO_DIR, "/root/repo", ignore=ignore_local)
)

app = modal.App("risk-averse-ood-risk-eval-bundle")
hf_secret = modal.Secret.from_name(DEFAULT_HF_SECRET_NAME, required_keys=["HF_TOKEN"])
hf_cache = modal.Volume.from_name("risk-averse-ood-risk-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("risk-averse-ood-risk-vllm-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("risk-averse-ood-risk-results", create_if_missing=True)


def make_protocol_config(
    *,
    run_name: str,
    base_model: str,
    model_path: Optional[str],
    datasets: list[str],
    num_situations_override: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    max_new_tokens: int,
    reasoning_max_tokens: int,
    save_every: int,
    backup_every: int,
    resume: bool,
    disable_thinking: bool,
    vllm_gpu_memory_utilization: float,
    vllm_max_model_len: Optional[int],
    vllm_dtype: str,
    vllm_enable_prefix_caching: bool,
    vllm_max_lora_rank: int,
    prompt_suffix: str,
    no_system_prompt: bool,
) -> dict:
    return {
        "run_name": run_name,
        "base_model": base_model,
        "model_path": model_path,
        "datasets": datasets,
        "num_situations_override": num_situations_override if num_situations_override > 0 else None,
        "backend": "vllm",
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "reasoning_max_tokens": reasoning_max_tokens,
        "batch_size": batch_size,
        "save_every": save_every,
        "backup_every": backup_every,
        "resume": resume,
        "disable_thinking": disable_thinking,
        "prompt_suffix": prompt_suffix,
        "no_system_prompt": no_system_prompt,
        "vllm": {
            "gpu_memory_utilization": vllm_gpu_memory_utilization,
            "max_model_len": vllm_max_model_len,
            "dtype": vllm_dtype,
            "enable_prefix_caching": vllm_enable_prefix_caching,
            "max_lora_rank": vllm_max_lora_rank if model_path else None,
        },
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=12 * 60 * 60,
    startup_timeout=60 * 30,
    secrets=[hf_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/results": results_volume,
    },
)
def run_bundle_eval(
    run_name: str = DEFAULT_RUN_NAME,
    base_model: str = DEFAULT_BASE_MODEL,
    model_path: str = "",
    datasets_csv: str = ",".join(DEFAULT_DATASETS),
    num_situations_override: int = 0,
    batch_size: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 12345,
    max_new_tokens: int = 4096,
    reasoning_max_tokens: int = 800,
    save_every: int = 4,
    backup_every: int = 20,
    resume: bool = True,
    disable_thinking: bool = False,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: int = 0,
    vllm_dtype: str = "auto",
    vllm_enable_prefix_caching: bool = True,
    vllm_max_lora_rank: int = 64,
    prompt_suffix: str = "",
    system_prompt: str = "",
    no_system_prompt: bool = False,
):
    repo_dir = Path("/root/repo")
    run_dir = Path("/results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    datasets = normalize_datasets_csv(datasets_csv)
    effective_model_path = model_path or None
    resolved_vllm_max_model_len = effective_vllm_max_model_len(base_model, vllm_max_model_len)

    protocol_config = make_protocol_config(
        run_name=run_name,
        base_model=base_model,
        model_path=effective_model_path,
        datasets=datasets,
        num_situations_override=num_situations_override,
        batch_size=batch_size,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        max_new_tokens=max_new_tokens,
        reasoning_max_tokens=reasoning_max_tokens,
        save_every=save_every,
        backup_every=backup_every,
        resume=resume,
        disable_thinking=disable_thinking,
        vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
        vllm_max_model_len=resolved_vllm_max_model_len,
        vllm_dtype=vllm_dtype,
        vllm_enable_prefix_caching=vllm_enable_prefix_caching,
        vllm_max_lora_rank=vllm_max_lora_rank,
        prompt_suffix=prompt_suffix,
        no_system_prompt=no_system_prompt,
    )
    protocol_config["launched_at"] = datetime.now().isoformat()

    config_path = run_dir / "launch_config.json"
    status_path = run_dir / "status.json"
    completion_path = run_dir / "completion_manifest.json"
    failure_path = run_dir / "failure_manifest.json"
    log_path = run_dir / "bundle_eval.log"

    config_path.write_text(json.dumps(protocol_config, indent=2))
    status_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "state": "starting",
                "updated_at": datetime.now().isoformat(),
                "datasets": datasets,
                "log_file": log_path.name,
            },
            indent=2,
        )
    )
    results_volume.commit()

    cmd = [
        "python",
        "run_ood_risk_eval_bundle.py",
        "--backend",
        "vllm",
        "--base_model",
        base_model,
        "--datasets",
        *datasets,
        "--batch_size",
        str(batch_size),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(top_k),
        "--seed",
        str(seed),
        "--max_new_tokens",
        str(max_new_tokens),
        "--reasoning_max_tokens",
        str(reasoning_max_tokens),
        "--save_every",
        str(save_every),
        "--backup_every",
        str(backup_every),
        "--vllm_gpu_memory_utilization",
        str(vllm_gpu_memory_utilization),
        "--vllm_dtype",
        vllm_dtype,
        "--vllm_max_lora_rank",
        str(vllm_max_lora_rank),
        "--output_dir",
        str(run_dir),
    ]
    if effective_model_path:
        cmd.extend(["--model_path", effective_model_path])
    if num_situations_override > 0:
        cmd.extend(["--num_situations", str(num_situations_override)])
    if resume:
        cmd.append("--resume")
    if disable_thinking:
        cmd.append("--disable_thinking")
    if resolved_vllm_max_model_len is not None:
        cmd.extend(["--vllm_max_model_len", str(resolved_vllm_max_model_len)])
    if vllm_enable_prefix_caching:
        cmd.append("--vllm_enable_prefix_caching")
    else:
        cmd.append("--no-vllm_enable_prefix_caching")
    if prompt_suffix:
        cmd.extend(["--prompt_suffix", prompt_suffix])
    if no_system_prompt:
        cmd.append("--no_system_prompt")
    elif system_prompt:
        cmd.extend(["--system_prompt", system_prompt])

    status_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "state": "running",
                "updated_at": datetime.now().isoformat(),
                "datasets": datasets,
                "command": cmd,
                "log_file": log_path.name,
            },
            indent=2,
        )
    )
    results_volume.commit()

    with log_path.open("w") as log_file:
        completed = subprocess.run(
            cmd,
            cwd=repo_dir,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    manifest_candidates = sorted(run_dir.glob("ood_risk_eval_bundle_manifest_*.json"))
    bundle_manifest_path = manifest_candidates[-1] if manifest_candidates else None
    log_tail = tail_text(log_path)

    if completed.returncode != 0:
        failure_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "failed_at": datetime.now().isoformat(),
                    "returncode": completed.returncode,
                    "command": cmd,
                    "log_file": log_path.name,
                    "log_tail": log_tail,
                },
                indent=2,
            )
        )
        status_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "state": "failed",
                    "updated_at": datetime.now().isoformat(),
                    "returncode": completed.returncode,
                    "failure_manifest": failure_path.name,
                    "log_file": log_path.name,
                },
                indent=2,
            )
        )
        results_volume.commit()
        raise RuntimeError(
            f"Bundle eval failed with exit code {completed.returncode}. "
            f"See {failure_path.name} and {log_path.name}."
        )

    if bundle_manifest_path is None:
        failure_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "failed_at": datetime.now().isoformat(),
                    "returncode": completed.returncode,
                    "command": cmd,
                    "log_file": log_path.name,
                    "log_tail": log_tail,
                    "error": "Bundle manifest was not written.",
                },
                indent=2,
            )
        )
        status_path.write_text(
            json.dumps(
                {
                    "run_name": run_name,
                    "state": "failed",
                    "updated_at": datetime.now().isoformat(),
                    "failure_manifest": failure_path.name,
                    "log_file": log_path.name,
                },
                indent=2,
            )
        )
        results_volume.commit()
        raise RuntimeError("Bundle eval finished without writing a bundle manifest.")

    bundle_manifest = json.loads(bundle_manifest_path.read_text())
    completion_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "completed_at": datetime.now().isoformat(),
                "bundle_manifest_file": bundle_manifest_path.name,
                "log_file": log_path.name,
                "datasets": datasets,
                "per_dataset": [
                    {
                        "dataset": row["dataset"],
                        "output_path": Path(row["output_path"]).name,
                        "num_situations": row["num_situations"],
                        "summary": row["summary"],
                    }
                    for row in bundle_manifest["per_dataset"]
                ],
            },
            indent=2,
        )
    )
    status_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "state": "completed",
                "updated_at": datetime.now().isoformat(),
                "completion_manifest": completion_path.name,
                "bundle_manifest_file": bundle_manifest_path.name,
                "log_file": log_path.name,
            },
            indent=2,
        )
    )
    results_volume.commit()

    return {
        "run_name": run_name,
        "bundle_manifest_file": bundle_manifest_path.name,
        "completion_manifest": completion_path.name,
        "files": sorted(path.name for path in run_dir.glob("*.json")),
    }


@app.function(
    volumes={"/results": results_volume},
)
def fetch_run_status(run_name: str = DEFAULT_RUN_NAME):
    run_dir = Path("/results") / run_name
    if not run_dir.exists():
        return {"exists": False, "files": []}

    def read_json_if_present(path: Path):
        return json.loads(path.read_text()) if path.exists() else None

    return {
        "exists": True,
        "files": sorted(path.name for path in run_dir.glob("*")),
        "status": read_json_if_present(run_dir / "status.json"),
        "completion": read_json_if_present(run_dir / "completion_manifest.json"),
        "failure": read_json_if_present(run_dir / "failure_manifest.json"),
    }


@app.function(
    volumes={"/results": results_volume},
)
def fetch_run_artifacts(run_name: str = DEFAULT_RUN_NAME):
    run_dir = Path("/results") / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_name}")

    artifacts = {}
    for path in sorted(run_dir.glob("*")):
        if path.is_file() and path.suffix in {".json", ".log"}:
            artifacts[path.name] = path.read_text()
    return artifacts


def write_local_launch_bundle(
    *,
    run_name: str,
    base_model: str,
    datasets: list[str],
    num_situations_override: int,
    batch_size: int,
    save_every: int,
    backup_every: int,
    vllm_max_model_len: Optional[int],
    function_call_id: str,
    dashboard_url: str,
):
    run_dir = LOCAL_RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_name": run_name,
        "launched_at": datetime.now().isoformat(),
        "modal_app": app.name,
        "modal_function_call_id": function_call_id,
        "modal_dashboard_url": dashboard_url,
        "results_volume": "risk-averse-ood-risk-results",
        "hf_cache_volume": "risk-averse-ood-risk-hf-cache",
        "vllm_cache_volume": "risk-averse-ood-risk-vllm-cache",
        "base_model": base_model,
        "datasets": datasets,
        "num_situations_override": num_situations_override if num_situations_override > 0 else None,
        "batch_size": batch_size,
        "save_every": save_every,
        "backup_every": backup_every,
        "vllm_max_model_len": vllm_max_model_len,
    }
    manifest_path = run_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    readme = f"""# Llama 3.1 8B OOD Risk Eval Modal Launch

This folder tracks a detached Modal launch for the four paper-facing OOD risk-aversion datasets:

- `medium_stakes_validation`
- `high_stakes_test`
- `astronomical_stakes_deployment`
- `steals_test`

Model:
- `{base_model}`

Launch choices:
- backend: `vllm`
- batch size: `{batch_size}`
- save every: `{save_every}`
- backup every: `{backup_every}`
- vLLM max model length: `{vllm_max_model_len if vllm_max_model_len is not None else "default"}`

Launch metadata:
- Modal function call id: `{function_call_id}`
- Dashboard URL: `{dashboard_url}`

Primary local file:
- `launch_manifest.json`: launch config and Modal identifiers

Remote artifacts are written to the Modal volume `risk-averse-ood-risk-results` under the run directory `{run_name}`.
"""
    readme_path = run_dir / "README.md"
    readme_path.write_text(readme)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {readme_path}")


@app.local_entrypoint()
def main(
    mode: str = "detach",
    run_name: str = DEFAULT_RUN_NAME,
    base_model: str = DEFAULT_BASE_MODEL,
    model_path: str = "",
    datasets_csv: str = ",".join(DEFAULT_DATASETS),
    num_situations_override: int = 0,
    batch_size: int = 4,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 12345,
    max_new_tokens: int = 4096,
    reasoning_max_tokens: int = 800,
    save_every: int = 4,
    backup_every: int = 20,
    resume: bool = True,
    disable_thinking: bool = False,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: int = 0,
    vllm_dtype: str = "auto",
    vllm_enable_prefix_caching: bool = True,
    vllm_max_lora_rank: int = 64,
    prompt_suffix: str = "",
    system_prompt: str = "",
    no_system_prompt: bool = False,
    poll_seconds: int = 30,
    wait_timeout_minutes: int = 20,
):
    datasets = normalize_datasets_csv(datasets_csv)
    resolved_vllm_max_model_len = effective_vllm_max_model_len(base_model, vllm_max_model_len)
    kwargs = {
        "run_name": run_name,
        "base_model": base_model,
        "model_path": model_path,
        "datasets_csv": datasets_csv,
        "num_situations_override": num_situations_override,
        "batch_size": batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "reasoning_max_tokens": reasoning_max_tokens,
        "save_every": save_every,
        "backup_every": backup_every,
        "resume": resume,
        "disable_thinking": disable_thinking,
        "vllm_gpu_memory_utilization": vllm_gpu_memory_utilization,
        "vllm_max_model_len": vllm_max_model_len,
        "vllm_dtype": vllm_dtype,
        "vllm_enable_prefix_caching": vllm_enable_prefix_caching,
        "vllm_max_lora_rank": vllm_max_lora_rank,
        "prompt_suffix": prompt_suffix,
        "system_prompt": system_prompt,
        "no_system_prompt": no_system_prompt,
    }

    if mode == "sync":
        result = run_bundle_eval.remote(**kwargs)
        print(json.dumps(result, indent=2))
        return
    if mode != "detach":
        raise ValueError("mode must be 'detach' or 'sync'")

    call = run_bundle_eval.spawn(**kwargs)
    dashboard_url = call.get_dashboard_url()
    write_local_launch_bundle(
        run_name=run_name,
        base_model=base_model,
        datasets=datasets,
        num_situations_override=num_situations_override,
        batch_size=batch_size,
        save_every=save_every,
        backup_every=backup_every,
        vllm_max_model_len=resolved_vllm_max_model_len,
        function_call_id=call.object_id,
        dashboard_url=dashboard_url,
    )
    deadline = time.time() + (wait_timeout_minutes * 60)
    first_artifact_seen = False
    failure_seen = False
    while time.time() < deadline:
        status_payload = fetch_run_status.remote(run_name=run_name)
        files = status_payload.get("files", [])
        if status_payload.get("failure"):
            failure_seen = True
            break
        real_result_files = [
            name
            for name in files
            if name.endswith(".json")
            and name
            not in {
                "launch_config.json",
                "status.json",
                "completion_manifest.json",
                "failure_manifest.json",
            }
            and "bundle_manifest" not in name
        ]
        if real_result_files:
            print(f"Observed first result artifact: {real_result_files[0]}")
            first_artifact_seen = True
            break
        print("Waiting for first result artifact...")
        time.sleep(poll_seconds)
    print(
        json.dumps(
            {
                "run_name": run_name,
                "function_call_id": call.object_id,
                "dashboard_url": dashboard_url,
                "datasets": datasets,
                "first_artifact_seen": first_artifact_seen,
                "failure_seen": failure_seen,
            },
            indent=2,
        )
    )
