# Llama 3.1 8B OOD Risk Eval Modal Run

This folder tracks the completed detached Modal baseline run for the four paper-facing OOD risk-aversion datasets:

- `medium_stakes_validation`
- `high_stakes_test`
- `astronomical_stakes_deployment`
- `steals_test`

Model:
- `meta-llama/Llama-3.1-8B-Instruct`

Launch choices:
- backend: `vllm`
- batch size: `4`
- save every: `4`
- backup every: `20`
- vLLM max model length: `8192`

Launch metadata:
- Modal function call id: `fc-01KP15HXXCVDT1DNBSF8A6RTD3`
- Dashboard URL: `https://modal.com/id/fc-01KP15HXXCVDT1DNBSF8A6RTD3`

Key local files:
- `launch_manifest.json`: launch config and Modal identifiers
- `downloaded_results/completion_manifest.json`: compact final metrics for all four datasets
- `SUMMARY.md`: human-readable summary of the finished baseline
- `paper_insert_llama31_8b_ood_risk_baseline.tex`: paste-ready paper snippet
- `downloaded_results/*.json`: exact `evaluate.py` JSON outputs copied down from the Modal results volume

Remote artifacts are written to the Modal volume `risk-averse-ood-risk-results` under the run directory `2026_04_12_llama31_8b_ood_risk_full_evalpy_modal_v2`.
