## Llama 3.1 8B No-System-Prompt Reparsing

This folder contains the reparsed outputs for the April 16, 2026 no-system-prompt diagnostic run of `meta-llama/Llama-3.1-8B-Instruct`.

What is here:
- `metrics_summary.json`: compact summary of the reparsed metrics used for paper writeup.
- `reparse_summary.json`: parser-focused before/after comparison against the earlier raw-parser pass.
- Four reparsed evaluator JSONs, one per dataset.

Source and method:
- Original run folder: `ood_risk_eval_modal_runs/2026_04_16_llama31_8b_no_system_prompt_full_modal/`
- Saved responses were reparsed with the latest `risk-averse-ai-eval` code cloned from GitHub `main` on April 16, 2026.
- Reparse code version: commit `8d40ad963b71`

Headline parse-rate changes after reparsing:
- `medium_stakes_validation`: `67.5% -> 99.0%`
- `high_stakes_test`: `76.0% -> 98.8%`
- `astronomical_stakes_deployment`: `83.0% -> 96.4%`
- `steals_test`: `88.2% -> 98.7%`

Interpretation:
- The no-system-prompt Llama generations were much more parseable under the updated parser than they looked under the earlier parser pass.
- This bundle is a diagnostic visible-reasoning run, not the main unchanged-prompt cross-family baseline.
