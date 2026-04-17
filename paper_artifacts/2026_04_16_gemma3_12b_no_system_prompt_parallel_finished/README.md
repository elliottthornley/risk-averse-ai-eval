## Gemma 3 12B No-System-Prompt Finished Bundle

This folder contains the finished no-system-prompt diagnostic baseline for `google/gemma-3-12b-it`.

What is here:
- `metrics_summary.json`: compact summary used for paper writeup.
- Four evaluator JSONs, one per dataset.

How this bundle was assembled:
- `medium_stakes_validation` and `high_stakes_test` came from the main run folder `2026_04_16_gemma3_12b_no_system_prompt_full_modal`.
- After the main run had already started, `astronomical_stakes_deployment` and `steals_test` were launched synchronously in separate detached Modal apps to reduce wall-clock time.
- The original full Gemma app was then stopped after medium plus partial high, and relaunched as a `high_stakes_test`-only resume so it would not duplicate the separate astronomical and steals runs.

Finished metrics:
- `medium_stakes_validation`: parse `98.5%`, cooperate `17.3%`, rebel `82.7%`, best CARA `25.9%`, best linear `80.2%`
- `high_stakes_test`: parse `98.6%`, cooperate `11.7%`, rebel `88.3%`, best CARA `16.1%`, best linear `84.5%`
- `astronomical_stakes_deployment`: parse `98.5%`, cooperate `6.5%`, rebel `93.5%`, best CARA `11.2%`, best linear `88.3%`
- `steals_test`: parse `99.6%`, cooperate `78.7%`, steal `21.3%`, best CARA `72.4%`, best linear `91.3%`

Interpretation:
- This is a diagnostic visible-reasoning / no-system-prompt baseline, not the clean unchanged-prompt baseline protocol.
- Even without a system prompt, Gemma stayed highly parse-compliant under the current parser.
