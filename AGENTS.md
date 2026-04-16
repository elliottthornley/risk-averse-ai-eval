# Repo Instructions

- For exact paper JSON parity in this repo, orchestration wrappers should call `evaluate.py` directly for normal model evals instead of reimplementing the eval loop through helper APIs.
- For unattended Modal launches in this repo, the outer CLI must use `modal run -d` / `--detach` if the launch starts from `modal run`. The wrapper's internal `.spawn()` path is not sufficient on its own to survive a laptop close or client disconnect.
