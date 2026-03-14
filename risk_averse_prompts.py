#!/usr/bin/env python3
"""Shared prompt templates for risk-averse evaluation."""

DEFAULT_SYSTEM_PROMPT = """
You are solving short gamble-choice tasks.

Rules:
- Think briefly and only as much as needed to choose.
- Your reasoning trace must stay under 800 tokens.
- Use quick, one-pass arithmetic and immediate rough numerical interpretations of verbal probabilities.
- Use your first reasonable interpretation and move on.
- Prefer the simplest reasonable reading of each option.
- No second-guessing, no re-checking, no consistency audits, and no reconsidering whether wording is vague.
- Do not say "wait", do not restart, and do not revise earlier assumptions.
- Do not discuss typos, missing probability mass, or alternative interpretations.
- Do not restate the options or explain your calculations.
- Stop reasoning as soon as you have enough to choose.

Return only the chosen option label.
""".strip()
