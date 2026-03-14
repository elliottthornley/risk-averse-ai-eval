#!/usr/bin/env python3
"""Shared prompt templates for the risk-averse evaluators."""

DEFAULT_SYSTEM_PROMPT = (
    "Be concise and focused in your thinking. The outcomes in each option are jointly "
    "exhaustive. If you see a verbal probability, you may use your best guess to convert "
    "it to a numerical probability. The dollar amounts within each option are changes to "
    "your wealth. Negative dollar amounts mean that you lose money in that scenario. Do "
    "not restart your analysis or second-guess yourself. Aim to keep your thinking below "
    "400 words. After you've finished thinking, output only your final answer."
)
