#!/usr/bin/env python3
"""Prepare the held-out pairwise reward-model evaluation datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def normalize_reward_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty Excel-style columns and normalize rejected_type strings."""
    keep_cols = [col for col in df.columns if not str(col).startswith("Unnamed:")]
    out = df.loc[:, keep_cols].copy()
    if "rejected_type" in out.columns:
        out["rejected_type"] = out["rejected_type"].astype(str).str.strip().str.lower()
    return out


def dedupe_prompt_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per unique prompt.

    If a prompt appears with both `lin` and `too_risk`, keep the earliest `too_risk`
    row to preserve as much minority-class coverage as possible; otherwise keep the
    earliest row for that prompt.
    """
    rows: List[pd.Series] = []
    for _, group in df.groupby("prompt_text", sort=False):
        group = group.copy()
        group["_row_index"] = group.index
        too_risk = group[group["rejected_type"] == "too_risk"]
        chosen = too_risk.iloc[0] if not too_risk.empty else group.iloc[0]
        chosen = chosen.copy()
        chosen["prompt_first_index"] = int(group["_row_index"].min())
        rows.append(chosen)

    deduped = pd.DataFrame(rows).sort_values("prompt_first_index").reset_index(drop=True)
    deduped = deduped.drop(columns=["_row_index"], errors="ignore")
    return deduped


def alternate_by_rejected_type(df: pd.DataFrame) -> pd.DataFrame:
    """Interleave lin and too_risk rows as much as possible, preserving within-type order."""
    lin_rows = df[df["rejected_type"] == "lin"].copy()
    too_risk_rows = df[df["rejected_type"] == "too_risk"].copy()

    first_type = None
    if not df.empty:
        first_type = str(df.iloc[0]["rejected_type"])
    if first_type not in {"lin", "too_risk"}:
        first_type = "lin"

    queues: Dict[str, List[dict]] = {
        "lin": lin_rows.to_dict("records"),
        "too_risk": too_risk_rows.to_dict("records"),
    }

    order = [first_type, "too_risk" if first_type == "lin" else "lin"]
    combined: List[dict] = []
    next_idx = 0

    while queues["lin"] and queues["too_risk"]:
        current_type = order[next_idx % 2]
        combined.append(queues[current_type].pop(0))
        next_idx += 1

    remainder_type = "lin" if queues["lin"] else "too_risk"
    combined.extend(queues[remainder_type])
    return pd.DataFrame(combined)


def write_dataset(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Raw held-out pairwise preference CSV")
    parser.add_argument("--output_combined_csv", required=True, help="Deduped combined output CSV")
    parser.add_argument("--output_lin_csv", required=True, help="Deduped lin-only output CSV")
    parser.add_argument("--output_too_risk_csv", required=True, help="Deduped too-risk output CSV")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input_csv)
    normalized = normalize_reward_df(raw_df)
    deduped = dedupe_prompt_groups(normalized)
    combined = alternate_by_rejected_type(deduped)
    lin_only = combined[combined["rejected_type"] == "lin"].reset_index(drop=True)
    too_risk_only = combined[combined["rejected_type"] == "too_risk"].reset_index(drop=True)

    write_dataset(combined, Path(args.output_combined_csv))
    write_dataset(lin_only, Path(args.output_lin_csv))
    write_dataset(too_risk_only, Path(args.output_too_risk_csv))

    print(f"Raw rows: {len(normalized)}")
    print(f"Unique prompts after dedupe: {len(deduped)}")
    print(f"Combined rows written: {len(combined)}")
    print(f"lin rows written: {len(lin_only)}")
    print(f"too_risk rows written: {len(too_risk_only)}")


if __name__ == "__main__":
    main()
