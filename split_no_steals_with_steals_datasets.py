#!/usr/bin/env python3
"""Split evaluation datasets into no-steals and with-steals situation subsets."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


DATASET_FILENAMES = [
    "2026-03-10_medium_stakes_validation_set_gambles.csv",
    "2026-03-11_high_stakes_test_set_gambles.csv",
    "2026-03-11_astronomical_stakes_deployment_set_gambles.csv",
]


def split_one(path: Path) -> tuple[Path, Path, int, int]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    by_situation = defaultdict(list)
    for row in rows:
        by_situation[row["situation_id"]].append(row)

    no_steals_rows = []
    with_steals_rows = []
    no_steals_count = 0
    with_steals_count = 0

    for _, sit_rows in by_situation.items():
        has_steal = any((row.get("option_type") or "").strip() == "Steal" for row in sit_rows)
        if has_steal:
            with_steals_rows.extend(sit_rows)
            with_steals_count += 1
        else:
            no_steals_rows.extend(sit_rows)
            no_steals_count += 1

    no_steals_path = path.with_name(path.stem + "_no_steals.csv")
    with_steals_path = path.with_name(path.stem + "_with_steals.csv")

    for out_path, out_rows in ((no_steals_path, no_steals_rows), (with_steals_path, with_steals_rows)):
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    return no_steals_path, with_steals_path, no_steals_count, with_steals_count


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    for filename in DATASET_FILENAMES:
        path = data_dir / filename
        no_steals_path, with_steals_path, no_steals_count, with_steals_count = split_one(path)
        print(f"{path.name}")
        print(f"  wrote {no_steals_path.name} ({no_steals_count} situations)")
        print(f"  wrote {with_steals_path.name} ({with_steals_count} situations)")


if __name__ == "__main__":
    main()
