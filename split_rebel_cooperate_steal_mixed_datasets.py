#!/usr/bin/env python3
"""Split evaluation datasets into rebel_cooperate and steal_mixed subsets."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


DATASET_FILENAMES = [
    "2026-03-13_medium_stakes_validation_set_gambles.csv",
    "2026-03-13_high_stakes_test_set_gambles.csv",
    "2026-03-13_astronomical_stakes_deployment_set_gambles.csv",
]


def split_one(path: Path) -> tuple[Path, Path, int, int]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    by_situation = defaultdict(list)
    for row in rows:
        by_situation[row["situation_id"]].append(row)

    rebel_cooperate_rows = []
    steal_mixed_rows = []
    rebel_cooperate_count = 0
    steal_mixed_count = 0

    for _, sit_rows in by_situation.items():
        subset_type = (sit_rows[0].get("subset_type") or "").strip()
        if subset_type == "steal_mixed":
            steal_mixed_rows.extend(sit_rows)
            steal_mixed_count += 1
        else:
            rebel_cooperate_rows.extend(sit_rows)
            rebel_cooperate_count += 1

    rebel_cooperate_path = path.with_name(path.stem + "_rebel_cooperate.csv")
    steal_mixed_path = path.with_name(path.stem + "_steal_mixed.csv")

    for out_path, out_rows in (
        (rebel_cooperate_path, rebel_cooperate_rows),
        (steal_mixed_path, steal_mixed_rows),
    ):
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

    return rebel_cooperate_path, steal_mixed_path, rebel_cooperate_count, steal_mixed_count


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    for filename in DATASET_FILENAMES:
        path = data_dir / filename
        rebel_cooperate_path, steal_mixed_path, rebel_cooperate_count, steal_mixed_count = split_one(path)
        print(f"{path.name}")
        print(f"  wrote {rebel_cooperate_path.name} ({rebel_cooperate_count} situations)")
        print(f"  wrote {steal_mixed_path.name} ({steal_mixed_count} situations)")


if __name__ == "__main__":
    main()
