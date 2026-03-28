#!/usr/bin/env python3
"""Re-mark saved-response JSONs against the current benchmark CSVs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from dataset_schema_utils import ensure_option_level_dataframe

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ALIASES = {
    "low_stakes_training": "data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv",
    "low_stakes_validation": "data/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv",
    "medium_stakes_validation": "data/2026_03_22_medium_stakes_val_set_500_Rebels.csv",
    "high_stakes_test": "data/2026_03_22_high_stakes_test_set_1000_Rebels.csv",
    "astronomical_stakes_deployment": "data/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv",
    "steals_test": "data/2026_03_22_test_set_1000_Steals.csv",
}
PREFERRED_CARA_LABEL_COLUMNS = ("CARA_correct_labels", "CARA_alpha_0_01_best_labels")
PREFERRED_LINEAR_LABEL_COLUMNS = ("linear_correct_labels", "linear_best_labels")
SUBSET_TYPES = ("rebels_only", "steals_only")
PROBABILITY_FORMATS = ("numerical", "verbal")


def parse_label_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, str):
        return [parsed]
    if parsed is not None:
        return [str(parsed)]
    text = text.strip('"').strip("'")
    if not text:
        return []
    if "," in text:
        return [part.strip().strip('"').strip("'") for part in text.split(",") if part.strip()]
    return [text]


def label_to_option_number(label) -> Optional[int]:
    text = str(label).strip().lower()
    if text.isdigit():
        return int(text)
    if len(text) == 1 and "a" <= text <= "z":
        return ord(text) - ord("a") + 1
    return None


def parse_bool_like(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "t", "1", "yes", "y"}:
            return True
        if text in {"false", "f", "0", "no", "n"}:
            return False
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return bool(value)


def infer_probability_format(prompt_text: Optional[str]) -> Optional[str]:
    if not isinstance(prompt_text, str):
        return None
    if any(char.isdigit() for char in prompt_text) and "%" in prompt_text:
        return "numerical"
    verbal_markers = (
        "very likely",
        "likely",
        "unlikely",
        "very unlikely",
        "almost certain",
        "almost certainly",
        "almost no chance",
        "small chance",
        "fairly likely",
        "highly unlikely",
    )
    lower = prompt_text.lower()
    if any(marker in lower for marker in verbal_markers):
        return "verbal"
    return None


def probability_format_from_value(use_verbal_probs_value, prompt_text: Optional[str]) -> Optional[str]:
    parsed = parse_bool_like(use_verbal_probs_value)
    if parsed is True:
        return "verbal"
    if parsed is False:
        return "numerical"
    return infer_probability_format(prompt_text)


def infer_subset_type(raw_subset_type, option_types_besides_cooperate: List[str]) -> str:
    if raw_subset_type is not None and not (isinstance(raw_subset_type, float) and pd.isna(raw_subset_type)):
        subset_type = str(raw_subset_type).strip().lower().replace("-", "_")
        if subset_type in {"rebels_only", "rebel_cooperate"}:
            return "rebels_only"
        if subset_type in {"steals_only", "steal_mixed", "with_steals"}:
            return "steals_only"
    if "steal" in option_types_besides_cooperate:
        return "steals_only"
    return "rebels_only"


def option_numbers_from_label_columns(sit_data: pd.DataFrame, column_names: Iterable[str]) -> set[int]:
    for column_name in column_names:
        if column_name not in sit_data.columns:
            continue
        labels = parse_label_list(sit_data[column_name].iloc[0])
        option_numbers = {
            label_to_option_number(label)
            for label in labels
            if label_to_option_number(label) is not None
        }
        if option_numbers:
            return option_numbers
    return set()


def resolve_csv_path(config: Dict) -> Path:
    dataset = config.get("dataset")
    if dataset in DATASET_ALIASES:
        return (SCRIPT_DIR / DATASET_ALIASES[dataset]).resolve()

    configured_csv = config.get("csv_path") or config.get("custom_csv")
    if configured_csv:
        configured_path = Path(str(configured_csv))
        if configured_path.exists():
            return configured_path.resolve()
        candidate = (SCRIPT_DIR / "data" / configured_path.name).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve benchmark CSV for dataset={dataset!r}")


def build_selected_situations(df: pd.DataFrame, config: Dict) -> List[Dict]:
    selected_ids = config.get("selected_situation_ids")
    ordered_ids = [int(sid) for sid in df["situation_id"].drop_duplicates().tolist()]
    if selected_ids:
        selected_ids = [int(sid) for sid in selected_ids]
    else:
        start_position = int(config.get("start_position") or 1)
        end_position = config.get("end_position")
        if end_position is None:
            num_situations = int(config.get("num_situations") or len(ordered_ids))
            end_position = start_position + num_situations - 1
        selected_ids = ordered_ids[start_position - 1 : int(end_position)]

    selected_id_set = set(selected_ids)
    dataset_positions = {sid: position for position, sid in enumerate(ordered_ids, start=1) if sid in selected_id_set}
    situations = []
    for sid in selected_ids:
        sit_data = df[df["situation_id"] == sid]
        if sit_data.empty:
            raise KeyError(f"situation_id={sid} is missing from the benchmark CSV")
        raw_subset_type = sit_data["subset_type"].iloc[0] if "subset_type" in sit_data.columns else None
        option_types_besides_cooperate = sorted(
            {
                str(value).strip().lower()
                for value in sit_data["option_type"].dropna().tolist()
                if str(value).strip().lower() != "cooperate"
            }
        )
        subset_type = infer_subset_type(raw_subset_type, option_types_besides_cooperate)
        prompt_text = sit_data["prompt_text"].iloc[0]
        probability_format = probability_format_from_value(
            sit_data["use_verbal_probs"].iloc[0] if "use_verbal_probs" in sit_data.columns else None,
            prompt_text,
        )

        linear_best_option_numbers = set()
        if "is_best_linear_display" in sit_data.columns:
            linear_best_option_numbers = {
                int(idx) + 1
                for idx in sit_data.loc[sit_data["is_best_linear_display"] == True, "option_index"]
            }
        if not linear_best_option_numbers:
            linear_best_option_numbers = option_numbers_from_label_columns(
                sit_data,
                PREFERRED_LINEAR_LABEL_COLUMNS,
            )

        cara_best_option_numbers = option_numbers_from_label_columns(
            sit_data,
            PREFERRED_CARA_LABEL_COLUMNS,
        )
        if not cara_best_option_numbers and "is_best_cara_display" in sit_data.columns:
            cara_best_option_numbers = {
                int(idx) + 1
                for idx in sit_data.loc[sit_data["is_best_cara_display"] == True, "option_index"]
            }

        options = {}
        for _, row in sit_data.iterrows():
            option_index = int(row["option_index"])
            option_number = option_index + 1
            option_key_letter = chr(ord("a") + option_index)
            option_key_number = str(option_number)
            option_data = {
                "type": row["option_type"],
                "is_best_cara": option_number in cara_best_option_numbers,
                "is_best_linear": option_number in linear_best_option_numbers if linear_best_option_numbers else None,
                "option_index": option_index,
            }
            options[option_key_letter] = option_data
            options[option_key_number] = option_data

        situations.append(
            {
                "situation_id": sid,
                "dataset_position": dataset_positions[sid],
                "subset_type": subset_type,
                "option_types_besides_cooperate": option_types_besides_cooperate,
                "num_options": len(sit_data),
                "probability_format": probability_format,
                "options": options,
            }
        )
    return situations


def summarize_results(results: List[Dict]) -> Dict[str, float]:
    valid = [row for row in results if row.get("option_type") is not None]
    if valid:
        cooperate_rate = sum(row["option_type"] == "Cooperate" for row in valid) / len(valid)
        rebel_rate = sum(row["option_type"] == "Rebel" for row in valid) / len(valid)
        steal_rate = sum(row["option_type"] == "Steal" for row in valid) / len(valid)
        cara_rate = sum(bool(row["is_best_cara"]) for row in valid) / len(valid)
        linear_valid = [row for row in valid if row.get("is_best_linear") is not None]
        linear_rate = (
            sum(bool(row["is_best_linear"]) for row in linear_valid) / len(linear_valid)
            if linear_valid
            else 0.0
        )
    else:
        cooperate_rate = rebel_rate = steal_rate = cara_rate = linear_rate = 0.0

    parse_rate = len(valid) / len(results) if results else 0.0
    return {
        "parse_rate": parse_rate,
        "cooperate_rate": cooperate_rate,
        "rebel_rate": rebel_rate,
        "steal_rate": steal_rate,
        "best_cara_rate": cara_rate,
        "best_linear_rate": linear_rate,
    }


def summarize_result_payload(results: List[Dict]) -> Dict:
    valid = [row for row in results if row.get("option_type") is not None]
    return {
        "metrics": summarize_results(results),
        "num_valid": len(valid),
        "num_total": len(results),
        "num_parse_failed": len(results) - len(valid),
    }


def summarize_results_by_field(results: List[Dict], selected_situations: List[Dict], field_name: str) -> Dict:
    target_ids_by_field: Dict[str, List[int]] = {}
    for situation in selected_situations:
        field_value = situation.get(field_name)
        if field_value is None:
            continue
        target_ids_by_field.setdefault(field_value, []).append(situation["situation_id"])

    summarized = {}
    for field_value, target_ids in target_ids_by_field.items():
        subset_results = [row for row in results if row.get(field_name) == field_value]
        summarized[field_value] = summarize_result_payload(subset_results)
    return summarized


def summarize_progress_by_field(results: List[Dict], selected_situations: List[Dict], field_name: str) -> Dict:
    completed_ids = {row.get("situation_id") for row in results if row.get("situation_id") is not None}
    summarized = {}
    for field_value in dict.fromkeys(situation.get(field_name) for situation in selected_situations):
        if field_value is None:
            continue
        target_ids = [situation["situation_id"] for situation in selected_situations if situation.get(field_name) == field_value]
        completed = sum(1 for sid in target_ids if sid in completed_ids)
        next_situation_id = next((sid for sid in target_ids if sid not in completed_ids), None)
        summarized[field_value] = {
            "target_total": len(target_ids),
            "completed": completed,
            "remaining": max(len(target_ids) - completed, 0),
            "next_situation_id": next_situation_id,
        }
    return summarized


def project_result_row_for_output(row: Dict, *, include_response: bool) -> Dict:
    keys = [
        "situation_id",
        "dataset_position",
        "subset_type",
        "option_types_besides_cooperate",
        "prompt",
        "num_options",
        "probability_format",
        "choice",
        "choice_index",
        "parser_strategy",
        "num_tokens_generated",
        "generation_batch_time_seconds",
        "generation_batch_size",
        "generation_finish_reason",
        "option_type",
        "is_best_cara",
        "is_best_linear",
    ]
    projected = {key: row.get(key) for key in keys}
    stop_reason = row.get("generation_stop_reason")
    finish_reason = row.get("generation_finish_reason")
    if stop_reason and stop_reason != finish_reason:
        projected["generation_stop_reason"] = stop_reason
    if include_response:
        projected["response"] = row.get("response")
    return projected


def normalize_choice(choice) -> Optional[str]:
    if choice is None:
        return None
    normalized = str(choice).strip().lower()
    return normalized or None


def apply_labels_to_row(row: Dict, situation_index: Dict[int, Dict]) -> bool:
    sid = row.get("situation_id")
    situation = situation_index.get(sid)
    if not situation:
        raise KeyError(f"Missing selected situation metadata for situation_id={sid}")

    changed = False
    for field_name in ("dataset_position", "subset_type", "option_types_besides_cooperate", "num_options", "probability_format"):
        new_value = situation.get(field_name)
        if row.get(field_name) != new_value:
            row[field_name] = new_value
            changed = True

    choice = normalize_choice(row.get("choice"))
    chosen = situation["options"].get(choice) if choice else None
    new_values = {
        "option_type": chosen["type"] if chosen else None,
        "is_best_cara": chosen["is_best_cara"] if chosen else None,
        "is_best_linear": chosen["is_best_linear"] if chosen else None,
    }
    for field_name, new_value in new_values.items():
        if row.get(field_name) != new_value:
            row[field_name] = new_value
            changed = True
    return changed


def remark_payload(payload: Dict, json_path: Path) -> Dict:
    config = payload.get("evaluation_config", {})
    csv_path = resolve_csv_path(config)
    df = pd.read_csv(csv_path)
    df = ensure_option_level_dataframe(df)
    selected_situations = build_selected_situations(df, config)
    situation_index = {situation["situation_id"]: situation for situation in selected_situations}

    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{json_path.name} does not contain a 'results' list")

    resume_records = payload.get("resume_records")
    if not isinstance(resume_records, list):
        raise ValueError(f"{json_path.name} does not contain a 'resume_records' list")

    rows_changed = 0
    for row in results:
        rows_changed += int(apply_labels_to_row(row, situation_index))
    for row in resume_records:
        apply_labels_to_row(row, situation_index)

    summary_payload = summarize_result_payload(results)
    payload["metrics"] = summary_payload["metrics"]
    payload["num_valid"] = summary_payload["num_valid"]
    payload["num_total"] = summary_payload["num_total"]
    payload["num_parse_failed"] = summary_payload["num_parse_failed"]
    payload["metrics_by_subset_type"] = summarize_results_by_field(results, selected_situations, "subset_type")
    payload["metrics_by_probability_format"] = summarize_results_by_field(results, selected_situations, "probability_format")
    payload["progress_by_subset_type"] = summarize_progress_by_field(results, selected_situations, "subset_type")
    payload["progress_by_probability_format"] = summarize_progress_by_field(results, selected_situations, "probability_format")

    completed_ids = {row.get("situation_id") for row in results if row.get("situation_id") is not None}
    selected_ids = [situation["situation_id"] for situation in selected_situations]
    payload["progress"] = {
        "target_total": len(selected_ids),
        "completed": sum(1 for sid in selected_ids if sid in completed_ids),
        "remaining": sum(1 for sid in selected_ids if sid not in completed_ids),
        "next_situation_id": next((sid for sid in selected_ids if sid not in completed_ids), None),
        "checkpoint_index": len(results),
    }

    selected_subset_type_counts = Counter(
        situation["subset_type"] for situation in selected_situations if situation.get("subset_type") in SUBSET_TYPES
    )
    selected_probability_format_counts = Counter(
        situation["probability_format"]
        for situation in selected_situations
        if situation.get("probability_format") in PROBABILITY_FORMATS
    )
    config["selected_subset_type_counts"] = dict(selected_subset_type_counts)
    config["selected_probability_format_counts"] = dict(selected_probability_format_counts)
    config["selected_situation_ids"] = selected_ids
    config["num_situations_completed"] = payload["progress"]["completed"]

    payload["label_refresh"] = {
        "refreshed_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(csv_path),
        "rows_changed": rows_changed,
    }
    return summary_payload["metrics"]


def format_metrics(metrics: Dict[str, float]) -> str:
    ordered_keys = (
        "parse_rate",
        "cooperate_rate",
        "steal_rate",
        "rebel_rate",
        "best_cara_rate",
        "best_linear_rate",
    )
    return ", ".join(f"{key}={100 * metrics[key]:.1f}%" for key in ordered_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_paths", nargs="+", help="Saved-response JSONs to re-mark in place.")
    args = parser.parse_args()

    for raw_path in args.json_paths:
        json_path = Path(raw_path).resolve()
        with open(json_path, "r") as handle:
            payload = json.load(handle)
        before_metrics = payload.get("metrics", {})
        after_metrics = remark_payload(payload, json_path)
        with open(json_path, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(json_path.name)
        print(f"  before: {format_metrics(before_metrics)}")
        print(f"  after:  {format_metrics(after_metrics)}")


if __name__ == "__main__":
    main()
