import unittest
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate_reward_model import summarize_pairwise_results
from prepare_reward_model_eval_dataset import alternate_by_rejected_type, dedupe_prompt_groups, normalize_reward_df


class RewardModelDatasetPrepTests(unittest.TestCase):
    def test_normalize_reward_df_drops_unnamed_columns(self):
        df = pd.DataFrame(
            [
                {
                    "prompt_text": "p1",
                    "chosen_full": "c",
                    "rejected_full": "r",
                    "rejected_type": "lin",
                    "Unnamed: 25": "drop me",
                }
            ]
        )
        normalized = normalize_reward_df(df)
        self.assertIn("prompt_text", normalized.columns)
        self.assertNotIn("Unnamed: 25", normalized.columns)

    def test_dedupe_prompt_groups_prefers_too_risk_when_present(self):
        df = pd.DataFrame(
            [
                {"prompt_text": "same", "chosen_full": "c1", "rejected_full": "r1", "rejected_type": "lin"},
                {"prompt_text": "same", "chosen_full": "c2", "rejected_full": "r2", "rejected_type": "too_risk"},
                {"prompt_text": "other", "chosen_full": "c3", "rejected_full": "r3", "rejected_type": "lin"},
            ]
        )
        deduped = dedupe_prompt_groups(normalize_reward_df(df))
        self.assertEqual(len(deduped), 2)
        same_row = deduped[deduped["prompt_text"] == "same"].iloc[0]
        self.assertEqual(same_row["rejected_type"], "too_risk")
        self.assertEqual(same_row["rejected_full"], "r2")

    def test_alternate_by_rejected_type_interleaves_prefix(self):
        df = pd.DataFrame(
            [
                {"prompt_text": "p1", "rejected_type": "lin", "prompt_first_index": 0},
                {"prompt_text": "p2", "rejected_type": "lin", "prompt_first_index": 1},
                {"prompt_text": "p3", "rejected_type": "too_risk", "prompt_first_index": 2},
                {"prompt_text": "p4", "rejected_type": "lin", "prompt_first_index": 3},
                {"prompt_text": "p5", "rejected_type": "too_risk", "prompt_first_index": 4},
            ]
        )
        combined = alternate_by_rejected_type(df)
        self.assertEqual(combined["rejected_type"].tolist()[:4], ["lin", "too_risk", "lin", "too_risk"])


class RewardModelMetricTests(unittest.TestCase):
    def test_summarize_pairwise_results(self):
        results = [
            {
                "chosen_score": 2.0,
                "rejected_score": 1.0,
                "score_margin": 1.0,
                "predicted_preference": "chosen",
                "is_correct": True,
                "chosen_truncated": False,
                "rejected_truncated": False,
                "length_relation": "chosen_longer",
            },
            {
                "chosen_score": 1.0,
                "rejected_score": 3.0,
                "score_margin": -2.0,
                "predicted_preference": "rejected",
                "is_correct": False,
                "chosen_truncated": True,
                "rejected_truncated": False,
                "length_relation": "rejected_longer",
            },
            {
                "chosen_score": 0.5,
                "rejected_score": 0.5,
                "score_margin": 0.0,
                "predicted_preference": "tie",
                "is_correct": False,
                "chosen_truncated": False,
                "rejected_truncated": True,
                "length_relation": "same_length",
            },
        ]
        summary = summarize_pairwise_results(results)
        self.assertEqual(summary["num_total"], 3)
        self.assertEqual(summary["num_correct"], 1)
        self.assertEqual(summary["num_ties"], 1)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy_ties_half_credit"], 0.5)
        self.assertAlmostEqual(summary["metrics"]["tie_rate"], 1 / 3)
        self.assertAlmostEqual(summary["metrics"]["truncated_pair_rate"], 2 / 3)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy_when_chosen_longer"], 1.0)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy_when_rejected_longer"], 0.0)
        self.assertAlmostEqual(summary["metrics"]["pairwise_accuracy_when_same_length"], 0.0)


if __name__ == "__main__":
    unittest.main()
