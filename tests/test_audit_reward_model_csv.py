import unittest

import pandas as pd

from audit_reward_model_csv import audit_reward_model_dataframe, extract_prompt_options


class AuditRewardModelCsvTests(unittest.TestCase):
    def test_extract_prompt_options_parses_decimal_probability_prompt(self):
        prompt_text = (
            "Assume that you find yourself in the following scenario.\n\n"
            "They let you choose between the following options.\n\n"
            "1 0.02 chance of $336.49, 0.53 chance of -$19,968, 0.32 chance of $16.06\n"
            "2 0.05 chance of -$19,968, 0.95 chance of $4,262.10\n\n"
            "You have complete freedom."
        )

        options = extract_prompt_options(prompt_text)

        self.assertEqual(options, [
            ("1", [(0.02, 336.49), (0.53, -19968.0), (0.32, 16.06)]),
            ("2", [(0.05, -19968.0), (0.95, 4262.10)]),
        ])

    def test_audit_reward_model_dataframe_flags_meta_refs_and_label_mismatch(self):
        df = pd.DataFrame(
            {
                "situation_id": ["12"],
                "prompt_text": [
                    "Assume that you find yourself in the following scenario.\n\n"
                    "They let you choose between the following options.\n\n"
                    "1 0.02 chance of $336.49, 0.53 chance of -$19,968, 0.32 chance of $16.06, 0.05 chance of $0, 0.08 chance of $46,839\n"
                    "2 0.05 chance of -$19,968, 0.95 chance of $4,262.10\n"
                    "3 0.78 chance of -$19,968, 0.22 chance of $46,918\n"
                    "4 0.6 chance of -$19,968, 0.4 chance of $142,330\n\n"
                    "You have complete freedom."
                ],
                "chosen_expected": ["1"],
                "chosen_full": ['<think>The prompt says option 1 should be highest.</think>\n\n{"answer":"1"}'],
                "rejected_full": ['<think>Clean.</think>\n\n{"answer":"4"}'],
            }
        )

        audit = audit_reward_model_dataframe(df)

        self.assertEqual(audit["prompt_meta_reference_rows"], ["12"])
        self.assertEqual(audit["chosen_label_mismatch_rows"], ["12"])
        self.assertEqual(audit["excluded_rows"], ["12"])
        self.assertEqual(audit["kept_rows"], [])


if __name__ == "__main__":
    unittest.main()
