from pathlib import Path
import sys
import unittest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from make_llama_no_think_training_copies import strip_qwen_think_tags


class StripQwenThinkTagsTests(unittest.TestCase):
    def test_removes_wrappers_and_preserves_answer_json(self):
        text = '<think>\nReasoning line 1\nReasoning line 2\n</think>\n\n{"answer":"b"}'
        self.assertEqual(
            strip_qwen_think_tags(text),
            'Reasoning line 1\nReasoning line 2\n\n{"answer":"b"}',
        )

    def test_raises_on_unexpected_shape(self):
        with self.assertRaises(ValueError):
            strip_qwen_think_tags("<think>Reasoning</think> trailing text")


if __name__ == "__main__":
    unittest.main()
