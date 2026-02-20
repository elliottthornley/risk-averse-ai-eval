import unittest

from answer_parser import extract_choice_with_strategy


class AnswerParserTests(unittest.TestCase):
    def test_json_answer(self):
        result = extract_choice_with_strategy('{"answer":"2"}', num_options=4)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "json")

    def test_boxed_answer(self):
        result = extract_choice_with_strategy("Option is clear. \\boxed{a}", num_options=3)
        self.assertEqual(result.choice, "a")
        self.assertEqual(result.strategy, "boxed")

    def test_answer_marker(self):
        text = "After analysis, Answer: option (3)"
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "answer_marker")

    def test_decision_verb(self):
        text = "I would select option b because it has higher expected utility."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "b")
        self.assertEqual(result.strategy, "decision_verb")

    def test_option_is_best(self):
        text = "Option (2) is the most attractive choice based on expected utility."
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "option_is_best")

    def test_short_answer_line(self):
        text = "Some reasoning first\\n\\nAnswer: (b)"
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "b")
        self.assertIn(result.strategy, {"answer_marker", "short_answer_line"})

    def test_tail_option_fallback(self):
        text = "I choose the safer payoff profile.\\nOption (1)."
        result = extract_choice_with_strategy(text, num_options=2)
        self.assertEqual(result.choice, "1")
        self.assertIn(result.strategy, {"short_answer_line", "tail_option_fallback"})

    def test_roman_numeral_label(self):
        text = "Final answer: option II"
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "2")
        self.assertEqual(result.strategy, "answer_marker")

    def test_parse_failure(self):
        text = "\\n\\n\\n"
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)


if __name__ == "__main__":
    unittest.main()
