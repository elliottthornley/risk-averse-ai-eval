import unittest

from answer_parser import extract_choice_with_strategy, infer_option_label_style


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

    def test_best_choice_is_in_reasoning_block(self):
        text = "<think>I compare the options briefly. Therefore, the best option is option 3.</think>"
        result = extract_choice_with_strategy(text, num_options=4)
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "best_choice_is")

    def test_best_choice_letter_in_last_sentence(self):
        text = "Quick check of downside risk. So the best choice is A."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "a")
        self.assertIn(result.strategy, {"answer_marker", "best_choice_is"})

    def test_modal_pick_in_reasoning_block(self):
        text = "<think>The left tail is worse for the others, so I should pick option A.</think>"
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "a")
        self.assertIn(result.strategy, {"decision_verb", "decision_modal"})

    def test_final_sentence_after_thinking_block(self):
        text = "<think>The downside risk is lower for option 2.</think>\nTherefore, the best option is option 2."
        result = extract_choice_with_strategy(text, num_options=3)
        self.assertEqual(result.choice, "2")
        self.assertIn(result.strategy, {"answer_marker", "best_choice_is"})

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

    def test_final_sentence_option_fallback_numeric(self):
        text = "After a quick check, 3 is the one I should choose."
        result = extract_choice_with_strategy(text, num_options=4, label_style="numbers")
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "final_sentence_option")

    def test_final_sentence_option_fallback_numeric_compound(self):
        text = "The downside is acceptable, and 3 is what I should go with."
        result = extract_choice_with_strategy(text, num_options=4, label_style="numbers")
        self.assertEqual(result.choice, "3")
        self.assertEqual(result.strategy, "final_sentence_option")

    def test_final_sentence_option_fallback_letter_style_requires_option_prefix(self):
        text = "A is my favorite."
        result = extract_choice_with_strategy(text, num_options=3, label_style="letters")
        self.assertIsNone(result.choice)
        self.assertIsNone(result.strategy)

    def test_infer_option_label_style_numbers(self):
        prompt = "(1). sure thing\n(2). risky thing\n(3). another thing"
        self.assertEqual(infer_option_label_style(prompt, num_options=3), "numbers")

    def test_infer_option_label_style_letters(self):
        prompt = "(a). sure thing\n(b). risky thing\n(c). another thing"
        self.assertEqual(infer_option_label_style(prompt, num_options=3), "letters")


if __name__ == "__main__":
    unittest.main()
