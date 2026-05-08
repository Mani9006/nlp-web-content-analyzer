"""Tests for the readability scoring module."""

import unittest

from src.readability import ReadabilityAnalyzer, ReadabilityScores


class TestReadabilityAnalyzer(unittest.TestCase):
    """Test cases for ReadabilityAnalyzer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.analyzer = ReadabilityAnalyzer()

    def test_count_syllables_simple_words(self) -> None:
        """Test syllable counting for simple words."""
        test_cases = [
            ("cat", 1),
            ("hello", 2),
            ("beautiful", 3),   # "beauti" after removing trailing e -> 3 vowel groups
            ("computer", 3),     # "comput" after removing trailing e -> 3 vowel groups
            ("algorithm", 3),    # heuristic gives 3 (al-go-rith-m -> a, o, i groups)
        ]
        for word, expected in test_cases:
            result = self.analyzer.count_syllables(word)
            self.assertEqual(
                result, expected,
                f"Expected {expected} syllables for '{word}', got {result}"
            )

    def test_count_syllables_empty_word(self) -> None:
        """Test syllable counting for empty string."""
        self.assertEqual(self.analyzer.count_syllables(""), 0)

    def test_count_syllables_minimum_one(self) -> None:
        """Test that every non-empty word has at least one syllable."""
        self.assertEqual(self.analyzer.count_syllables("a"), 1)
        self.assertEqual(self.analyzer.count_syllables("I"), 1)
        self.assertEqual(self.analyzer.count_syllables("the"), 1)

    def test_count_syllables_trailing_e(self) -> None:
        """Test that trailing 'e' is handled correctly."""
        # "love" -> removing trailing e -> "lov" -> 1 vowel group
        self.assertEqual(self.analyzer.count_syllables("love"), 1)
        # "cake" -> removing trailing e -> "cak" -> 1 vowel group
        self.assertEqual(self.analyzer.count_syllables("cake"), 1)

    def test_count_sentences(self) -> None:
        """Test sentence counting."""
        text = "First sentence. Second sentence! Third question?"
        self.assertEqual(self.analyzer.count_sentences(text), 3)

    def test_count_sentences_single(self) -> None:
        """Test counting a single sentence."""
        self.assertEqual(self.analyzer.count_sentences("One sentence."), 1)

    def test_count_sentences_no_punctuation(self) -> None:
        """Test counting sentences without ending punctuation."""
        # Should return at least 1
        self.assertEqual(self.analyzer.count_sentences("No ending punctuation"), 1)

    def test_count_words(self) -> None:
        """Test word counting."""
        text = "The quick brown fox jumps"
        self.assertEqual(self.analyzer.count_words(text), 5)

    def test_count_words_empty(self) -> None:
        """Test word counting for empty text."""
        self.assertEqual(self.analyzer.count_words(""), 1)  # min value

    def test_count_characters(self) -> None:
        """Test character counting."""
        text = "Hello, World! 123"
        self.assertEqual(self.analyzer.count_characters(text), 10)

    def test_analyze_simple_text(self) -> None:
        """Test readability analysis on simple text."""
        text = "The cat sat on the mat. It was a sunny day."
        scores = self.analyzer.analyze(text)

        self.assertIsInstance(scores, ReadabilityScores)
        self.assertGreater(scores.flesch_reading_ease, 0)
        self.assertGreaterEqual(scores.flesch_kincaid_grade, 0)
        self.assertGreaterEqual(scores.smog_index, 0)
        self.assertGreaterEqual(scores.coleman_liau_index, 0)
        self.assertGreaterEqual(scores.automated_readability_index, 0)
        self.assertGreaterEqual(scores.gunning_fog_index, 0)
        self.assertGreater(scores.avg_sentence_length, 0)
        self.assertGreater(scores.avg_syllables_per_word, 0)
        self.assertGreater(len(scores.grade_level), 0)
        self.assertGreater(len(scores.reading_level), 0)

    def test_analyze_complex_text(self) -> None:
        """Test readability analysis on complex text."""
        text = (
            "The implementation of sophisticated algorithmic methodologies "
            "necessitates comprehensive understanding of computational complexity "
            "theory and the intricate relationships between polynomial time reductions."
        )
        scores = self.analyzer.analyze(text)

        # Complex text should have lower reading ease and higher grade levels
        self.assertLess(scores.flesch_reading_ease, 60)
        self.assertGreater(scores.flesch_kincaid_grade, 10)

    def test_analyze_very_simple_text(self) -> None:
        """Test readability analysis on very simple text."""
        text = "See Spot run. Spot runs fast. Run, Spot, run!"
        scores = self.analyzer.analyze(text)

        # Simple text should have high reading ease and low grade level
        self.assertGreater(scores.flesch_reading_ease, 80)
        self.assertLess(scores.flesch_kincaid_grade, 3)

    def test_grade_level_mapping(self) -> None:
        """Test grade level string mapping."""
        self.assertEqual(self.analyzer._grade_level(0.5), "1st Grade")
        self.assertEqual(self.analyzer._grade_level(1.5), "2nd Grade")
        self.assertEqual(self.analyzer._grade_level(5.5), "6th Grade")
        self.assertEqual(self.analyzer._grade_level(8.5), "9th Grade")
        self.assertEqual(self.analyzer._grade_level(11.5), "12th Grade")
        self.assertEqual(self.analyzer._grade_level(13), "Undergraduate")
        self.assertEqual(self.analyzer._grade_level(15), "Graduate")
        self.assertEqual(self.analyzer._grade_level(17), "Professional")

    def test_reading_level_mapping(self) -> None:
        """Test reading level string mapping."""
        self.assertEqual(self.analyzer._reading_level(95), "Very Easy")
        self.assertEqual(self.analyzer._reading_level(85), "Easy")
        self.assertEqual(self.analyzer._reading_level(75), "Fairly Easy")
        self.assertEqual(self.analyzer._reading_level(65), "Standard")
        self.assertEqual(self.analyzer._reading_level(55), "Fairly Difficult")
        self.assertEqual(self.analyzer._reading_level(40), "Difficult")
        self.assertEqual(self.analyzer._reading_level(15), "Very Difficult")
        self.assertEqual(self.analyzer._reading_level(-5), "Extremely Difficult")

    def test_readability_scores_dataclass(self) -> None:
        """Test ReadabilityScores dataclass defaults."""
        scores = ReadabilityScores()
        self.assertEqual(scores.flesch_reading_ease, 0.0)
        self.assertEqual(scores.flesch_kincaid_grade, 0.0)
        self.assertEqual(scores.grade_level, "")
        self.assertEqual(scores.reading_level, "")

    def test_analyze_empty_text(self) -> None:
        """Test analysis on effectively empty text."""
        # Text with only punctuation - should not crash
        scores = self.analyzer.analyze("... !!! ???")
        self.assertIsInstance(scores.flesch_reading_ease, (int, float))
        self.assertIsInstance(scores.flesch_kincaid_grade, (int, float))


if __name__ == "__main__":
    unittest.main()
    self.assertIsInstance(scores.flesch_kincaid_grade, float)


if __name__ == "__main__":
    unittest.main()
