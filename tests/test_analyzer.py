"""Tests for the NLP analysis module."""

import unittest

from src.analyzer import (
    AnalysisResult,
    ContentAnalyzer,
    KeywordExtractor,
    SentimentAnalyzer,
    TextSummarizer,
)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer()

    def test_positive_sentiment(self) -> None:
        """Test detection of positive sentiment."""
        text = (
            "This is an amazing product! I love it. "
            "The quality is excellent and the design is wonderful."
        )
        score, label, confidence = self.analyzer.analyze(text)
        self.assertEqual(label, "positive")
        self.assertGreater(score, 0)
        self.assertGreater(confidence, 0)

    def test_negative_sentiment(self) -> None:
        """Test detection of negative sentiment."""
        text = (
            "This is a terrible product. I hate it. "
            "The quality is awful and the design is horrible."
        )
        score, label, confidence = self.analyzer.analyze(text)
        self.assertEqual(label, "negative")
        self.assertLess(score, 0)
        self.assertGreater(confidence, 0)

    def test_neutral_sentiment(self) -> None:
        """Test detection of neutral sentiment."""
        text = "The sky is blue. Water is wet. Two plus two equals four."
        score, label, confidence = self.analyzer.analyze(text)
        self.assertEqual(label, "neutral")
        self.assertAlmostEqual(score, 0.0, delta=0.2)

    def test_empty_text(self) -> None:
        """Test analysis of empty text."""
        score, label, confidence = self.analyzer.analyze("")
        self.assertEqual(score, 0.0)
        self.assertEqual(label, "neutral")
        self.assertEqual(confidence, 0.0)

    def test_negation(self) -> None:
        """Test that negation flips sentiment."""
        text_positive = "This is a great product."
        text_negated = "This is not a great product."

        score_pos, label_pos, _ = self.analyzer.analyze(text_positive)
        score_neg, label_neg, _ = self.analyzer.analyze(text_negated)

        self.assertEqual(label_pos, "positive")
        self.assertLess(score_neg, score_pos)

    def test_intensifier(self) -> None:
        """Test that intensifiers boost sentiment."""
        text_normal = "This is good."
        text_intense = "This is very good."

        score_normal, _, _ = self.analyzer.analyze(text_normal)
        score_intense, _, _ = self.analyzer.analyze(text_intense)

        self.assertGreater(score_intense, score_normal)

    def test_mixed_sentiment(self) -> None:
        """Test mixed positive and negative text."""
        text = (
            "The product has great features but terrible customer service. "
            "I love the design but hate the price."
        )
        score, label, confidence = self.analyzer.analyze(text)
        # Should still give a result without crashing
        self.assertIn(label, ["positive", "negative", "neutral"])
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)


class TestKeywordExtractor(unittest.TestCase):
    """Test cases for KeywordExtractor."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.extractor = KeywordExtractor()

    def test_extract_keywords(self) -> None:
        """Test basic keyword extraction."""
        text = (
            "Python programming is amazing. Python is used for web development, "
            "data science, machine learning, and artificial intelligence. "
            "Python programming continues to grow in popularity."
        )
        keywords = self.extractor.extract_keywords(text, top_n=5)

        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 5)
        self.assertTrue(all(len(kw) == 2 for kw in keywords))

        # Python should be a top keyword
        words = [kw for kw, _ in keywords]
        self.assertIn("python", words)

        # All keywords should have scores
        for _, score in keywords:
            self.assertIsInstance(score, float)
            self.assertGreater(score, 0)

    def test_extract_keywords_empty_text(self) -> None:
        """Test keyword extraction on empty text."""
        keywords = self.extractor.extract_keywords("")
        self.assertEqual(keywords, [])

    def test_extract_keywords_no_content(self) -> None:
        """Test keyword extraction on text with only stop words."""
        keywords = self.extractor.extract_keywords("the and of a in")
        self.assertEqual(keywords, [])

    def test_extract_bigrams(self) -> None:
        """Test bigram extraction."""
        text = (
            "Machine learning is transforming artificial intelligence. "
            "Deep learning and machine learning are closely related."
        )
        bigrams = self.extractor.extract_bigrams(text, top_n=5)

        self.assertIsInstance(bigrams, list)
        # machine learning should be a top bigram
        phrases = [bg for bg, _ in bigrams]
        self.assertIn("machine learning", phrases)

    def test_extract_bigrams_too_short(self) -> None:
        """Test bigram extraction with insufficient text."""
        # "hello world" has exactly 2 words, so it produces one bigram
        bigrams = self.extractor.extract_bigrams("hello world")
        self.assertIsInstance(bigrams, list)
        # With only 2 words, we get exactly one bigram
        self.assertEqual(len(bigrams), 1)
        self.assertEqual(bigrams[0][0], "hello world")

    def test_position_boost(self) -> None:
        """Test that keywords appearing early get boosted."""
        text = (
            "Artificial intelligence is the main topic of this document. "
            "Machine learning is also discussed here. "
            "Deep learning appears much later in the document text. "
            "Neural networks are mentioned at the very end of this document."
        )
        keywords = self.extractor.extract_keywords(text, top_n=10)
        words = [kw for kw, _ in keywords]
        self.assertIn("artificial", words)
        self.assertIn("intelligence", words)


class TestTextSummarizer(unittest.TestCase):
    """Test cases for TextSummarizer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.summarizer = TextSummarizer()

    def test_summarize(self) -> None:
        """Test basic summarization."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a well-known pangram that contains every letter of the alphabet. "
            "Pangrams are often used for testing fonts and keyboard layouts. "
            "They provide a convenient way to see all characters in a typeface. "
            "Many languages have their own famous pangrams."
        )
        summary = self.summarizer.summarize(text, num_sentences=2)

        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLess(len(summary), len(text))

        # Summary should contain sentences from the original
        self.assertIn("pangram", summary.lower())

    def test_summarize_short_text(self) -> None:
        """Test summarization of text shorter than requested sentences."""
        text = "This is a single short sentence."
        summary = self.summarizer.summarize(text, num_sentences=3)
        self.assertEqual(summary, text)

    def test_summarize_empty_text(self) -> None:
        """Test summarization of empty text."""
        summary = self.summarizer.summarize("")
        self.assertEqual(summary, "")

    def test_summarize_preserves_order(self) -> None:
        """Test that summary preserves original sentence order."""
        text = (
            "First sentence introduces the topic clearly. "
            "Second sentence provides additional context and details. "
            "Third sentence explains the implications thoroughly. "
            "Fourth sentence concludes with final thoughts and recommendations."
        )
        summary = self.summarizer.summarize(text, num_sentences=2)

        # First sentence should appear before last in summary
        first_pos = summary.find("First")
        last_pos = summary.find("Fourth")
        if first_pos >= 0 and last_pos >= 0:
            self.assertLess(first_pos, last_pos)


class TestContentAnalyzer(unittest.TestCase):
    """Test cases for ContentAnalyzer."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.analyzer = ContentAnalyzer()
        self.sample_text = (
            "Python is a powerful programming language for data science. "
            "It offers excellent libraries for machine learning and artificial intelligence. "
            "Developers love Python for its simplicity and flexibility. "
            "The language continues to evolve with new features and improvements. "
            "Many companies use Python for web development, automation, and scientific computing."
        )

    def test_analyze(self) -> None:
        """Test full content analysis."""
        result = self.analyzer.analyze(
            url="https://example.com/article",
            title="Python Programming",
            text=self.sample_text,
            language="en",
        )

        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.url, "https://example.com/article")
        self.assertEqual(result.title, "Python Programming")
        self.assertEqual(result.language, "en")
        self.assertGreater(result.word_count, 0)
        self.assertGreater(result.sentence_count, 0)
        self.assertGreater(result.avg_word_length, 0)
        self.assertGreater(result.avg_sentence_length, 0)
        self.assertIsNotNone(result.summary)
        self.assertGreater(len(result.summary), 0)
        self.assertIsInstance(result.keywords, list)
        self.assertIsInstance(result.bigrams, list)
        self.assertIn(result.sentiment_label, ["positive", "negative", "neutral"])
        self.assertGreaterEqual(result.sentiment_confidence, 0.0)
        self.assertLessEqual(result.sentiment_confidence, 1.0)
        self.assertGreater(result.processing_time_ms, 0)

    def test_analyze_empty_text(self) -> None:
        """Test analysis of empty text."""
        result = self.analyzer.analyze(
            url="https://example.com",
            title="Empty",
            text="",
            language=None,
        )

        self.assertEqual(result.word_count, 0)
        # Empty text produces 1 sentence from the regex split minimum
        self.assertGreaterEqual(result.sentence_count, 0)
        self.assertEqual(result.sentiment_label, "neutral")

    def test_statistics_accuracy(self) -> None:
        """Test that computed statistics are accurate."""
        text = "Hello world. This is a test. Python programming is fun."
        result = self.analyzer.analyze(
            url="https://example.com",
            title="Test",
            text=text,
        )

        self.assertEqual(result.word_count, 10)
        self.assertEqual(result.sentence_count, 3)
        self.assertAlmostEqual(result.avg_sentence_length, 10 / 3, places=1)


if __name__ == "__main__":
    unittest.main()
