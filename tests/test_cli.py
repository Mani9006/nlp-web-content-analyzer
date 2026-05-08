"""Tests for the CLI module."""

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cli import create_parser, main, read_url_file


class TestArgumentParser(unittest.TestCase):
    """Test cases for CLI argument parsing."""

    def test_create_parser(self) -> None:
        """Test that the parser is created with expected arguments."""
        parser = create_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_parse_single_url(self) -> None:
        """Test parsing a single URL argument."""
        parser = create_parser()
        args = parser.parse_args(["https://example.com/article"])
        self.assertEqual(args.url, "https://example.com/article")
        self.assertIsNone(args.batch)
        self.assertEqual(args.format, "json")

    def test_parse_batch_mode(self) -> None:
        """Test parsing batch mode arguments."""
        parser = create_parser()
        args = parser.parse_args(["--batch", "urls.txt"])
        self.assertIsNone(args.url)
        self.assertEqual(args.batch, "urls.txt")

    def test_parse_format_option(self) -> None:
        """Test parsing format option."""
        parser = create_parser()
        args = parser.parse_args(["--batch", "urls.txt", "--format", "text"])
        self.assertEqual(args.format, "text")

    def test_parse_workers_option(self) -> None:
        """Test parsing workers option."""
        parser = create_parser()
        args = parser.parse_args(["--batch", "urls.txt", "--workers", "8"])
        self.assertEqual(args.workers, 8)

    def test_parse_keywords_option(self) -> None:
        """Test parsing keywords count option."""
        parser = create_parser()
        args = parser.parse_args(["https://example.com", "--keywords", "5"])
        self.assertEqual(args.keywords, 5)

    def test_parse_output_option(self) -> None:
        """Test parsing output file option."""
        parser = create_parser()
        args = parser.parse_args(["https://example.com", "--output", "result.json"])
        self.assertEqual(args.output, "result.json")

    def test_parse_verbose_option(self) -> None:
        """Test parsing verbose option."""
        parser = create_parser()
        args = parser.parse_args(["https://example.com", "--verbose"])
        self.assertTrue(args.verbose)

    def test_parse_summary_only(self) -> None:
        """Test parsing summary-only option."""
        parser = create_parser()
        args = parser.parse_args(["https://example.com", "--summary-only"])
        self.assertTrue(args.summary_only)


class TestReadUrlFile(unittest.TestCase):
    """Test cases for URL file reading."""

    def setUp(self) -> None:
        """Set up temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_read_valid_url_file(self) -> None:
        """Test reading a valid URL file."""
        filepath = os.path.join(self.temp_dir, "urls.txt")
        with open(filepath, "w") as f:
            f.write("https://example.com/1\n")
            f.write("https://example.com/2\n")
            f.write("https://example.com/3\n")

        urls = read_url_file(filepath)
        self.assertEqual(len(urls), 3)
        self.assertEqual(urls[0], "https://example.com/1")

    def test_read_url_file_with_comments(self) -> None:
        """Test reading a URL file with comments and blank lines."""
        filepath = os.path.join(self.temp_dir, "urls.txt")
        with open(filepath, "w") as f:
            f.write("# This is a comment\n")
            f.write("\n")
            f.write("https://example.com/1\n")
            f.write("  \n")
            f.write("https://example.com/2\n")

        urls = read_url_file(filepath)
        self.assertEqual(len(urls), 2)

    def test_read_nonexistent_file(self) -> None:
        """Test reading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_url_file("/nonexistent/path/urls.txt")

    def test_read_empty_file(self) -> None:
        """Test reading an empty file raises ValueError."""
        filepath = os.path.join(self.temp_dir, "empty.txt")
        with open(filepath, "w") as f:
            f.write("")

        with self.assertRaises(ValueError):
            read_url_file(filepath)

    def test_read_file_with_only_comments(self) -> None:
        """Test reading a file with only comments raises ValueError."""
        filepath = os.path.join(self.temp_dir, "comments.txt")
        with open(filepath, "w") as f:
            f.write("# Comment 1\n")
            f.write("# Comment 2\n")

        with self.assertRaises(ValueError):
            read_url_file(filepath)


class TestMainFunction(unittest.TestCase):
    """Test cases for main CLI entry point."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_url = "https://example.com/article"

    def tearDown(self) -> None:
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("src.cli.BatchProcessor")
    def test_main_single_url_json(self, mock_processor_class: MagicMock) -> None:
        """Test main with single URL, JSON output."""
        mock_result = MagicMock()
        mock_result.url = self.test_url
        mock_result.title = "Test Title"
        mock_result.word_count = 100
        mock_result.sentence_count = 10
        mock_result.avg_word_length = 5.0
        mock_result.avg_sentence_length = 10.0
        mock_result.sentiment_score = 0.5
        mock_result.sentiment_label = "positive"
        mock_result.sentiment_confidence = 0.8
        mock_result.keywords = [("python", 0.1)]
        mock_result.bigrams = [("machine learning", 0.05)]
        mock_result.readability_scores = {
            "flesch_reading_ease": 60.0,
            "flesch_kincaid_grade": 8.0,
            "smog_index": 9.0,
            "coleman_liau_index": 8.5,
            "automated_readability_index": 7.5,
            "gunning_fog_index": 10.0,
        }
        mock_result.readability_grade = "8th Grade"
        mock_result.summary = "This is a summary."
        mock_result.processing_time_ms = 50.0
        mock_result.language = "en"

        mock_processor = MagicMock()
        mock_processor.process_single.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        exit_code = main([self.test_url])
        self.assertEqual(exit_code, 0)
        mock_processor.process_single.assert_called_once_with(self.test_url)

    @patch("src.cli.BatchProcessor")
    def test_main_single_url_not_found(self, mock_processor_class: MagicMock) -> None:
        """Test main with URL that returns no content."""
        mock_processor = MagicMock()
        mock_processor.process_single.return_value = None
        mock_processor_class.return_value = mock_processor

        exit_code = main([self.test_url])
        self.assertEqual(exit_code, 1)

    def test_main_no_arguments(self) -> None:
        """Test main with no arguments returns error."""
        with self.assertRaises(SystemExit) as ctx:
            main([])
        self.assertEqual(ctx.exception.code, 2)

    @patch("src.cli.read_url_file")
    @patch("src.cli.BatchProcessor")
    def test_main_batch_mode(
        self,
        mock_processor_class: MagicMock,
        mock_read_file: MagicMock,
    ) -> None:
        """Test main in batch mode."""
        mock_read_file.return_value = [
            "https://example.com/1",
            "https://example.com/2",
        ]

        mock_result1 = MagicMock()
        mock_result1.url = "https://example.com/1"
        mock_result2 = MagicMock()
        mock_result2.url = "https://example.com/2"

        mock_processor = MagicMock()
        mock_processor.process_parallel.return_value = [mock_result1, mock_result2]
        mock_processor.save_results.return_value = "/path/to/results.json"
        mock_processor_class.return_value = mock_processor

        output_path = os.path.join(self.temp_dir, "results.json")
        exit_code = main(["--batch", "urls.txt", "--output", output_path])

        self.assertEqual(exit_code, 0)
        mock_read_file.assert_called_once_with("urls.txt")
        mock_processor.process_parallel.assert_called_once()

    @patch("src.cli.BatchProcessor")
    def test_main_output_file(self, mock_processor_class: MagicMock) -> None:
        """Test main with output file."""
        mock_result = MagicMock()
        mock_result.url = self.test_url
        mock_result.title = "Test"
        mock_result.word_count = 50
        mock_result.sentence_count = 5
        mock_result.avg_word_length = 4.0
        mock_result.avg_sentence_length = 10.0
        mock_result.sentiment_score = 0.0
        mock_result.sentiment_label = "neutral"
        mock_result.sentiment_confidence = 0.5
        mock_result.keywords = []
        mock_result.bigrams = []
        mock_result.readability_scores = {
            "flesch_reading_ease": 70.0,
            "flesch_kincaid_grade": 6.0,
            "smog_index": 7.0,
            "coleman_liau_index": 6.5,
            "automated_readability_index": 5.5,
            "gunning_fog_index": 8.0,
        }
        mock_result.readability_grade = "6th Grade"
        mock_result.summary = "A summary."
        mock_result.processing_time_ms = 30.0
        mock_result.language = "en"

        mock_processor = MagicMock()
        mock_processor.process_single.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        output_file = os.path.join(self.temp_dir, "output.txt")
        exit_code = main([self.test_url, "--output", output_file])

        self.assertEqual(exit_code, 0)
        self.assertTrue(os.path.exists(output_file))

    @patch("src.cli.BatchProcessor")
    def test_main_keyboard_interrupt(self, mock_processor_class: MagicMock) -> None:
        """Test handling of KeyboardInterrupt."""
        mock_processor = MagicMock()
        mock_processor.process_single.side_effect = KeyboardInterrupt()
        mock_processor_class.return_value = mock_processor

        exit_code = main([self.test_url])
        self.assertEqual(exit_code, 130)

    @patch("src.cli.BatchProcessor")
    def test_main_exception(self, mock_processor_class: MagicMock) -> None:
        """Test handling of general exceptions."""
        mock_processor = MagicMock()
        mock_processor.process_single.side_effect = Exception("Unexpected error")
        mock_processor_class.return_value = mock_processor

        exit_code = main([self.test_url])
        self.assertEqual(exit_code, 1)

    @patch("src.cli.BatchProcessor")
    def test_main_summary_only(self, mock_processor_class: MagicMock) -> None:
        """Test main with --summary-only flag."""
        mock_result = MagicMock()
        mock_result.summary = "This is the article summary."
        mock_result.url = self.test_url
        mock_result.title = "Test"
        mock_result.word_count = 50
        mock_result.sentence_count = 5
        mock_result.avg_word_length = 4.0
        mock_result.avg_sentence_length = 10.0
        mock_result.sentiment_score = 0.0
        mock_result.sentiment_label = "neutral"
        mock_result.sentiment_confidence = 0.5
        mock_result.keywords = []
        mock_result.bigrams = []
        mock_result.readability_scores = {
            "flesch_reading_ease": 70.0,
            "flesch_kincaid_grade": 6.0,
            "smog_index": 7.0,
            "coleman_liau_index": 6.5,
            "automated_readability_index": 5.5,
            "gunning_fog_index": 8.0,
        }
        mock_result.readability_grade = "6th Grade"
        mock_result.processing_time_ms = 30.0
        mock_result.language = "en"

        mock_processor = MagicMock()
        mock_processor.process_single.return_value = mock_result
        mock_processor_class.return_value = mock_processor

        exit_code = main([self.test_url, "--summary-only"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
