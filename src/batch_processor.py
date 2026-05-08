"""Batch processing module for analyzing multiple URLs."""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, List, Optional

from src.analyzer import AnalysisResult, ContentAnalyzer
from src.config import Config, get_config
from src.readability import ReadabilityAnalyzer
from src.scraper import ScrapedContent, WebScraper

logger = logging.getLogger("web_content_analyzer")


class BatchProcessor:
    """Process multiple URLs for content analysis."""

    def __init__(
        self,
        scraper: Optional[WebScraper] = None,
        analyzer: Optional[ContentAnalyzer] = None,
        readability: Optional[ReadabilityAnalyzer] = None,
        config: Optional[Config] = None,
    ) -> None:
        """Initialize the batch processor.

        Args:
            scraper: Web scraper instance.
            analyzer: Content analyzer instance.
            readability: Readability analyzer instance.
            config: Application configuration.
        """
        self.config = config or get_config()
        self.scraper = scraper or WebScraper(self.config)
        self.analyzer = analyzer or ContentAnalyzer(self.config)
        self.readability = readability or ReadabilityAnalyzer(self.config)

    def process_single(self, url: str) -> Optional[AnalysisResult]:
        """Process a single URL through scrape + analyze pipeline.

        Args:
            url: The URL to process.

        Returns:
            AnalysisResult if successful, None otherwise.
        """
        try:
            scraped = self.scraper.scrape(url)
            if not scraped.text:
                logger.warning("No content extracted from %s", url)
                return None

            result = self.analyzer.analyze(
                url=scraped.url,
                title=scraped.title,
                text=scraped.text,
                language=scraped.language,
            )

            # Add readability scores
            readability_scores = self.readability.analyze(scraped.text)
            result.readability_scores = {
                "flesch_reading_ease": readability_scores.flesch_reading_ease,
                "flesch_kincaid_grade": readability_scores.flesch_kincaid_grade,
                "smog_index": readability_scores.smog_index,
                "coleman_liau_index": readability_scores.coleman_liau_index,
                "automated_readability_index": readability_scores.automated_readability_index,
                "gunning_fog_index": readability_scores.gunning_fog_index,
                "avg_sentence_length": readability_scores.avg_sentence_length,
                "avg_syllables_per_word": readability_scores.avg_syllables_per_word,
                "complex_word_percentage": readability_scores.complex_word_percentage,
            }
            result.readability_grade = readability_scores.grade_level

            return result

        except Exception as exc:
            logger.error("Error processing %s: %s", url, exc)
            return None

    def process_sequential(
        self,
        urls: List[str],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[AnalysisResult]:
        """Process URLs sequentially.

        Args:
            urls: List of URLs to process.
            progress_callback: Optional callback(current, total, url).

        Returns:
            List of successful analysis results.
        """
        results = []
        total = len(urls)

        logger.info("Starting sequential batch processing of %d URLs", total)
        start_time = time.perf_counter()

        for idx, url in enumerate(urls, 1):
            if progress_callback:
                progress_callback(idx, total, url)

            logger.info("[%d/%d] Processing: %s", idx, total, url)
            result = self.process_single(url)
            if result:
                results.append(result)

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Batch complete: %d/%d succeeded in %.2f seconds",
            len(results),
            total,
            elapsed,
        )
        return results

    def process_parallel(
        self,
        urls: List[str],
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[AnalysisResult]:
        """Process URLs in parallel using a thread pool.

        Args:
            urls: List of URLs to process.
            max_workers: Maximum number of concurrent workers.
            progress_callback: Optional callback(current, total, url).

        Returns:
            List of successful analysis results.
        """
        results = []
        total = len(urls)
        completed = 0

        logger.info(
            "Starting parallel batch processing of %d URLs (%d workers)",
            total,
            max_workers,
        )
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.process_single, url): url
                for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                completed += 1

                if progress_callback:
                    progress_callback(completed, total, url)

                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        logger.info(
                            "[%d/%d] Completed: %s", completed, total, url
                        )
                except Exception as exc:
                    logger.error(
                        "[%d/%d] Failed: %s - %s", completed, total, url, exc
                    )

        elapsed = time.perf_counter() - start_time
        logger.info(
            "Parallel batch complete: %d/%d succeeded in %.2f seconds",
            len(results),
            total,
            elapsed,
        )
        return results

    @staticmethod
    def save_results(
        results: List[AnalysisResult],
        output_path: str,
        format_type: str = "json",
    ) -> str:
        """Save analysis results to a file.

        Args:
            results: List of analysis results.
            output_path: Output file path.
            format_type: "json" or "text".

        Returns:
            Absolute path to the saved file.
        """
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if format_type.lower() == "json":
            data = [asdict(r) for r in results]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                for result in results:
                    f.write(_format_text_output(result))
                    f.write("\n" + "=" * 80 + "\n\n")

        logger.info("Results saved to: %s", output_path)
        return os.path.abspath(output_path)


def _format_text_output(result: AnalysisResult) -> str:
    """Format a single analysis result as human-readable text.

    Args:
        result: Analysis result to format.

    Returns:
        Formatted text string.
    """
    lines = [
        f"URL:         {result.url}",
        f"Title:       {result.title}",
        f"Language:    {result.language or 'unknown'}",
        "",
        "--- TEXT STATISTICS ---",
        f"Word Count:           {result.word_count}",
        f"Sentence Count:       {result.sentence_count}",
        f"Avg Word Length:      {result.avg_word_length} chars",
        f"Avg Sentence Length:  {result.avg_sentence_length} words",
        "",
        "--- SENTIMENT ---",
        f"Score:       {result.sentiment_score}",
        f"Label:       {result.sentiment_label}",
        f"Confidence:  {result.sentiment_confidence}",
        "",
        "--- KEYWORDS ---",
    ]

    if result.keywords:
        for word, score in result.keywords:
            lines.append(f"  {word:<25} {score:.6f}")
    else:
        lines.append("  (none found)")

    lines.extend([
        "",
        "--- BIGRAMS ---",
    ])

    if result.bigrams:
        for phrase, score in result.bigrams:
            lines.append(f"  {phrase:<35} {score:.6f}")
    else:
        lines.append("  (none found)")

    lines.extend([
        "",
        "--- READABILITY ---",
        f"Flesch Reading Ease:        {result.readability_scores.get('flesch_reading_ease', 'N/A')}",
        f"Flesch-Kincaid Grade:       {result.readability_scores.get('flesch_kincaid_grade', 'N/A')}",
        f"SMOG Index:                 {result.readability_scores.get('smog_index', 'N/A')}",
        f"Coleman-Liau Index:         {result.readability_scores.get('coleman_liau_index', 'N/A')}",
        f"Automated Readability:      {result.readability_scores.get('automated_readability_index', 'N/A')}",
        f"Gunning Fog Index:          {result.readability_scores.get('gunning_fog_index', 'N/A')}",
        f"Estimated Grade:            {result.readability_grade}",
        "",
        "--- SUMMARY ---",
        result.summary or "(not generated)",
        "",
        f"Processing time: {result.processing_time_ms:.2f} ms",
    ])

    return "\n".join(lines)
