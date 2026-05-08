"""Command-line interface for the Intelligent Web Content Analyzer."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from src.analyzer import AnalysisResult
from src.batch_processor import BatchProcessor
from src.config import configure_logging, get_config
from src.scraper import WebScraper

logger = logging.getLogger("web_content_analyzer")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="web-analyzer",
        description=(
            "Intelligent Web Content Analyzer - "
            "Fetch, analyze, and summarize web articles."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s https://example.com/article
  %(prog)s https://example.com/article --format text --output result.txt
  %(prog)s --batch urls.txt --workers 4 --output results.json
  %(prog)s --batch urls.txt --format text --output-dir ./reports/
  %(prog)s https://example.com/article --summary-only --keywords 5
        """,
    )

    # Input
    parser.add_argument(
        "url",
        nargs="?",
        help="Single URL to analyze",
    )
    parser.add_argument(
        "-b",
        "--batch",
        metavar="FILE",
        help="File containing URLs (one per line) for batch processing",
    )

    # Output
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Output directory for batch results",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)",
    )

    # Analysis options
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only output the summary",
    )
    parser.add_argument(
        "--keywords",
        type=int,
        metavar="N",
        help="Number of keywords to extract (default: 10)",
    )

    # Processing options
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Number of parallel workers for batch mode (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        metavar="SEC",
        help="Request timeout in seconds (default: 30)",
    )

    # Logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error output",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
    )

    return parser


def read_url_file(file_path: str) -> List[str]:
    """Read URLs from a text file (one per line).

    Args:
        file_path: Path to the URL list file.

    Returns:
        List of URL strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no valid URLs.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {file_path}")

    urls = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            url = line.strip()
            if url and not url.startswith("#"):
                urls.append(url)

    if not urls:
        raise ValueError(f"No valid URLs found in {file_path}")

    logger.info("Loaded %d URLs from %s", len(urls), file_path)
    return urls


def _format_single_result(
    result: AnalysisResult,
    fmt: str,
    summary_only: bool = False,
) -> str:
    """Format a single result for display.

    Args:
        result: Analysis result.
        fmt: Output format ("json" or "text").
        summary_only: If True, only return the summary.

    Returns:
        Formatted string.
    """
    if summary_only:
        return result.summary

    if fmt == "json":
        return json.dumps(
            {
                "url": result.url,
                "title": result.title,
                "language": result.language,
                "statistics": {
                    "word_count": result.word_count,
                    "sentence_count": result.sentence_count,
                    "avg_word_length": result.avg_word_length,
                    "avg_sentence_length": result.avg_sentence_length,
                },
                "sentiment": {
                    "score": result.sentiment_score,
                    "label": result.sentiment_label,
                    "confidence": result.sentiment_confidence,
                },
                "keywords": [
                    {"word": w, "score": s} for w, s in result.keywords
                ],
                "bigrams": [
                    {"phrase": p, "score": s} for p, s in result.bigrams
                ],
                "readability": {
                    **result.readability_scores,
                    "estimated_grade": result.readability_grade,
                },
                "summary": result.summary,
                "processing_time_ms": result.processing_time_ms,
            },
            indent=2,
            ensure_ascii=False,
        )
    else:
        return _format_text(result)


def _format_text(result: AnalysisResult) -> str:
    """Format result as human-readable text.

    Args:
        result: Analysis result.

    Returns:
        Formatted text.
    """
    lines = [
        "=" * 70,
        "  INTELLIGENT WEB CONTENT ANALYZER - RESULTS",
        "=" * 70,
        "",
        f"  URL:      {result.url}",
        f"  Title:    {result.title}",
        "",
        "-" * 70,
        "  TEXT STATISTICS",
        "-" * 70,
        f"    Word Count:            {result.word_count:,}",
        f"    Sentence Count:        {result.sentence_count:,}",
        f"    Avg Word Length:       {result.avg_word_length:.1f} characters",
        f"    Avg Sentence Length:   {result.avg_sentence_length:.1f} words",
        "",
        "-" * 70,
        "  SENTIMENT ANALYSIS",
        "-" * 70,
        f"    Score:       {result.sentiment_score:+.4f}",
        f"    Label:       {result.sentiment_label.upper()}",
        f"    Confidence:  {result.sentiment_confidence:.2%}",
        "",
        "-" * 70,
        "  TOP KEYWORDS",
        "-" * 70,
    ]

    if result.keywords:
        for i, (word, score) in enumerate(result.keywords[:10], 1):
            bar = "#" * max(1, int(score * 500))
            lines.append(f"    {i:2d}. {word:<20} {score:.4f} {bar}")
    else:
        lines.append("    (no keywords extracted)")

    lines.extend([
        "",
        "-" * 70,
        "  TOP BIGRAMS",
        "-" * 70,
    ])

    if result.bigrams:
        for i, (phrase, score) in enumerate(result.bigrams[:8], 1):
            lines.append(f"    {i:2d}. {phrase:<30} {score:.4f}")
    else:
        lines.append("    (no bigrams extracted)")

    lines.extend([
        "",
        "-" * 70,
        "  READABILITY SCORES",
        "-" * 70,
        f"    Flesch Reading Ease:        {result.readability_scores.get('flesch_reading_ease', 0):.2f}",
        f"    Flesch-Kincaid Grade:       {result.readability_scores.get('flesch_kincaid_grade', 0):.2f}",
        f"    SMOG Index:                 {result.readability_scores.get('smog_index', 0):.2f}",
        f"    Coleman-Liau Index:         {result.readability_scores.get('coleman_liau_index', 0):.2f}",
        f"    Gunning Fog Index:          {result.readability_scores.get('gunning_fog_index', 0):.2f}",
        f"    Estimated Grade Level:      {result.readability_grade}",
        "",
        "-" * 70,
        "  SUMMARY",
        "-" * 70,
        "",
    ])

    # Wrap summary at 66 characters
    summary = result.summary or "(summary not available)"
    for para in summary.split("\n"):
        while para:
            chunk = para[:66]
            para = para[66:]
            lines.append(f"    {chunk}")

    lines.extend([
        "",
        f"  Processing time: {result.processing_time_ms:.2f} ms",
        "",
        "=" * 70,
    ])

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    # Configure logging
    log_level = "DEBUG" if args.verbose else ("WARNING" if args.quiet else "INFO")
    config = get_config()
    config.log_level = log_level
    configure_logging(config)

    # Validate arguments
    if not args.url and not args.batch:
        parser.error("Must provide either a URL or --batch FILE")
        return 1

    if args.url and args.batch:
        parser.error("Cannot use both single URL and --batch mode")
        return 1

    if args.keywords:
        config.top_keywords = args.keywords

    config.default_timeout = args.timeout

    processor = BatchProcessor(config=config)

    try:
        if args.batch:
            # Batch processing mode
            urls = read_url_file(args.batch)

            if not args.quiet:
                print(f"Processing {len(urls)} URLs...", file=sys.stderr)

            def progress(current: int, total: int, url: str) -> None:
                if not args.quiet:
                    print(
                        f"  [{current}/{total}] {url[:60]}...",
                        file=sys.stderr,
                    )

            results = processor.process_parallel(
                urls,
                max_workers=args.workers,
                progress_callback=progress if not args.quiet else None,
            )

            if not results:
                logger.error("No URLs were successfully processed")
                return 1

            if not args.quiet:
                print(
                    f"\nCompleted: {len(results)}/{len(urls)} succeeded",
                    file=sys.stderr,
                )

            # Determine output path
            if args.output:
                output_path = args.output
            elif args.output_dir:
                ext = "json" if args.format == "json" else "txt"
                timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
                output_path = str(
                    Path(args.output_dir) / f"batch_results_{timestamp}.{ext}"
                )
            else:
                # Print to stdout
                for result in results:
                    print(_format_single_result(
                        result, args.format, args.summary_only
                    ))
                return 0

            processor.save_results(results, output_path, args.format)

            if not args.quiet:
                print(f"\nResults saved to: {output_path}", file=sys.stderr)

        else:
            # Single URL mode
            result = processor.process_single(args.url)
            if result is None:
                logger.error("Failed to process URL: %s", args.url)
                return 1

            output = _format_single_result(
                result, args.format, args.summary_only
            )

            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(output + "\n")
                if not args.quiet:
                    print(f"Results saved to: {args.output}", file=sys.stderr)
            else:
                print(output)

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as exc:
        logger.error("Error: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
