"""Configuration settings for the Intelligent Web Content Analyzer."""

import logging
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Application configuration container."""

    # Scraping settings
    default_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    request_headers: Optional[dict] = None

    # Analysis settings
    summary_sentences: int = 3
    top_keywords: int = 10
    min_keyword_length: int = 3

    # Output settings
    default_output_format: str = "json"  # "json" or "text"
    output_dir: str = "data"

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass creation."""
        if self.request_headers is None:
            self.request_headers = {
                "User-Agent": self.user_agent,
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
            }


def get_config() -> Config:
    """Return application configuration from environment or defaults.

    Environment variables:
        WEB_ANALYZER_TIMEOUT: Request timeout in seconds.
        WEB_ANALYZER_MAX_RETRIES: Maximum retry attempts.
        WEB_ANALYZER_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR).
        WEB_ANALYZER_OUTPUT_DIR: Directory for output files.

    Returns:
        Config: Populated configuration object.
    """
    return Config(
        default_timeout=int(os.getenv("WEB_ANALYZER_TIMEOUT", "30")),
        max_retries=int(os.getenv("WEB_ANALYZER_MAX_RETRIES", "3")),
        log_level=os.getenv("WEB_ANALYZER_LOG_LEVEL", "INFO"),
        output_dir=os.getenv("WEB_ANALYZER_OUTPUT_DIR", "data"),
    )


def configure_logging(config: Optional[Config] = None) -> logging.Logger:
    """Configure and return the root logger for the application.

    Args:
        config: Optional configuration object. Uses defaults if not provided.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if config is None:
        config = get_config()

    logger = logging.getLogger("web_content_analyzer")
    logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logger.level)
        formatter = logging.Formatter(config.log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
