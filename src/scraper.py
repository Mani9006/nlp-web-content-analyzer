"""Web scraping module for fetching and extracting article content."""

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from src.config import Config, get_config

logger = logging.getLogger("web_content_analyzer")


@dataclass
class ScrapedContent:
    """Container for scraped web content."""

    url: str
    title: str
    text: str
    html: str
    author: Optional[str] = None
    publish_date: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    status_code: int = 200
    fetch_time_ms: float = 0.0


class WebScraper:
    """Scraper for extracting article content from web pages."""

    # HTML tags to remove during text extraction
    UNWANTED_TAGS = {
        "script", "style", "nav", "footer", "header",
        "aside", "advertisement", "iframe", "noscript",
        "svg", "canvas", "form", "button", "input",
    }

    # Meta tags that may contain publish dates
    DATE_META_TAGS = [
        "article:published_time",
        "publishedDate",
        "datePublished",
        "pubdate",
        "date",
    ]

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the scraper with configuration.

        Args:
            config: Application configuration. Uses defaults if not provided.
        """
        self.config = config or get_config()
        self.session = requests.Session()
        self.session.headers.update(self.config.request_headers or {})

    def _is_valid_url(self, url: str) -> bool:
        """Validate that the URL has a proper scheme and netloc.

        Args:
            url: The URL string to validate.

        Returns:
            bool: True if the URL is valid, False otherwise.
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme in ("http", "https") and parsed.netloc)
        except Exception:
            return False

    def _fetch_with_retry(self, url: str) -> requests.Response:
        """Fetch a URL with retry logic.

        Args:
            url: The URL to fetch.

        Returns:
            requests.Response: The HTTP response.

        Raises:
            requests.RequestException: If all retry attempts fail.
        """
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    timeout=self.config.default_timeout,
                    allow_redirects=True,
                )
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                last_exception = exc
                logger.warning(
                    "Attempt %d/%d failed for %s: %s",
                    attempt,
                    self.config.max_retries,
                    url,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)

        raise last_exception or requests.RequestException(
            f"Failed to fetch {url} after {self.config.max_retries} attempts"
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract the article title from the page.

        Args:
            soup: Parsed BeautifulSoup object.

        Returns:
            str: The extracted title or an empty string.
        """
        # Try article-specific title tags first
        for selector in ["h1", "h1.entry-title", "h1.article-title", ".post-title"]:
            elem = soup.select_one(selector)
            if elem and elem.get_text(strip=True):
                return elem.get_text(strip=True)

        # Fall back to <title> tag
        if soup.title and soup.title.string:
            return soup.title.string.strip()

        return ""

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the article author from the page.

        Args:
            soup: Parsed BeautifulSoup object.

        Returns:
            Optional[str]: The author name or None.
        """
        # Try common author meta tags
        for name in ["author", "article:author", "og:article:author"]:
            tag = soup.find("meta", attrs={"name": name}) or soup.find(
                "meta", attrs={"property": name}
            )
            if tag and tag.get("content"):
                return tag["content"].strip()

        # Try schema.org author
        author_tag = soup.find(attrs={"class": re.compile(r"author", re.I)})
        if author_tag:
            return author_tag.get_text(strip=True)

        return None

    def _extract_publish_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the publish date from the page.

        Args:
            soup: Parsed BeautifulSoup object.

        Returns:
            Optional[str]: The publish date string or None.
        """
        for name in self.DATE_META_TAGS:
            tag = soup.find("meta", attrs={"name": name}) or soup.find(
                "meta", attrs={"property": name}
            )
            if tag and tag.get("content"):
                return tag["content"].strip()

        # Try time element
        time_tag = soup.find("time")
        if time_tag and time_tag.get("datetime"):
            return time_tag["datetime"]

        return None

    def _extract_description(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract the meta description from the page.

        Args:
            soup: Parsed BeautifulSoup object.

        Returns:
            Optional[str]: The description or None.
        """
        for name in ["description", "og:description", "twitter:description"]:
            tag = soup.find("meta", attrs={"name": name}) or soup.find(
                "meta", attrs={"property": name}
            )
            if tag and tag.get("content"):
                return tag["content"].strip()

        return None

    def _clean_html(self, soup: BeautifulSoup) -> str:
        """Remove unwanted tags and extract clean text.

        Args:
            soup: Parsed BeautifulSoup object.

        Returns:
            str: Clean extracted text.
        """
        # Remove unwanted tags
        for tag_name in self.UNWANTED_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        # Try to find main content area
        main_content = None
        for selector in [
            "article",
            "main",
            "[role='main']",
            ".post-content",
            ".entry-content",
            ".article-body",
            ".content",
        ]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content:
            text = main_content.get_text(separator="\n", strip=True)
        else:
            # Fall back to body content
            body = soup.find("body")
            text = body.get_text(separator="\n", strip=True) if body else ""

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def scrape(self, url: str) -> ScrapedContent:
        """Scrape a single URL and extract article content.

        Args:
            url: The URL to scrape.

        Returns:
            ScrapedContent: Extracted content container.

        Raises:
            ValueError: If the URL is invalid.
            requests.RequestException: If the fetch fails.
        """
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid URL: {url}")

        logger.info("Scraping URL: %s", url)
        start_time = time.perf_counter()

        response = self._fetch_with_retry(url)
        soup = BeautifulSoup(response.content, "html.parser")

        fetch_time = (time.perf_counter() - start_time) * 1000

        content = ScrapedContent(
            url=url,
            title=self._extract_title(soup),
            text=self._clean_html(soup),
            html=response.text,
            author=self._extract_author(soup),
            publish_date=self._extract_publish_date(soup),
            description=self._extract_description(soup),
            language=soup.find("html").get("lang") if soup.find("html") else None,
            status_code=response.status_code,
            fetch_time_ms=round(fetch_time, 2),
        )

        logger.info(
            "Scraped %s - Title: '%s' - %.1f ms - %d chars",
            url,
            content.title[:50],
            content.fetch_time_ms,
            len(content.text),
        )
        return content

    def scrape_multiple(self, urls: list[str]) -> list[ScrapedContent]:
        """Scrape multiple URLs sequentially.

        Args:
            urls: List of URLs to scrape.

        Returns:
            list[ScrapedContent]: List of extracted content containers.
        """
        results = []
        for url in urls:
            try:
                result = self.scrape(url)
                results.append(result)
            except Exception as exc:
                logger.error("Failed to scrape %s: %s", url, exc)
                results.append(
                    ScrapedContent(
                        url=url,
                        title="",
                        text="",
                        html="",
                        status_code=0,
                        fetch_time_ms=0.0,
                    )
                )
        return results
