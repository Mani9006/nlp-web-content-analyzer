"""Tests for the web scraper module."""

import unittest
from unittest.mock import MagicMock, patch

import requests

from src.scraper import ScrapedContent, WebScraper


SAMPLE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Test Article Title</title>
    <meta name="description" content="A test article description">
    <meta name="author" content="John Doe">
    <meta property="article:published_time" content="2024-01-15T10:00:00Z">
</head>
<body>
    <header>
        <nav>Home | About | Contact</nav>
    </header>
    <main>
        <article>
            <h1>Test Article Title</h1>
            <div class="author">John Doe</div>
            <div class="post-content">
                <p>This is the first paragraph of the test article.</p>
                <p>It contains multiple sentences for testing purposes.</p>
                <p>The content should be extracted cleanly without navigation elements.</p>
            </div>
        </article>
    </main>
    <footer>
        <p>Copyright 2024</p>
        <nav>Footer links</nav>
    </footer>
</body>
</html>
"""

HTML_NO_ARTICLE = """
<!DOCTYPE html>
<html lang="fr">
<head><title>Simple Page</title></head>
<body>
    <div class="content">
        <p>This page has no article tag. The scraper should fall back to the body content.</p>
        <p>Second paragraph here with more text content for the analyzer to process.</p>
    </div>
</body>
</html>
"""

HTML_EMPTY_BODY = """
<!DOCTYPE html>
<html>
<head><title>Empty</title></head>
<body></body>
</html>
"""


class TestWebScraper(unittest.TestCase):
    """Test cases for WebScraper."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.scraper = WebScraper()

    def test_is_valid_url_with_valid_urls(self) -> None:
        """Test URL validation with valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://example.com/path?query=1",
            "https://sub.domain.example.co.uk:8080/path",
        ]
        for url in valid_urls:
            self.assertTrue(
                self.scraper._is_valid_url(url),
                f"Expected {url} to be valid",
            )

    def test_is_valid_url_with_invalid_urls(self) -> None:
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://files.example.com",
            "javascript:void(0)",
            "/relative/path",
        ]
        for url in invalid_urls:
            self.assertFalse(
                self.scraper._is_valid_url(url),
                f"Expected {url} to be invalid",
            )

    @patch("src.scraper.requests.Session.get")
    def test_scrape_success(self, mock_get: MagicMock) -> None:
        """Test successful scraping of a well-formed article page."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = SAMPLE_HTML
        mock_response.content = SAMPLE_HTML.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = self.scraper.scrape("https://example.com/article")

        self.assertIsInstance(result, ScrapedContent)
        self.assertEqual(result.url, "https://example.com/article")
        self.assertEqual(result.title, "Test Article Title")
        self.assertIn("first paragraph", result.text)
        self.assertNotIn("Footer links", result.text)
        self.assertNotIn("Home | About | Contact", result.text)
        self.assertEqual(result.author, "John Doe")
        self.assertEqual(result.publish_date, "2024-01-15T10:00:00Z")
        self.assertEqual(result.description, "A test article description")
        self.assertEqual(result.language, "en")
        self.assertEqual(result.status_code, 200)
        self.assertGreater(result.fetch_time_ms, 0)

    @patch("src.scraper.requests.Session.get")
    def test_scrape_no_article_tag(self, mock_get: MagicMock) -> None:
        """Test scraping a page without an article tag."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = HTML_NO_ARTICLE
        mock_response.content = HTML_NO_ARTICLE.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = self.scraper.scrape("https://example.com/simple")

        self.assertEqual(result.title, "Simple Page")
        self.assertIn("no article tag", result.text)
        self.assertEqual(result.language, "fr")

    @patch("src.scraper.requests.Session.get")
    def test_scrape_empty_body(self, mock_get: MagicMock) -> None:
        """Test scraping a page with an empty body."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = HTML_EMPTY_BODY
        mock_response.content = HTML_EMPTY_BODY.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = self.scraper.scrape("https://example.com/empty")

        self.assertEqual(result.title, "Empty")
        self.assertEqual(result.text, "")

    def test_scrape_invalid_url(self) -> None:
        """Test that scraping an invalid URL raises ValueError."""
        with self.assertRaises(ValueError):
            self.scraper.scrape("not-a-valid-url")

        with self.assertRaises(ValueError):
            self.scraper.scrape("")

    @patch("src.scraper.requests.Session.get")
    def test_scrape_http_error(self, mock_get: MagicMock) -> None:
        """Test handling of HTTP errors."""
        mock_get.side_effect = requests.HTTPError("404 Not Found")

        with self.assertRaises(requests.HTTPError):
            self.scraper.scrape("https://example.com/notfound")

    @patch("src.scraper.requests.Session.get")
    def test_scrape_connection_error(self, mock_get: MagicMock) -> None:
        """Test handling of connection errors."""
        mock_get.side_effect = requests.ConnectionError("Connection refused")

        with self.assertRaises(requests.ConnectionError):
            self.scraper.scrape("https://example.com/down")

    @patch("src.scraper.requests.Session.get")
    def test_scrape_timeout(self, mock_get: MagicMock) -> None:
        """Test handling of request timeouts."""
        mock_get.side_effect = requests.Timeout("Request timed out")

        with self.assertRaises(requests.Timeout):
            self.scraper.scrape("https://example.com/slow")

    @patch("src.scraper.requests.Session.get")
    def test_scrape_with_retry(self, mock_get: MagicMock) -> None:
        """Test that retries are attempted on failure."""
        mock_get.side_effect = [
            requests.ConnectionError("First attempt failed"),
            requests.ConnectionError("Second attempt failed"),
            MagicMock(
                status_code=200,
                text=SAMPLE_HTML,
                content=SAMPLE_HTML.encode("utf-8"),
                raise_for_status=MagicMock(),
            ),
        ]

        result = self.scraper.scrape("https://example.com/article")
        self.assertEqual(result.title, "Test Article Title")
        self.assertEqual(mock_get.call_count, 3)

    @patch("src.scraper.requests.Session.get")
    def test_scrape_retry_exhausted(self, mock_get: MagicMock) -> None:
        """Test that all retries are exhausted properly."""
        mock_get.side_effect = requests.ConnectionError("Always fails")

        with self.assertRaises(requests.RequestException):
            self.scraper.scrape("https://example.com/fails")

        self.assertEqual(mock_get.call_count, self.scraper.config.max_retries)

    @patch("src.scraper.WebScraper.scrape")
    def test_scrape_multiple(self, mock_scrape: MagicMock) -> None:
        """Test batch scraping with mixed results."""
        mock_scrape.side_effect = [
            ScrapedContent(
                url="https://example.com/1",
                title="Article 1",
                text="Content 1",
                html="<html></html>",
                status_code=200,
                fetch_time_ms=100.0,
            ),
            Exception("Failed"),
            ScrapedContent(
                url="https://example.com/3",
                title="Article 3",
                text="Content 3",
                html="<html></html>",
                status_code=200,
                fetch_time_ms=150.0,
            ),
        ]

        results = self.scraper.scrape_multiple([
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ])

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].title, "Article 1")
        self.assertEqual(results[1].title, "")  # Failed
        self.assertEqual(results[2].title, "Article 3")


if __name__ == "__main__":
    unittest.main()
