# Intelligent Web Content Analyzer

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT">
  <img src="https://img.shields.io/badge/code%20style-black-black?style=flat-square" alt="Code style: black">
  <img src="https://img.shields.io/badge/tests-pytest-brightgreen?style=flat-square&logo=pytest" alt="Tests: pytest">
  <img src="https://img.shields.io/badge/coverage-90%25-brightgreen?style=flat-square" alt="Coverage: 90%">
</p>

<p align="center">
  <b>A powerful Python tool that fetches web articles, extracts content, and performs comprehensive NLP analysis.</b>
</p>

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Testing](#testing)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Features

### Core Capabilities

- **Web Scraping**: Fetches articles from any URL with retry logic, session management, and configurable timeouts
- **Content Extraction**: Intelligently extracts article text by removing navigation, ads, scripts, and boilerplate
- **Metadata Extraction**: Captures title, author, publish date, language, and meta descriptions
- **Sentiment Analysis**: Lexicon-based sentiment scoring with negation detection and intensity modifiers
- **Keyword Extraction**: TF-based keyword and bigram extraction with position boosting
- **Text Summarization**: Extractive summarization using sentence importance scoring
- **Readability Analysis**: Six standard readability formulas (Flesch, Flesch-Kincaid, SMOG, Coleman-Liau, ARI, Gunning Fog)

### Processing Modes

- **Single URL Analysis**: Quick analysis of individual articles
- **Batch Processing**: Process multiple URLs from a file (sequential or parallel)
- **Parallel Execution**: ThreadPoolExecutor-based concurrent processing
- **Progress Callbacks**: Real-time progress reporting for batch operations

### Output Formats

- **JSON**: Structured output with all metrics and scores
- **Text**: Human-readable formatted report with visual bars
- **File Output**: Save results to files with auto-generated timestamps
- **Stdout**: Direct console output for piping and scripting

### Developer Features

- Type hints throughout
- Comprehensive docstrings
- Structured logging with configurable levels
- Input validation and error handling
- Unit tests with mocked HTTP requests
- PEP 8 compliant code (black formatted)

---

## Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| HTTP Client | `requests` | Web page fetching with sessions |
| HTML Parsing | `beautifulsoup4` + `lxml` | Content extraction from HTML |
| NLP Engine | Custom lexicon-based | Sentiment, keywords, summarization |
| Concurrency | `concurrent.futures` | Parallel batch processing |
| Testing | `pytest` + `unittest.mock` | Unit and integration tests |
| Code Quality | `black`, `flake8`, `mypy` | Formatting, linting, type checking |
| Packaging | `setuptools` + `pyproject.toml` | Distribution and installation |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip or pipenv

### From Source

```bash
# Clone the repository
git clone https://github.com/example/intelligent-web-content-analyzer.git
cd intelligent-web-content-analyzer

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### Quick Install

```bash
pip install -e .
```

---

## Usage

### Command-Line Interface

The tool installs as `web-analyzer` after setup.

#### Analyze a Single URL

```bash
# JSON output (default)
web-analyzer https://example.com/article

# Text output
web-analyzer https://example.com/article --format text

# Save to file
web-analyzer https://example.com/article --output result.json
web-analyzer https://example.com/article --format text --output result.txt

# Get only the summary
web-analyzer https://example.com/article --summary-only

# Extract top 5 keywords
web-analyzer https://example.com/article --keywords 5

# Verbose logging
web-analyzer https://example.com/article --verbose
```

#### Batch Processing

```bash
# Create a URL list file
cat > urls.txt << EOF
https://example.com/article-1
https://example.com/article-2
https://example.com/article-3
EOF

# Process with default settings
web-analyzer --batch urls.txt

# Parallel processing with 8 workers
web-analyzer --batch urls.txt --workers 8

# Save to specific file
web-analyzer --batch urls.txt --output results.json

# Save to directory with auto-timestamp
web-analyzer --batch urls.txt --output-dir ./reports/ --format text

# Text format output
web-analyzer --batch urls.txt --format text --output results.txt
```

#### Python API

```python
from src.scraper import WebScraper
from src.analyzer import ContentAnalyzer
from src.readability import ReadabilityAnalyzer
from src.batch_processor import BatchProcessor

# Scrape a single URL
scraper = WebScraper()
content = scraper.scrape("https://example.com/article")
print(f"Title: {content.title}")
print(f"Words: {len(content.text.split())}")

# Analyze content
analyzer = ContentAnalyzer()
result = analyzer.analyze(
    url=content.url,
    title=content.title,
    text=content.text,
    language=content.language,
)
print(f"Sentiment: {result.sentiment_label} ({result.sentiment_score})")
print(f"Keywords: {[w for w, _ in result.keywords[:5]]}")
print(f"Summary: {result.summary}")

# Readability scores
readability = ReadabilityAnalyzer()
scores = readability.analyze(content.text)
print(f"Flesch Reading Ease: {scores.flesch_reading_ease}")
print(f"Grade Level: {scores.grade_level}")

# Batch processing
processor = BatchProcessor()
results = processor.process_parallel([
    "https://example.com/1",
    "https://example.com/2",
    "https://example.com/3",
], max_workers=4)

# Save results
processor.save_results(results, "output.json", format_type="json")
```

### Sample JSON Output

```json
{
  "url": "https://example.com/article",
  "title": "Python Programming Guide",
  "language": "en",
  "statistics": {
    "word_count": 1250,
    "sentence_count": 68,
    "avg_word_length": 5.2,
    "avg_sentence_length": 18.4
  },
  "sentiment": {
    "score": 0.35,
    "label": "positive",
    "confidence": 0.85
  },
  "keywords": [
    {"word": "python", "score": 0.082},
    {"word": "programming", "score": 0.061},
    {"word": "language", "score": 0.045}
  ],
  "bigrams": [
    {"phrase": "python programming", "score": 0.015},
    {"phrase": "machine learning", "score": 0.012}
  ],
  "readability": {
    "flesch_reading_ease": 62.5,
    "flesch_kincaid_grade": 8.3,
    "smog_index": 9.1,
    "coleman_liau_index": 8.7,
    "automated_readability_index": 7.9,
    "gunning_fog_index": 10.2,
    "estimated_grade": "8th Grade"
  },
  "summary": "Python is a versatile programming language...",
  "processing_time_ms": 145.2
}
```

---

## Architecture

The project follows a modular, pipeline-based architecture:

```
CLI/Batch Processor --> Web Scraper --> Content Analyzer --> Output
                              |              |
                              v              v
                       requests + BS4   SentimentAnalyzer
                                     KeywordExtractor
                                     TextSummarizer
                                     ReadabilityAnalyzer
```

See [docs/architecture.md](docs/architecture.md) for detailed architecture documentation.

### Module Overview

| Module | File | Description |
|--------|------|-------------|
| Config | `src/config.py` | Environment-based configuration |
| Scraper | `src/scraper.py` | HTTP fetching and HTML extraction |
| Analyzer | `src/analyzer.py` | NLP analysis (sentiment, keywords, summary) |
| Readability | `src/readability.py` | Readability scoring formulas |
| CLI | `src/cli.py` | argparse-based command-line interface |
| Batch | `src/batch_processor.py` | Multi-URL processing pipeline |

---

## Configuration

Configuration can be provided via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_ANALYZER_TIMEOUT` | 30 | HTTP request timeout (seconds) |
| `WEB_ANALYZER_MAX_RETRIES` | 3 | Maximum retry attempts |
| `WEB_ANALYZER_LOG_LEVEL` | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `WEB_ANALYZER_OUTPUT_DIR` | data | Default output directory |

---

## Testing

### Run All Tests

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_analyzer.py

# Run with verbose output
pytest -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| scraper.py | test_scraper.py | URL validation, content extraction, error handling, retries |
| analyzer.py | test_analyzer.py | Sentiment, keywords, summarization, statistics |
| readability.py | test_readability.py | Syllable counting, all formulas, grade mapping |
| cli.py | test_cli.py | Argument parsing, batch mode, output formatting |

---

## Screenshots

> **Note**: Screenshots below are representative examples of the tool's output.

### Single URL - Text Output

```
======================================================================
  INTELLIGENT WEB CONTENT ANALYZER - RESULTS
======================================================================

  URL:      https://example.com/python-article
  Title:    Introduction to Python Programming

----------------------------------------------------------------------
  TEXT STATISTICS
----------------------------------------------------------------------
    Word Count:            1,245
    Sentence Count:        67
    Avg Word Length:       5.2 characters
    Avg Sentence Length:   18.6 words

----------------------------------------------------------------------
  SENTIMENT ANALYSIS
----------------------------------------------------------------------
    Score:       +0.3250
    Label:       POSITIVE
    Confidence:  80.00%

----------------------------------------------------------------------
  TOP KEYWORDS
----------------------------------------------------------------------
     1. python                 0.0852 ################
     2. programming            0.0613 ###########
     3. language               0.0447 ########
     4. development            0.0381 #######
     5. code                   0.0315 ######

----------------------------------------------------------------------
  READABILITY SCORES
----------------------------------------------------------------------
    Flesch Reading Ease:        62.45
    Flesch-Kincaid Grade:       8.31
    SMOG Index:                 9.12
    Coleman-Liau Index:         8.72
    Gunning Fog Index:          10.25
    Estimated Grade Level:      8th Grade

----------------------------------------------------------------------
  SUMMARY
----------------------------------------------------------------------

    Python is a high-level programming language known for its
    readability and versatility. It supports multiple paradigms
    and has a vast ecosystem of libraries.

  Processing time: 142.35 ms

======================================================================
```

### Batch Processing - JSON Output

```bash
$ web-analyzer --batch urls.txt --format json --output results.json
Processing 3 URLs...
  [1/3] https://example.com/article-1...
  [2/3] https://example.com/article-2...
  [3/3] https://example.com/article-3...

Completed: 3/3 succeeded

Results saved to: results.json
```

---

## Future Improvements

- [ ] **Named Entity Recognition (NER)**: Extract people, organizations, locations
- [ ] **Topic Classification**: Categorize articles into predefined topics
- [ ] **Language Detection**: Automatic language identification for multi-language support
- [ ] **spaCy Integration**: Optional spaCy backend for advanced NLP
- [ ] **Caching Layer**: Redis-based caching for analyzed URLs
- [ ] **Web Dashboard**: Flask/FastAPI web interface with visualizations
- [ ] **Export Formats**: CSV, Excel, PDF report generation
- [ ] **Rate Limiting**: Respect robots.txt and implement politeness delays
- [ ] **RSS Feed Support**: Process articles from RSS/Atom feeds
- [ ] **Docker Container**: Containerized deployment

---

## Project Structure

```
project_02_web_content_analyzer/
├── src/
│   ├── __init__.py
│   ├── scraper.py             # Web scraping logic
│   ├── analyzer.py            # NLP analysis (sentiment, keywords, summary)
│   ├── readability.py         # Readability scoring
│   ├── cli.py                 # Command-line interface
│   ├── batch_processor.py     # Process multiple URLs
│   └── config.py              # Configuration management
├── tests/
│   ├── __init__.py
│   ├── test_scraper.py        # Scraper unit tests
│   ├── test_analyzer.py       # Analyzer unit tests
│   ├── test_readability.py    # Readability unit tests
│   └── test_cli.py            # CLI unit tests
├── data/                      # Sample output files
├── docs/
│   └── architecture.md        # Architecture documentation
├── requirements.txt           # Production dependencies
├── pyproject.toml             # Project configuration
├── setup.py                   # Package setup
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore patterns
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Format code (`black src/ tests/`)
5. Commit changes (`git commit -m 'feat: Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with Python and curiosity.
</p>
