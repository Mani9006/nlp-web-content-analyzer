# Architecture Documentation

## Intelligent Web Content Analyzer - System Architecture

## Overview

The Intelligent Web Content Analyzer is a modular Python application that fetches web articles, extracts their textual content, and performs comprehensive NLP analysis including sentiment scoring, keyword extraction, text summarization, and readability assessment.

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Composability**: Components can be combined flexibly
3. **Testability**: Heavy use of dependency injection and mocking
4. **Configurability**: Environment-driven configuration
5. **Error Resilience**: Graceful handling of network and parsing errors

## System Architecture

```
                    +---------------------+
                    |      CLI Layer      |
                    |     (src/cli.py)    |
                    +----------+----------+
                               |
                    +----------v----------+
                    |  Batch Processor    |
                    |(src/batch_processor)|
                    +----------+----------+
                               |
              +----------------v----------------+
              |                                   |
     +--------v--------+              +----------v----------+
     |   Web Scraper   |              |  Content Analyzer   |
     | (src/scraper.py)|              |(src/analyzer.py)    |
     +--------+--------+              +----------+----------+
              |                                   |
     +--------v--------+              +----------v----------+
     |  HTTP/requests  |              |  SentimentAnalyzer  |
     |  BeautifulSoup  |              |  KeywordExtractor   |
     |                 |              |  TextSummarizer     |
     +-----------------+              +----------+----------+
                                                  |
                                       +----------v----------+
                                       | ReadabilityAnalyzer |
                                       |(src/readability.py) |
                                       +---------------------+
```

## Module Descriptions

### 1. Configuration (`src/config.py`)

**Responsibility**: Centralized configuration management.

- `Config` dataclass holds all settings
- `get_config()` reads from environment variables
- `configure_logging()` sets up structured logging
- Supports `WEB_ANALYZER_*` environment variables

**Key Classes**:
- `Config`: Immutable configuration container

### 2. Web Scraper (`src/scraper.py`)

**Responsibility**: Fetch web pages and extract article content.

**Process**:
1. Validate URL format
2. Fetch with retry logic (configurable attempts)
3. Parse HTML with BeautifulSoup
4. Extract metadata (title, author, date, language)
5. Remove unwanted elements (scripts, navigation, ads)
6. Extract main content using semantic HTML5 tags
7. Clean and normalize whitespace

**Key Classes**:
- `ScrapedContent`: Data container for scraped results
- `WebScraper`: Main scraper with session management

**Error Handling**:
- Invalid URLs raise `ValueError`
- HTTP errors use exponential backoff retry
- Malformed HTML is parsed leniently

### 3. Content Analyzer (`src/analyzer.py`)

**Responsibility**: Perform NLP analysis on extracted text.

**Sub-components**:

#### 3.1 Sentiment Analyzer
- Lexicon-based approach with 100+ positive and 100+ negative words
- Supports negation detection ("not good" vs "good")
- Supports intensity modifiers ("very good" vs "good")
- Returns score (-1 to +1), label, and confidence

#### 3.2 Keyword Extractor
- TF (term frequency) scoring with position boosting
- Stop word filtering
- Configurable minimum keyword length
- Bigram extraction for phrase identification

#### 3.3 Text Summarizer
- Extractive summarization using sentence scoring
- Scores based on keyword density, position, named entities
- Preserves original sentence order
- Configurable summary length

**Key Classes**:
- `AnalysisResult`: Complete analysis container
- `SentimentAnalyzer`: Lexicon-based sentiment scoring
- `KeywordExtractor`: TF-based keyword and bigram extraction
- `TextSummarizer`: Extractive summary generation
- `ContentAnalyzer`: Orchestrates all sub-components

### 4. Readability Analyzer (`src/readability.py`)

**Responsibility**: Compute standard readability metrics.

**Formulas Implemented**:
1. **Flesch Reading Ease**: 206.835 - (1.015 x ASL) - (84.6 x ASW)
2. **Flesch-Kincaid Grade**: (0.39 x ASL) + (11.8 x ASW) - 15.59
3. **SMOG Index**: sqrt(polysyllables x 30/sentences) + 3
4. **Coleman-Liau Index**: (0.0588 x L) - (0.296 x S) - 15.8
5. **Automated Readability Index**: (4.71 x chars/words) + (0.5 x ASL) - 21.43
6. **Gunning Fog Index**: 0.4 x (ASL + % complex words)

**Key Classes**:
- `ReadabilityScores`: Container for all scores
- `ReadabilityAnalyzer`: Computes all metrics with syllable estimation

### 5. Batch Processor (`src/batch_processor.py`)

**Responsibility**: Process multiple URLs efficiently.

**Features**:
- Sequential and parallel processing modes
- ThreadPoolExecutor for parallel execution
- Progress callbacks for real-time status
- Result persistence (JSON and text formats)
- Error isolation (one failure doesn't stop batch)

**Key Classes**:
- `BatchProcessor`: Orchestrates scrape + analyze pipeline

### 6. CLI (`src/cli.py`)

**Responsibility**: User-facing command-line interface.

**Features**:
- Single URL analysis
- Batch file processing
- JSON and text output formats
- Configurable keyword count, workers, timeout
- Verbose/quiet logging modes

**Entry Point**: `web-analyzer` console script

## Data Flow

```
User Input (URL or Batch File)
    |
    v
[CLI Parser] -- validates args --> [Config Loader]
    |                                    |
    v                                    v
[BatchProcessor] <-------------> [WebScraper]
    |                                   |
    |                                   v
    |                           [HTTP Request]
    |                                   |
    |                                   v
    |                           [BeautifulSoup Parser]
    |                                   |
    |                                   v
    |                           [Content Extraction]
    |                                   |
    v                                   v
[ContentAnalyzer] <--------- [ScrapedContent]
    |
    +---> [SentimentAnalyzer] --> sentiment scores
    +---> [KeywordExtractor] --> keywords & bigrams
    +---> [TextSummarizer] --> summary
    |
    v
[ReadabilityAnalyzer] --> readability scores
    |
    v
[AnalysisResult] --> [Output Formatter] --> stdout/file
```

## Extension Points

### Adding a New NLP Analyzer

1. Create a new class in `src/analyzer.py` or a new module
2. Implement the analysis method
3. Add results to the `AnalysisResult` dataclass
4. Integrate into `ContentAnalyzer.analyze()`

### Adding a New Output Format

1. Add format option to CLI parser
2. Implement formatter function in `src/batch_processor.py`
3. Update `BatchProcessor.save_results()` dispatch

### Adding a New Readability Formula

1. Implement formula in `ReadabilityAnalyzer.analyze()`
2. Add score field to `ReadabilityScores`
3. Update `AnalysisResult` serialization

## Testing Strategy

```
+------------------+--------------------------+----------------+
| Test Type        | Tools                    | Coverage Target|
+------------------+--------------------------+----------------+
| Unit Tests       | unittest, MagicMock      | >90%           |
| Integration      | unittest                 | Key paths       |
| HTTP Mocking     | unittest.mock.patch      | All requests    |
| CLI Testing      | argparse, unittest       | All commands    |
+------------------+--------------------------+----------------+
```

### Mock Strategy

- `requests.Session.get` is patched in scraper tests
- `BatchProcessor` is patched in CLI tests
- No actual network calls in test suite

## Configuration Hierarchy

1. **Defaults**: Hardcoded in `Config` dataclass
2. **Environment Variables**: `WEB_ANALYZER_*` prefixed variables
3. **CLI Arguments**: Override all previous levels

## Dependencies

```
requests          HTTP client with session management
beautifulsoup4    HTML parsing and content extraction
lxml              Fast HTML parser backend (used by BS4)
pytest            Testing framework
responses         HTTP request mocking
```
