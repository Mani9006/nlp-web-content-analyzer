"""NLP analysis module for sentiment, keyword extraction, and summarization."""

import logging
import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config import Config, get_config

logger = logging.getLogger("web_content_analyzer")


@dataclass
class AnalysisResult:
    """Container for comprehensive NLP analysis results."""

    url: str
    title: str = ""
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0

    # Sentiment
    sentiment_score: float = 0.0  # Range: -1.0 (negative) to 1.0 (positive)
    sentiment_label: str = "neutral"  # "positive", "negative", or "neutral"
    sentiment_confidence: float = 0.0

    # Keywords
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    bigrams: List[Tuple[str, float]] = field(default_factory=list)

    # Summary
    summary: str = ""

    # Readability
    readability_scores: Dict[str, float] = field(default_factory=dict)
    readability_grade: str = ""

    # Meta
    processing_time_ms: float = 0.0
    language: Optional[str] = None


class SentimentAnalyzer:
    """Lexicon-based sentiment analyzer."""

    # Positive and negative word lists (simplified VADER-style approach)
    POSITIVE_WORDS = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "love", "happy", "best", "beautiful", "awesome", "perfect",
        "brilliant", "outstanding", "superb", "remarkable", "impressive",
        "exciting", "pleased", "delighted", "glad", "positive", "success",
        "effective", "beneficial", "valuable", "important", "useful",
        "helpful", "easy", "clear", "strong", "better", "improved",
        "innovative", "creative", "smart", "efficient", "reliable",
        "robust", "powerful", "flexible", "simple", "elegant", "nice",
        "enjoyable", "satisfying", "rewarding", "encouraging", "promising",
        "hopeful", "optimistic", "confident", "proud", "grateful",
        "fortunate", "lucky", "thrilled", "enthusiastic", "passionate",
        "inspiring", "motivating", "uplifting", "refreshing", "pleasant",
        "charming", "lovely", "gorgeous", "stunning", "magnificent",
        "extraordinary", "exceptional", "phenomenal", "terrific",
        "marvelous", "fabulous", "incredible", "unbelievable", "spectacular",
    }

    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "horrible", "worst", "hate", "sad",
        "angry", "poor", "disappointing", "frustrating", "annoying",
        "boring", "difficult", "hard", "complicated", "confusing",
        "useless", "waste", "fail", "failure", "problem", "issue",
        "error", "bug", "broken", "slow", "outdated", "obsolete",
        "expensive", "costly", "risky", "dangerous", "harmful", "wrong",
        "false", "misleading", "unfair", "biased", "limited", "restricted",
        "weak", "unstable", "unreliable", "insufficient", "inadequate",
        "unsatisfactory", "unacceptable", "disagreeable", "unpleasant",
        "ugly", "messy", "chaotic", "stressful", "worrying", "concerning",
        "alarming", "threatening", "damaging", "destructive", "devastating",
        "tragic", "unfortunate", "regrettable", "upsetting", "disturbing",
        "discouraging", "disheartening", "depressing", "bleak", "grim",
        "serious", "severe", "critical", "urgent", "troubling", "painful",
        "uncomfortable", "awkward", "embarrassing", "shameful",
        "disgraceful", "scandalous", "outrageous", "ridiculous",
        "absurd", "pathetic", "hopeless", "helpless", "desperate",
    }

    INTENSIFIERS = {
        "very": 1.5, "extremely": 2.0, "incredibly": 2.0, "absolutely": 1.8,
        "completely": 1.6, "totally": 1.6, "really": 1.4, "quite": 1.3,
        "pretty": 1.2, "fairly": 1.1, "rather": 1.15, "so": 1.4,
        "too": 1.3, "highly": 1.5, "deeply": 1.4, "strongly": 1.5,
        "especially": 1.3, "particularly": 1.3, "remarkably": 1.6,
        "exceptionally": 1.7, "extraordinarily": 1.8,
    }

    NEGATORS = {
        "not", "no", "never", "neither", "nor", "none", "nobody",
        "nothing", "nowhere", "hardly", "scarcely", "barely",
        "doesn't", "isn't", "wasn't", "shouldn't", "wouldn't",
        "couldn't", "can't", "won't", "don't", "didn't", "hasn't",
        "haven't", "hadn't", "aren't", "weren't",
    }

    def analyze(self, text: str) -> Tuple[float, str, float]:
        """Analyze sentiment of the given text.

        Args:
            text: Input text to analyze.

        Returns:
            Tuple of (sentiment_score, label, confidence).
            Score ranges from -1.0 (negative) to 1.0 (positive).
        """
        words = self._tokenize(text.lower())
        if not words:
            return 0.0, "neutral", 0.0

        score = 0.0
        positive_count = 0
        negative_count = 0
        i = 0

        while i < len(words):
            word = words[i]
            multiplier = 1.0

            # Check for negation in the previous 3 words
            negated = any(
                w in self.NEGATORS for w in words[max(0, i - 3):i]
            )

            # Check for intensifier in the previous 2 words
            for j in range(max(0, i - 2), i):
                if words[j] in self.INTENSIFIERS:
                    multiplier = self.INTENSIFIERS[words[j]]
                    break

            if word in self.POSITIVE_WORDS:
                word_score = 1.0 * multiplier
                if negated:
                    word_score = -word_score
                score += word_score
                positive_count += 1
            elif word in self.NEGATIVE_WORDS:
                word_score = -1.0 * multiplier
                if negated:
                    word_score = -word_score
                score += word_score
                negative_count += 1

            i += 1

        # Normalize score to [-1, 1]
        total_markers = positive_count + negative_count
        if total_markers == 0:
            return 0.0, "neutral", 0.0

        normalized_score = max(-1.0, min(1.0, score / (total_markers * 2)))

        # Determine label and confidence
        if normalized_score > 0.1:
            label = "positive"
            confidence = min(1.0, abs(normalized_score) * 2)
        elif normalized_score < -0.1:
            label = "negative"
            confidence = min(1.0, abs(normalized_score) * 2)
        else:
            label = "neutral"
            confidence = 1.0 - abs(normalized_score) * 3

        return round(normalized_score, 4), label, round(confidence, 4)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple word tokenizer.

        Args:
            text: Text to tokenize.

        Returns:
            List of word tokens.
        """
        # Remove punctuation except apostrophes, then split
        text = re.sub(r"[^\w\s']", " ", text)
        return text.split()


class KeywordExtractor:
    """Extract keywords and bigrams from text using TF-IDF-style scoring."""

    # Common English stop words
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are",
        "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may",
        "might", "must", "shall", "can", "need", "dare", "ought",
        "used", "it", "its", "this", "that", "these", "those",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him",
        "his", "himself", "she", "her", "hers", "herself", "they",
        "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "whose", "where", "when", "why", "how",
        "all", "any", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "also", "then", "here",
        "there", "up", "down", "out", "off", "over", "under", "again",
        "further", "once", "during", "before", "after", "above",
        "below", "between", "through", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "don", "now", "said", "says", "say", "get", "go",
        "make", "see", "know", "take", "come", "think", "look",
        "want", "give", "use", "find", "tell", "ask", "work", "seem",
        "feel", "try", "leave", "call", "good", "new", "first", "last",
        "long", "great", "little", "own", "other", "old", "right",
        "big", "high", "different", "small", "large", "next", "early",
        "young", "important", "few", "public", "bad", "same", "able",
    }

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the keyword extractor.

        Args:
            config: Application configuration.
        """
        self.config = config or get_config()

    def extract_keywords(self, text: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """Extract top keywords from text using TF-like scoring.

        Args:
            text: Input text.
            top_n: Number of top keywords to return. Uses config default if None.

        Returns:
            List of (keyword, score) tuples sorted by score descending.
        """
        if top_n is None:
            top_n = self.config.top_keywords

        words = self._tokenize(text.lower())
        words = [
            w for w in words
            if w not in self.STOP_WORDS
            and len(w) >= self.config.min_keyword_length
            and w.isalpha()
        ]

        if not words:
            return []

        # Term frequency with position boosting
        word_counts = Counter(words)
        total_words = len(words)

        # Score = tf * (1 + log(position_factor))
        scores = {}
        for word, count in word_counts.most_common(top_n * 3):
            tf = count / total_words
            # Boost words that appear early in the document
            first_positions = [
                i for i, w in enumerate(words) if w == word
            ]
            position_boost = 1.0
            if first_positions:
                position_boost = 1.0 + math.log1p(
                    len(words) / (first_positions[0] + 1)
                ) * 0.3

            scores[word] = round(tf * position_boost, 6)

        # Return sorted by score
        sorted_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]

    def extract_bigrams(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """Extract top bigrams (two-word phrases) from text.

        Args:
            text: Input text.
            top_n: Number of top bigrams to return.

        Returns:
            List of (bigram, score) tuples sorted by score descending.
        """
        words = self._tokenize(text.lower())
        words = [
            w for w in words
            if w not in self.STOP_WORDS and w.isalpha()
        ]

        if len(words) < 2:
            return []

        bigrams = []
        for i in range(len(words) - 1):
            # Filter out bigrams with stop words or short words
            w1, w2 = words[i], words[i + 1]
            if len(w1) >= self.config.min_keyword_length and len(w2) >= self.config.min_keyword_length:
                bigrams.append(f"{w1} {w2}")

        if not bigrams:
            return []

        bigram_counts = Counter(bigrams)
        total = len(bigrams)

        sorted_bigrams = sorted(
            ((bg, round(count / total, 6)) for bg, count in bigram_counts.most_common(top_n * 2)),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_bigrams[:top_n]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize.

        Returns:
            List of word tokens.
        """
        text = re.sub(r"[^\w\s]", " ", text.lower())
        return text.split()


class TextSummarizer:
    """Extractive text summarizer using sentence scoring."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the summarizer.

        Args:
            config: Application configuration.
        """
        self.config = config or get_config()

    def summarize(self, text: str, num_sentences: Optional[int] = None) -> str:
        """Generate an extractive summary of the text.

        Scores sentences based on:
        - Keyword frequency within the sentence
        - Position in document (earlier = higher)
        - Presence of named-like entities (capitalized words)
        - Length normalization

        Args:
            text: Input text to summarize.
            num_sentences: Number of sentences in summary. Uses config default if None.

        Returns:
            Summary string.
        """
        if num_sentences is None:
            num_sentences = self.config.summary_sentences

        sentences = self._split_sentences(text)
        if len(sentences) <= num_sentences:
            return text.strip()

        if not sentences:
            return ""

        # Build word frequency map (excluding stop words)
        stop_words = KeywordExtractor.STOP_WORDS
        word_freq = Counter(
            w for s in sentences for w in s.lower().split()
            if w not in stop_words and w.isalpha() and len(w) > 2
        )

        # Score each sentence
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            score = self._score_sentence(sentence, idx, len(sentences), word_freq)
            scored_sentences.append((score, idx, sentence))

        # Select top sentences, preserving original order
        top_sentences = sorted(
            scored_sentences,
            key=lambda x: x[0],
            reverse=True,
        )[:num_sentences]
        top_sentences.sort(key=lambda x: x[1])  # Re-sort by original position

        summary = " ".join(s for _, _, s in top_sentences)
        return summary.strip()

    def _score_sentence(
        self,
        sentence: str,
        position: int,
        total: int,
        word_freq: Counter,
    ) -> float:
        """Score a single sentence for importance.

        Args:
            sentence: The sentence text.
            position: Index of the sentence in the document.
            total: Total number of sentences.
            word_freq: Global word frequency counter.

        Returns:
            Importance score.
        """
        words = sentence.lower().split()
        if not words:
            return 0.0

        # Keyword frequency score
        keyword_score = sum(
            word_freq.get(w, 0) for w in words if w.isalpha()
        ) / len(words)

        # Position bonus (earlier sentences are more important)
        position_score = 1.0 - (position / total) * 0.5

        # Named entity bonus (capitalized words in original)
        original_words = sentence.split()
        named_entities = sum(
            1 for w in original_words
            if w[0].isupper() and w.isalpha() and len(w) > 2
        )
        entity_score = named_entities / len(original_words) if original_words else 0

        # Length normalization (prefer medium-length sentences)
        length = len(words)
        length_score = 1.0
        if length < 5:
            length_score = 0.5
        elif length > 40:
            length_score = 0.7

        total_score = (
            keyword_score * 2.0
            + position_score * 1.5
            + entity_score * 1.0
        ) * length_score

        return total_score

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Input text.

        Returns:
            List of sentence strings.
        """
        # Simple sentence splitting on punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]


class ContentAnalyzer:
    """Main content analyzer combining all NLP components."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the content analyzer.

        Args:
            config: Application configuration.
        """
        self.config = config or get_config()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.keyword_extractor = KeywordExtractor(config)
        self.text_summarizer = TextSummarizer(config)

    def analyze(
        self,
        url: str,
        title: str,
        text: str,
        language: Optional[str] = None,
    ) -> AnalysisResult:
        """Perform full NLP analysis on extracted content.

        Args:
            url: Source URL.
            title: Article title.
            text: Article body text.
            language: Detected language code.

        Returns:
            AnalysisResult: Complete analysis results.
        """
        import time
        start_time = time.perf_counter()

        # Basic text statistics
        word_count = len(text.split())
        sentence_count = len(re.split(r'(?<=[.!?])\s+', text))
        avg_word_length = (
            sum(len(w) for w in text.split()) / word_count
            if word_count > 0 else 0.0
        )
        avg_sentence_length = (
            word_count / sentence_count if sentence_count > 0 else 0.0
        )

        # Sentiment analysis
        sentiment_score, sentiment_label, sentiment_confidence = (
            self.sentiment_analyzer.analyze(text)
        )

        # Keyword extraction
        keywords = self.keyword_extractor.extract_keywords(text)
        bigrams = self.keyword_extractor.extract_bigrams(text)

        # Summarization
        summary = self.text_summarizer.summarize(text)

        processing_time = (time.perf_counter() - start_time) * 1000

        result = AnalysisResult(
            url=url,
            title=title,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_word_length=round(avg_word_length, 2),
            avg_sentence_length=round(avg_sentence_length, 2),
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            sentiment_confidence=sentiment_confidence,
            keywords=keywords,
            bigrams=bigrams,
            summary=summary,
            processing_time_ms=round(processing_time, 2),
            language=language,
        )

        logger.info(
            "Analyzed %s - %d words, %d sentences, sentiment=%s (%.2f)",
            url,
            word_count,
            sentence_count,
            sentiment_label,
            sentiment_score,
        )
        return result
