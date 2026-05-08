"""Readability scoring module implementing standard readability formulas."""

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Optional

from src.config import Config, get_config

logger = logging.getLogger("web_content_analyzer")


@dataclass
class ReadabilityScores:
    """Container for all readability scores."""

    flesch_reading_ease: float = 0.0
    flesch_kincaid_grade: float = 0.0
    smog_index: float = 0.0
    coleman_liau_index: float = 0.0
    automated_readability_index: float = 0.0
    gunning_fog_index: float = 0.0
    avg_sentence_length: float = 0.0
    avg_syllables_per_word: float = 0.0
    complex_word_percentage: float = 0.0
    grade_level: str = ""
    reading_level: str = ""


class ReadabilityAnalyzer:
    """Analyzer for computing text readability scores."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialize the readability analyzer.

        Args:
            config: Application configuration.
        """
        self.config = config or get_config()

    @staticmethod
    def count_syllables(word: str) -> int:
        """Estimate the number of syllables in a word.

        Uses a heuristic based on vowel groups. This is a simplified
        approximation that works well for English text.

        Args:
            word: The word to analyze.

        Returns:
            int: Estimated syllable count (minimum 1).
        """
        word = word.lower().strip()
        if not word:
            return 0

        # Remove trailing 'e' (silent in many words)
        if word.endswith("e") and len(word) > 2:
            word = word[:-1]

        # Count vowel groups
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel

        # Every word has at least one syllable
        return max(1, syllable_count)

    @staticmethod
    def count_sentences(text: str) -> int:
        """Count sentences in text.

        Args:
            text: Input text.

        Returns:
            int: Number of sentences.
        """
        sentences = re.split(r'[.!?]+', text)
        return max(1, len([s for s in sentences if s.strip()]))

    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text.

        Args:
            text: Input text.

        Returns:
            int: Number of words.
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return max(1, len(words))

    @staticmethod
    def count_characters(text: str) -> int:
        """Count alphabetic characters in text.

        Args:
            text: Input text.

        Returns:
            int: Number of alphabetic characters.
        """
        return len(re.findall(r'[a-zA-Z]', text))

    def _get_text_stats(self, text: str) -> Dict[str, float]:
        """Compute basic text statistics needed for readability formulas.

        Args:
            text: Input text.

        Returns:
            Dictionary of text statistics.
        """
        sentences = self.count_sentences(text)
        words_list = re.findall(r'\b[a-zA-Z]+\b', text)
        words = max(1, len(words_list))
        characters = self.count_characters(text)

        total_syllables = sum(self.count_syllables(w) for w in words_list)
        avg_syllables_per_word = total_syllables / words

        # Complex words: 3+ syllables (for SMOG and Gunning Fog)
        complex_words = sum(
            1 for w in words_list if self.count_syllables(w) >= 3
        )

        # Count sentences with 30+ words (for SMOG)
        sentence_list = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        long_sentences = sum(1 for s in sentence_list if len(s.split()) >= 30)
        long_sentences = max(1, long_sentences)

        return {
            "sentences": sentences,
            "words": words,
            "characters": characters,
            "total_syllables": total_syllables,
            "avg_syllables_per_word": avg_syllables_per_word,
            "complex_words": complex_words,
            "complex_word_percentage": (complex_words / words) * 100,
            "avg_sentence_length": words / sentences,
            "long_sentences": long_sentences,
        }

    def analyze(self, text: str) -> ReadabilityScores:
        """Compute all readability scores for the given text.

        Args:
            text: Input text to analyze.

        Returns:
            ReadabilityScores: Container with all computed scores.
        """
        stats = self._get_text_stats(text)

        words = stats["words"]
        sentences = stats["sentences"]
        characters = stats["characters"]
        total_syllables = stats["total_syllables"]
        avg_syllables_per_word = stats["avg_syllables_per_word"]
        avg_sentence_length = stats["avg_sentence_length"]
        complex_words = stats["complex_words"]

        # Flesch Reading Ease
        # 206.835 - (1.015 * ASL) - (84.6 * ASW)
        flesch_reading_ease = (
            206.835
            - (1.015 * avg_sentence_length)
            - (84.6 * avg_syllables_per_word)
        )

        # Flesch-Kincaid Grade Level
        # (0.39 * ASL) + (11.8 * ASW) - 15.59
        flesch_kincaid_grade = (
            (0.39 * avg_sentence_length)
            + (11.8 * avg_syllables_per_word)
            - 15.59
        )

        # SMOG Index
        # sqrt(polysyllables * (30 / sentences)) + 3
        smog_index = (
            math.sqrt(complex_words * (30.0 / sentences))
            + 3.0
            if sentences > 0 else 0.0
        )

        # Coleman-Liau Index
        # (0.0588 * L) - (0.296 * S) - 15.8
        # L = letters per 100 words, S = sentences per 100 words
        letters_per_100 = (characters / words) * 100
        sentences_per_100 = (sentences / words) * 100
        coleman_liau = (
            (0.0588 * letters_per_100)
            - (0.296 * sentences_per_100)
            - 15.8
        )

        # Automated Readability Index (ARI)
        # (4.71 * chars/words) + (0.5 * words/sentences) - 21.43
        ari = (
            (4.71 * (characters / words))
            + (0.5 * avg_sentence_length)
            - 21.43
        )

        # Gunning Fog Index
        # 0.4 * (ASL + percentage of complex words)
        gunning_fog = 0.4 * (
            avg_sentence_length + stats["complex_word_percentage"]
        )

        # Determine grade level
        grade_level = self._grade_level(flesch_kincaid_grade)
        reading_level = self._reading_level(flesch_reading_ease)

        scores = ReadabilityScores(
            flesch_reading_ease=round(flesch_reading_ease, 2),
            flesch_kincaid_grade=round(max(0, flesch_kincaid_grade), 2),
            smog_index=round(max(0, smog_index), 2),
            coleman_liau_index=round(max(0, coleman_liau), 2),
            automated_readability_index=round(max(0, ari), 2),
            gunning_fog_index=round(max(0, gunning_fog), 2),
            avg_sentence_length=round(avg_sentence_length, 2),
            avg_syllables_per_word=round(avg_syllables_per_word, 2),
            complex_word_percentage=round(stats["complex_word_percentage"], 2),
            grade_level=grade_level,
            reading_level=reading_level,
        )

        logger.debug(
            "Readability scores - Flesch: %.1f, Grade: %s, Level: %s",
            scores.flesch_reading_ease,
            scores.grade_level,
            scores.reading_level,
        )
        return scores

    @staticmethod
    def _grade_level(flesch_kincaid: float) -> str:
        """Map Flesch-Kincaid grade score to a readable grade level.

        Args:
            flesch_kincaid: Flesch-Kincaid grade score.

        Returns:
            Human-readable grade level string.
        """
        grade = max(0, flesch_kincaid)
        if grade <= 1:
            return "1st Grade"
        elif grade <= 2:
            return "2nd Grade"
        elif grade <= 3:
            return "3rd Grade"
        elif grade <= 4:
            return "4th Grade"
        elif grade <= 5:
            return "5th Grade"
        elif grade <= 6:
            return "6th Grade"
        elif grade <= 7:
            return "7th Grade"
        elif grade <= 8:
            return "8th Grade"
        elif grade <= 9:
            return "9th Grade"
        elif grade <= 10:
            return "10th Grade"
        elif grade <= 11:
            return "11th Grade"
        elif grade <= 12:
            return "12th Grade"
        elif grade <= 14:
            return "Undergraduate"
        elif grade <= 16:
            return "Graduate"
        else:
            return "Professional"

    @staticmethod
    def _reading_level(flesch_score: float) -> str:
        """Map Flesch Reading Ease score to a readable level.

        Args:
            flesch_score: Flesch Reading Ease score.

        Returns:
            Human-readable reading level string.
        """
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        elif flesch_score >= 0:
            return "Very Difficult"
        else:
            return "Extremely Difficult"
