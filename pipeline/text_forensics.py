"""
TruthLens Text Forensics Module
Analyzes textual claims, social media captions, and suspicious articles for linguistic markers:
- Lexical diversity (Type-Token Ratio)
- Sentence length variance (Burstiness)
- Repetition index & formulaic AI transition patterns
- Readability scores (Flesch-Kincaid & Reading Ease)
- Transparent probabilistic score with explicit uncertainty disclaimer
"""

import math
import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TextForensicsResult(BaseModel):
    """Linguistic and structural forensics result for written text."""
    word_count: int
    char_count: int
    sentence_count: int
    avg_sentence_length: float
    lexical_diversity: float = Field(
        description="Type-Token Ratio (0.0 - 1.0): ratio of unique words to total words"
    )
    burstiness_score: float = Field(
        description="Sentence length variance. Low burstiness (< 4.0) often indicates formulaic/synthetic writing."
    )
    repetition_index: float = Field(
        description="Frequency of repeated bigrams/trigrams (0.0 - 1.0)"
    )
    flesch_reading_ease: float
    flesch_kincaid_grade: float
    detected_patterns: List[str] = Field(default_factory=list)
    ai_likelihood_score: float = Field(
        ge=0.0, le=1.0,
        description="Probabilistic indication of AI drafting (0.0 = highly varied/human, 1.0 = highly uniform/synthetic)"
    )
    verdict: str
    disclaimer: str = (
        "Text AI detection is probabilistic and subject to significant false positives on formal human writing. "
        "Use structural metrics as supplementary signals rather than definitive proof."
    )


def _count_syllables(word: str) -> int:
    """Estimates syllable count using English phonetic vowel rules."""
    w = word.lower().strip()
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', w)
    w = re.sub(r'^y', '', w)
    syllables = len(re.findall(r'[aeiouy]{1,2}', w))
    return max(1, syllables)


def analyze_text_forensics(text: str) -> TextForensicsResult:
    """
    Performs deterministic linguistic and statistical analysis on input text.
    Returns calibrated structural metrics, readability, and explicit caveats.
    """
    cleaned = text.strip()
    if not cleaned:
        return TextForensicsResult(
            word_count=0,
            char_count=0,
            sentence_count=0,
            avg_sentence_length=0.0,
            lexical_diversity=0.0,
            burstiness_score=0.0,
            repetition_index=0.0,
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            detected_patterns=["Empty input text."],
            ai_likelihood_score=0.0,
            verdict="INSUFFICIENT TEXT"
        )

    # Tokenization
    sentences = [s.strip() for s in re.split(r'[.!?]+', cleaned) if s.strip()]
    if not sentences:
        sentences = [cleaned]

    words = re.findall(r'\b[a-zA-Z0-9_\'-]+\b', cleaned)
    word_count = len(words)
    char_count = len(cleaned)
    sentence_count = len(sentences)

    if word_count == 0:
        return TextForensicsResult(
            word_count=0,
            char_count=char_count,
            sentence_count=sentence_count,
            avg_sentence_length=0.0,
            lexical_diversity=0.0,
            burstiness_score=0.0,
            repetition_index=0.0,
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            detected_patterns=["No alphanumeric words identified."],
            ai_likelihood_score=0.0,
            verdict="INSUFFICIENT TEXT"
        )

    # Lexical Diversity (Type-Token Ratio)
    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    ttr = len(unique_words) / word_count

    # Sentence lengths and burstiness (standard deviation of words per sentence)
    sent_lengths = []
    for s in sentences:
        s_words = re.findall(r'\b[a-zA-Z0-9_\'-]+\b', s)
        if s_words:
            sent_lengths.append(len(s_words))

    if not sent_lengths:
        sent_lengths = [word_count]

    avg_sent_len = sum(sent_lengths) / len(sent_lengths)

    if len(sent_lengths) > 1:
        variance = sum((l - avg_sent_len) ** 2 for l in sent_lengths) / len(sent_lengths)
        burstiness = math.sqrt(variance)
    else:
        burstiness = 0.0

    # Repetition index (bigram repetition)
    bigrams = [f"{lower_words[i]} {lower_words[i+1]}" for i in range(len(lower_words) - 1)]
    if bigrams:
        unique_bigrams = set(bigrams)
        rep_index = 1.0 - (len(unique_bigrams) / len(bigrams))
    else:
        rep_index = 0.0

    # Syllables and Readability
    total_syllables = sum(_count_syllables(w) for w in words)
    syllables_per_word = total_syllables / word_count
    words_per_sent = word_count / max(sentence_count, 1)

    flesch_ease = round(206.835 - (1.015 * words_per_sent) - (84.6 * syllables_per_word), 2)
    flesch_kincaid = round((0.39 * words_per_sent) + (11.8 * syllables_per_word) - 15.59, 2)

    # AI Pattern Detection
    ai_cliches = [
        "in conclusion", "it is important to note", "delve into", "tapestry",
        "beacon", "crucial", "furthermore", "testament to", "game-changer",
        "multifaceted", "paramount", "it is worth noting", "seamlessly"
    ]
    detected_patterns = []
    lower_text = cleaned.lower()
    for cliche in ai_cliches:
        if cliche in lower_text:
            detected_patterns.append(f"Formulaic phrase: '{cliche}'")

    if burstiness < 3.0 and len(sent_lengths) >= 3:
        detected_patterns.append("Unusually uniform sentence length cadence (low burstiness)")
    if ttr < 0.45 and word_count >= 50:
        detected_patterns.append(f"Low lexical diversity (TTR: {ttr:.2f})")

    # Score calculation
    # Factors that increase AI likelihood:
    # 1. Low burstiness on multi-sentence text
    # 2. Formulaic AI clichés
    # 3. Very uniform paragraph structure
    ai_score = 0.15  # baseline
    if len(sent_lengths) >= 3:
        if burstiness < 3.0:
            ai_score += 0.25
        elif burstiness < 5.0:
            ai_score += 0.10
        else:
            ai_score -= 0.05

    if detected_patterns:
        ai_score += min(len(detected_patterns) * 0.12, 0.40)

    if ttr < 0.40 and word_count >= 60:
        ai_score += 0.15

    ai_score = max(0.05, min(0.95, round(ai_score, 2)))

    if word_count < 25:
        verdict = "INCONCLUSIVE (SAMPLE TOO SHORT)"
        ai_score = 0.20
    elif ai_score >= 0.70:
        verdict = "HIGHLY FORMULAIC / SUSPECTED SYNTHETIC"
    elif ai_score >= 0.40:
        verdict = "MODERATE STRUCTURAL UNIFORMITY"
    else:
        verdict = "ORGANIC LINGUISTIC VARIATION"

    return TextForensicsResult(
        word_count=word_count,
        char_count=char_count,
        sentence_count=sentence_count,
        avg_sentence_length=round(avg_sent_len, 2),
        lexical_diversity=round(ttr, 3),
        burstiness_score=round(burstiness, 2),
        repetition_index=round(rep_index, 3),
        flesch_reading_ease=flesch_ease,
        flesch_kincaid_grade=max(0.0, flesch_kincaid),
        detected_patterns=detected_patterns,
        ai_likelihood_score=ai_score,
        verdict=verdict
    )
