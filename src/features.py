"""
features.py
------------
Reusable linguistic/statistical feature extractor for the
IT41033 NIA project (AI-generated vs human-written text detection).

CPU-only. No torch, no external model downloads, no internet access
required at run time (stopword list is bundled below).

Usage:
    from features import extract_features
    feats = extract_features("Some text here.")
    # -> {'word_count': ..., 'sentence_count': ..., ...}
"""

import re
import string
from statistics import mean, pvariance

# --------------------------------------------------------------------------
# Bundled English stopword list (NLTK's standard 179-word list).
# Kept as a plain set here so this module has zero external dependencies
# and works even with no internet access on any teammate's machine.
# --------------------------------------------------------------------------
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "don't", "should", "should've", "now",
    "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't",
    "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn",
    "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma",
    "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't",
    "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't",
}

_WORD_RE = re.compile(r"[A-Za-z']+")
# Splits on ., !, ? (one or more) followed by whitespace or end of string.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PUNCT_SET = set(string.punctuation)


def _split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    # Drop empty fragments (can happen with trailing punctuation/whitespace)
    return [p.strip() for p in parts if p.strip()]


def _words_in(sentence_or_text: str):
    return _WORD_RE.findall(sentence_or_text)


def extract_features(text: str) -> dict:
    """
    Extract linguistic/statistical features from a single piece of text.

    Parameters
    ----------
    text : str
        Raw input text (one document / row).

    Returns
    -------
    dict
        {
            'word_count': int,
            'sentence_count': int,
            'avg_sentence_length': float,   # avg words per sentence
            'sentence_length_variance': float,  # burstiness
            'avg_word_length': float,       # avg chars per word
            'type_token_ratio': float,      # unique words / total words
            'punctuation_freq': float,      # punctuation marks per word
            'stopword_ratio': float,        # stopwords / total words
        }
    """
    if text is None:
        text = ""
    text = str(text)

    sentences = _split_sentences(text)
    words = _words_in(text)
    word_count = len(words)
    sentence_count = len(sentences)

    # --- avg sentence length & burstiness (variance of sentence lengths) ---
    if sentence_count > 0:
        sentence_lengths = [len(_words_in(s)) for s in sentences]
        avg_sentence_length = mean(sentence_lengths)
        # population variance; 0.0 if only one sentence
        sentence_length_variance = (
            pvariance(sentence_lengths) if len(sentence_lengths) > 1 else 0.0
        )
    else:
        avg_sentence_length = 0.0
        sentence_length_variance = 0.0

    # --- avg word length ---
    avg_word_length = mean(len(w) for w in words) if word_count > 0 else 0.0

    # --- type-token ratio (vocabulary diversity) ---
    if word_count > 0:
        lowered = [w.lower() for w in words]
        type_token_ratio = len(set(lowered)) / word_count
    else:
        type_token_ratio = 0.0

    # --- punctuation frequency (punctuation marks per word) ---
    punct_count = sum(1 for ch in text if ch in _PUNCT_SET)
    punctuation_freq = (punct_count / word_count) if word_count > 0 else 0.0

    # --- stopword ratio ---
    if word_count > 0:
        lowered = [w.lower() for w in words]
        stopword_count = sum(1 for w in lowered if w in STOPWORDS)
        stopword_ratio = stopword_count / word_count
    else:
        stopword_ratio = 0.0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 4),
        "sentence_length_variance": round(sentence_length_variance, 4),
        "avg_word_length": round(avg_word_length, 4),
        "type_token_ratio": round(type_token_ratio, 4),
        "punctuation_freq": round(punctuation_freq, 4),
        "stopword_ratio": round(stopword_ratio, 4),
    }


FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "avg_sentence_length",
    "sentence_length_variance",
    "avg_word_length",
    "type_token_ratio",
    "punctuation_freq",
    "stopword_ratio",
]


if __name__ == "__main__":
    sample = (
        "This is a short test sentence. It has two sentences! "
        "Does it work correctly?"
    )
    import json
    print(json.dumps(extract_features(sample), indent=2))
