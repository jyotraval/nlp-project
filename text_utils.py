"""Text pre-processing: lowercase -> tokenize -> drop stop words -> lemmatize.

The same function is used on WordNet definitions and on the user's query, so both
sides end up in the same "vocabulary space" (this is what makes matching work).
"""
import re
from functools import lru_cache

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# sklearn's stop list also removes negations ("not", "no", "without"...).
# For a dictionary these matter ("having no money" != "having money"), so keep them.
KEEP = {"not", "no", "nor", "never", "none", "nothing", "without", "cannot"}
STOP_WORDS = ENGLISH_STOP_WORDS - KEEP

_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=None)
def lemmatize(word):
    """books -> book, loving -> love. Tries noun first, then verb.

    lru_cache: WordNet definitions repeat the same words thousands of times,
    so remembering results makes pre-processing ~10x faster.
    """
    as_noun = _lemmatizer.lemmatize(word, "n")
    if as_noun != word:
        return as_noun
    return _lemmatizer.lemmatize(word, "v")


def raw_words(text):
    """Lowercase alphabetic words only (digits and punctuation are dropped)."""
    return re.findall(r"[a-z]+", text.lower())


def preprocess(text):
    """Text -> list of lemmatized content words."""
    words = [w for w in raw_words(text) if len(w) > 1 and w not in STOP_WORDS]
    return [lemmatize(w) for w in words]
