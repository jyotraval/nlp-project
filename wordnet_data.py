"""Turn WordNet into a list of searchable "documents".

One document = one WordNet synset = one *meaning*. It stores
  - words:  the words that can express that meaning, most common first
  - pos:    part of speech (noun / verb / adjective / adverb)
  - gloss:  the dictionary definition
  - tokens: pre-processed text that the retrievers actually search over

Searching meanings (not words) keeps the index small, and we convert back to
words only at the very end.
"""
import pickle
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

from text_utils import preprocess

CACHE = Path(__file__).parent / ".cache"
POS_NAMES = {"n": "noun", "v": "verb", "a": "adj", "s": "adj", "r": "adv"}


def ensure_wordnet():
    """Download WordNet once if it is missing (needs internet the first time)."""
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def is_plain_word(name):
    """Keep lowercase single words. Drops phrases ("fire_engine") and names ("Paris")."""
    return name.isalpha() and name.islower() and len(name) > 1


def build_documents(use_related=False):
    """List of dicts, one per WordNet synset that has at least one usable word.

    use_related=False -> search over the definition only.
    use_related=True  -> also add the example sentences and the parent concept's
                         words (e.g. acrophobia -> "phobia"). More text to match
                         against, but also more noise. The evaluation compares both.
    """
    ensure_wordnet()
    docs = []
    for syn in wn.all_synsets():
        lemmas = [lem for lem in syn.lemmas() if is_plain_word(lem.name())]
        if not lemmas:
            continue
        # Most frequently used word first (WordNet counts how often each sense appears
        # in a tagged corpus), so "bookworm" would beat an obscure synonym on a tie.
        lemmas.sort(key=lambda lem: (-lem.count(), lem.name()))

        text = syn.definition()
        if use_related:
            text += " " + " ".join(syn.examples())
            for parent in syn.hypernyms():
                text += " " + " ".join(lem.name().replace("_", " ") for lem in parent.lemmas())

        docs.append({
            "words": [lem.name() for lem in lemmas],
            "pos": POS_NAMES[syn.pos()],
            "gloss": syn.definition(),
            "tokens": preprocess(text),
        })
    return docs


def load_documents(use_related=False):
    """build_documents() with an on-disk cache (building takes ~30 s, loading ~1 s)."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"docs_{'related' if use_related else 'gloss'}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    docs = build_documents(use_related)
    with open(path, "wb") as f:
        pickle.dump(docs, f)
    return docs
