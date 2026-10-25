"""The four retrieval methods, written as plain functions.

    tfidf      sparse   : cosine similarity of TF-IDF vectors (word overlap)
    glove_avg  dense    : average of the GloVe vectors of the definition's words
    glove_idf  dense    : same, but rarer words count more (IDF-weighted average)
    headword   dense    : compare the query with the candidate word's OWN GloVe vector
                          (GloVe knows "herbivore" lives near "plants", "eats", "animal")
    hybrid     all      : Reciprocal Rank Fusion of tfidf + glove_idf + headword

All methods score *meanings* (WordNet synsets); `search` turns the best meanings
into a ranked list of words.
"""
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from text_utils import preprocess, raw_words
from wordnet_data import CACHE, load_documents

ROOT = Path(__file__).parent
GLOVE_FILE = "glove-wiki-gigaword-100.gz"
GLOVE_URL = ("https://github.com/RaRe-Technologies/gensim-data/releases/download/"
             "glove-wiki-gigaword-100/" + GLOVE_FILE)
METHODS = ["tfidf", "glove_avg", "glove_idf", "headword", "hybrid"]
HYBRID_PARTS = ["tfidf", "glove_idf", "headword"]   # what the hybrid fuses
DEFAULT_WEIGHTS = (0.4, 0.3, 0.3)                   # one weight per part, sums to 1
RRF_K = 60  # standard constant for Reciprocal Rank Fusion


# --------------------------------------------------------------------------- #
# GloVe word vectors
# --------------------------------------------------------------------------- #
def find_glove_file():
    """Look for the GloVe file (also where gensim keeps it), download if missing."""
    candidates = [CACHE / GLOVE_FILE,
                  Path.home() / "gensim-data" / "glove-wiki-gigaword-100" / GLOVE_FILE]
    for path in candidates:
        if path.exists():
            return path
    CACHE.mkdir(exist_ok=True)
    print("Downloading GloVe vectors (~130 MB, one time only)...")
    urllib.request.urlretrieve(GLOVE_URL, candidates[0])
    return candidates[0]


def load_glove():
    """Return {"index": word -> row number, "vectors": (n_words, 100) array}.

    Vectors are mean-centered: GloVe vectors share one big common direction, which makes
    every word look similar to every other word. Subtracting the average vector of the
    50,000 most frequent words (the file is sorted by frequency) removes it. On the dev
    queries this raised Recall@10 of the embedding methods from 0.46 to 0.49.
    """
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / "glove100_centered.npz"
    if cached.exists():
        data = np.load(cached)
        words, vectors = data["words"], data["vectors"]
    else:
        words, rows = [], []
        with gzip.open(find_glove_file(), "rt", encoding="utf8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) != 101:      # skips the optional "400000 100" header line
                    continue
                words.append(parts[0])
                rows.append(np.array(parts[1:], dtype=np.float32))
        words, vectors = np.array(words), np.vstack(rows)
        vectors = (vectors - vectors[:50000].mean(axis=0)).astype(np.float32)
        np.savez(cached, words=words, vectors=vectors)
    return {"index": {w: i for i, w in enumerate(words)}, "vectors": vectors}


# --------------------------------------------------------------------------- #
# Building the index
# --------------------------------------------------------------------------- #
def keep_tokens(tokens):
    """Documents are already tokenized; tell sklearn not to tokenize them again."""
    return tokens


def build_index(glove, use_related=False):
    """Pre-compute everything that does not depend on the query."""
    docs = load_documents(use_related)

    # 1) TF-IDF: every row is one meaning, L2-normalised, so dot product = cosine.
    tfv = TfidfVectorizer(analyzer=keep_tokens)
    tfidf_matrix = tfv.fit_transform([d["tokens"] for d in docs])

    # 2) Embedding matrix aligned with the TF-IDF vocabulary (zeros if GloVe lacks the word).
    vocab = tfv.get_feature_names_out()
    rows = np.array([glove["index"].get(w, -1) for w in vocab])
    emb = np.zeros((len(vocab), glove["vectors"].shape[1]), dtype=np.float32)
    known = rows >= 0
    emb[known] = glove["vectors"][rows[known]]

    # 3) Document vectors = weighted sum of word vectors, then L2-normalised.
    #    TF-IDF weights -> "glove_idf".  All weights = 1 -> plain average "glove_avg".
    plain = tfidf_matrix.copy()
    plain.data = np.ones_like(plain.data)
    doc_vecs = {
        "glove_idf": normalize(tfidf_matrix @ emb),
        "glove_avg": normalize(plain @ emb),
    }

    # 4) Headword vectors: mean GloVe vector of the words of each meaning (not of its definition).
    dim = glove["vectors"].shape[1]
    head = np.zeros((len(docs), dim), dtype=np.float32)
    for i, doc in enumerate(docs):
        rows_ = [glove["index"][w] for w in doc["words"] if w in glove["index"]]
        if rows_:
            head[i] = glove["vectors"][rows_].mean(axis=0)
    doc_vecs["headword"] = normalize(head)

    idf = dict(zip(vocab, tfv.idf_))
    return {"docs": docs, "tfv": tfv, "tfidf": tfidf_matrix, "doc_vecs": doc_vecs,
            "idf": idf, "max_idf": float(tfv.idf_.max()), "glove": glove, "memo": {}}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def embed_query(index, tokens, weighted):
    """Query vector = (IDF-weighted) sum of the GloVe vectors of its words.

    Words that never occur in any definition get the maximum IDF: they are rare,
    so they should count a lot. GloVe still knows them, which is exactly why
    embeddings can match "pediatrician" even if the query says "children's doctor".
    """
    glove = index["glove"]
    vec = np.zeros(glove["vectors"].shape[1], dtype=np.float32)
    for tok in tokens:
        i = glove["index"].get(tok)
        if i is None:
            continue
        weight = index["idf"].get(tok, index["max_idf"]) if weighted else 1.0
        vec += weight * glove["vectors"][i]
    return vec


def score_all(index, tokens, method):
    """One cosine score per meaning (a 1-D array as long as the number of meanings)."""
    if method == "tfidf":
        query = index["tfv"].transform([tokens])
        return (index["tfidf"] @ query.T).toarray().ravel()
    vec = embed_query(index, tokens, weighted=(method != "glove_avg"))
    length = np.linalg.norm(vec)
    if length == 0:                      # none of the query words are in GloVe
        return np.zeros(len(index["docs"]), dtype=np.float32)
    return index["doc_vecs"][method] @ (vec / length)


def top_ids(scores, n):
    """Indices of the n highest scores, best first (ignoring zero scores)."""
    n = min(n, len(scores))
    best = np.argpartition(-scores, n - 1)[:n]
    best = best[np.argsort(-scores[best], kind="stable")]
    return [int(i) for i in best if scores[i] > 0]


def reciprocal_rank_fusion(rankings, weights):
    """Merge several ranked lists. A meaning scores sum(weight / (K + rank)).

    Ranks (not raw scores) are combined because TF-IDF and GloVe cosines live on
    different scales: 0.4 means "good" for one and "poor" for the other.
    """
    fused = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (RRF_K + rank)
    return sorted(fused.items(), key=lambda item: -item[1])


def ranked_ids(index, tokens, method, top_n=1000):
    """Best meaning ids for one method. Memoised: tuning the hybrid weights re-uses
    the same three rankings hundreds of times."""
    key = (tuple(tokens), method, top_n)
    if key not in index["memo"]:
        index["memo"][key] = top_ids(score_all(index, tokens, method), top_n)
    return index["memo"][key]


def rank_meanings(index, tokens, method, weights=DEFAULT_WEIGHTS, top_n=1000):
    """[(meaning id, score)] best first. For "hybrid", `weights` gives the share of
    (tfidf, glove_idf, headword)."""
    if method == "hybrid":
        rankings = [ranked_ids(index, tokens, part, top_n) for part in HYBRID_PARTS]
        return reciprocal_rank_fusion(rankings, weights)
    scores = score_all(index, tokens, method)
    return [(i, float(scores[i])) for i in top_ids(scores, top_n)]


# --------------------------------------------------------------------------- #
# From meanings to words
# --------------------------------------------------------------------------- #
def search(index, query, method="hybrid", k=10, weights=DEFAULT_WEIGHTS):
    """Main entry point: description in, list of candidate words out."""
    tokens = preprocess(query)
    # A reverse dictionary should never answer with a word you already typed.
    banned = set(raw_words(query)) | set(tokens)

    results, seen = [], set()
    for doc_id, score in rank_meanings(index, tokens, method, weights):
        doc = index["docs"][doc_id]
        for word in doc["words"]:
            if word in banned or word in seen:
                continue
            seen.add(word)
            results.append({"word": word, "pos": doc["pos"], "gloss": doc["gloss"],
                            "score": score, "doc_id": doc_id})
            if len(results) == k:
                return results
    return results


def semantic_links(index, query, doc_id, min_sim=0.45):
    """Explain a match: pairs (query word, definition word, similarity) that are close
    in GloVe space but are *different* words, e.g. ("heights", "altitude", 0.6).
    These are the matches TF-IDF cannot see."""
    glove = index["glove"]
    q_words = [t for t in dict.fromkeys(preprocess(query)) if t in glove["index"]]
    d_words = [t for t in dict.fromkeys(index["docs"][doc_id]["tokens"]) if t in glove["index"]]
    if not q_words or not d_words:
        return []

    def unit(words):
        mat = glove["vectors"][[glove["index"][w] for w in words]]
        return mat / np.linalg.norm(mat, axis=1, keepdims=True)

    sims = unit(q_words) @ unit(d_words).T
    links = []
    for i, q in enumerate(q_words):
        j = int(np.argmax(sims[i]))
        if q != d_words[j] and sims[i, j] >= min_sim:
            links.append((q, d_words[j], round(float(sims[i, j]), 2)))
    return links


def shared_words(index, query, doc_id):
    """Query words that literally appear in the definition (what TF-IDF sees)."""
    return sorted(set(preprocess(query)) & set(index["docs"][doc_id]["tokens"]))


def default_settings():
    """Best configuration found by evaluate.py (falls back to sensible defaults)."""
    path = ROOT / "results" / "best_params.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"use_related": False, "weights": list(DEFAULT_WEIGHTS)}
