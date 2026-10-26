"""Run with:  pytest -q

The first group needs no data (pure logic); the second group uses the real index
and checks a few well-known behaviours end to end.
"""
import pytest

from evaluate import answer_rank, bootstrap_difference, split_queries, summarize
from retrieval import (
    METHODS,
    build_index,
    load_glove,
    reciprocal_rank_fusion,
    search,
    semantic_links,
    shared_words,
)
from text_utils import preprocess


# ---------- pure logic ----------
def test_preprocess_lemmatizes_and_drops_stop_words():
    assert preprocess("A person who loves books") == ["person", "love", "book"]


def test_preprocess_keeps_negations():
    assert "no" in preprocess("having no money")


def test_answer_rank_is_one_based_and_handles_miss():
    assert answer_rank(["a", "b", "c"], {"b"}) == 2
    assert answer_rank(["a", "b"], {"z"}) is None


def test_summarize_metrics():
    m = summarize([1, 5, None, 20])
    assert m["Recall@1"] == 0.25 and m["Recall@5"] == 0.5 and m["Recall@10"] == 0.5
    assert m["MRR"] == pytest.approx((1 + 1 / 5 + 1 / 20) / 4)


def test_rrf_rewards_agreement_between_rankings():
    # doc 2 is near the top of both lists; doc 1 is first in one list but last in the other
    fused = reciprocal_rank_fusion([[1, 2, 3, 4, 5], [2, 3, 4, 5, 1]], [0.5, 0.5])
    assert fused[0][0] == 2


def test_rrf_tie_breaking_is_deterministic():
    # two items with identical scores must be sorted by doc_id
    fused = reciprocal_rank_fusion([[5, 2]], [1.0])
    # scores are 1/(60+1) for 5 and 1/(60+2) for 2, so 5 first
    assert fused[0][0] == 5
    # identical rank lists reversed
    fused_tied = reciprocal_rank_fusion([[10], [20]], [0.5, 0.5])
    # both score 0.5/(60+1) -> identical score -> tie broken by doc_id ascending (10 < 20)
    assert fused_tied[0][0] == 10 and fused_tied[1][0] == 20


def test_split_is_deterministic_and_disjoint():
    qs = [{"query": str(i), "answers": {"x"}} for i in range(20)]
    dev1, test1 = split_queries(qs)
    dev2, test2 = split_queries(qs)
    assert dev1 == dev2 and test1 == test2
    assert not {q["query"] for q in dev1} & {q["query"] for q in test1}


def test_bootstrap_interval_contains_observed_difference():
    a, b = [1, 2, None, 3, 1, 4], [2, None, None, 5, 3, 9]
    diff, (low, high) = bootstrap_difference(a, b, "MRR", rounds=300)
    assert low <= diff <= high


# ---------- end to end (uses WordNet + GloVe) ----------
@pytest.fixture(scope="module")
def index():
    return build_index(load_glove(), use_related=False)


@pytest.mark.parametrize("method", ["tfidf", "glove_idf", "hybrid"])
def test_fear_of_heights_finds_acrophobia(index, method):
    words = [r["word"] for r in search(index, "fear of heights", method, 5)]
    assert "acrophobia" in words


def test_embeddings_understand_noon_means_midday(index):
    # No shared content word with the definition "a midday meal" except "meal":
    # TF-IDF buries lunch, embeddings put it in the top 5.
    dense = [r["word"] for r in search(index, "the meal you eat around noon", "glove_idf", 5)]
    assert "lunch" in dense


def test_never_returns_a_word_from_the_query(index):
    words = [r["word"] for r in search(index, "a baby dog", "hybrid", 20)]
    assert "baby" not in words and "dog" not in words


def test_nonsense_query_returns_empty_not_crash(index):
    assert search(index, "zzzqx qqwwee", "hybrid", 5) == []
    assert search(index, "", "tfidf", 5) == []


def test_pos_filtering(index):
    noun_results = search(index, "fear of heights", "hybrid", 6, pos="noun")
    assert len(noun_results) > 0
    assert all(r["pos"] == "noun" for r in noun_results)

    verb_results = search(index, "to cause fear in someone", "hybrid", 6, pos="verb")
    assert len(verb_results) > 0
    assert all(r["pos"] == "verb" for r in verb_results)


def test_k_parameter_limits_count(index):
    r3 = search(index, "fear of heights", "hybrid", k=3)
    r7 = search(index, "fear of heights", "hybrid", k=7)
    assert len(r3) == 3
    assert len(r7) == 7


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_return_expected_schema(index, method):
    res = search(index, "someone who bakes bread", method, k=4)
    assert len(res) > 0
    for item in res:
        assert "word" in item
        assert "pos" in item
        assert "gloss" in item
        assert "score" in item
        assert "doc_id" in item


def test_explain_shared_and_semantic_links(index):
    res = search(index, "fear of heights", "hybrid", k=1)
    assert len(res) == 1
    doc_id = res[0]["doc_id"]
    shared = shared_words(index, "fear of heights", doc_id)
    assert "fear" in shared or "height" in shared
    links = semantic_links(index, "fear of heights", doc_id)
    assert isinstance(links, list)
