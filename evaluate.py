"""Evaluate all retrieval methods on the hand-written queries in queries.csv.

Protocol (the part that makes the numbers trustworthy):
  1. Queries are split once, at random with a fixed seed, into DEV and TEST halves.
  2. Anything we tune (hybrid weight, "gloss" vs "gloss+related") is chosen on DEV only.
  3. The headline numbers are reported on TEST, which never influenced any choice.

Metrics (a query has several acceptable answers, e.g. synonyms; best one counts):
  Recall@k : share of queries whose answer appears in the top k words
  MRR      : mean of 1/rank of the first correct word (rank looked at up to 100)

Run:  python evaluate.py        (writes everything into results/)
"""
import csv
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from retrieval import DEFAULT_WEIGHTS, METHODS, build_index, load_glove, search, semantic_links, shared_words

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
MODES = {"gloss": False, "gloss+related": True}   # name -> use_related
MAX_RANK = 100
# Every way to split 100% between (tfidf, glove_idf, headword) in steps of 20%.
WEIGHT_GRID = [(a / 5, b / 5, (5 - a - b) / 5) for a in range(6) for b in range(6 - a)]


# --------------------------------------------------------------------------- #
# Queries and metrics (pure functions, unit-tested in test_retrieval.py)
# --------------------------------------------------------------------------- #
def load_queries(path, known_words):
    """Read queries.csv. Answers that WordNet does not contain are dropped, and
    queries left without any answer are skipped (a method cannot be blamed for those)."""
    queries = []
    with open(path, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            answers = {a for a in row["answers"].split("|") if a in known_words}
            if answers:
                queries.append({"query": row["query"], "answers": answers})
            else:
                print(f"  skipped (no answer in WordNet): {row['query']}")
    return queries


def split_queries(queries, seed=42):
    """Random 50/50 split into (dev, test)."""
    shuffled = queries[:]
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


def answer_rank(ranked_words, answers):
    """1-based rank of the first correct word, or None if it is not in the list."""
    for rank, word in enumerate(ranked_words, start=1):
        if word in answers:
            return rank
    return None


def summarize(ranks):
    """Recall@1/5/10 and MRR from a list of ranks (None = not found)."""
    n = len(ranks)
    return {
        "Recall@1": sum(r is not None and r <= 1 for r in ranks) / n,
        "Recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "Recall@10": sum(r is not None and r <= 10 for r in ranks) / n,
        "MRR": sum(1 / r for r in ranks if r is not None) / n,
    }


def bootstrap_difference(ranks_a, ranks_b, metric, rounds=2000, seed=0):
    """Paired bootstrap for (method A - method B) on one metric.

    Re-draws the queries with replacement many times and recomputes the difference each
    time. Returns (observed difference, 95% interval). If the interval contains 0, the
    gap could easily be luck: with ~55 queries that happens a lot.
    """
    rng = random.Random(seed)
    n = len(ranks_a)
    diffs = []
    for _ in range(rounds):
        pick = [rng.randrange(n) for _ in range(n)]
        a = summarize([ranks_a[i] for i in pick])[metric]
        b = summarize([ranks_b[i] for i in pick])[metric]
        diffs.append(a - b)
    diffs.sort()
    observed = summarize(ranks_a)[metric] - summarize(ranks_b)[metric]
    return observed, (diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1])


# --------------------------------------------------------------------------- #
# Running the methods
# --------------------------------------------------------------------------- #
def rank_queries(index, queries, method, weights=DEFAULT_WEIGHTS):
    """Rank of the correct answer for every query."""
    ranks = []
    for q in queries:
        words = [r["word"] for r in search(index, q["query"], method, MAX_RANK, weights)]
        ranks.append(answer_rank(words, q["answers"]))
    return ranks


def tune_weights(index, dev):
    """Pick the hybrid weights with the best MRR on DEV. Ties go to the most even split."""
    scored = []
    for w in WEIGHT_GRID:
        mrr = summarize(rank_queries(index, dev, "hybrid", w))["MRR"]
        scored.append((round(mrr, 6), -max(w), w))
    best_mrr, _, best_w = max(scored)
    return best_w, best_mrr


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(table, path):
    """Grouped bars on TEST: one panel per metric, one bar per text mode."""
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    colors = {"gloss": "#2a78d6", "gloss+related": "#eb6834"}
    metrics = ["Recall@1", "Recall@10", "MRR"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=surface, sharey=True)
    width = 0.38
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor(surface)
        for k, mode in enumerate(MODES):
            values = [table[(mode, m)]["test"][metric] for m in METHODS]
            xs = [i + (k - 0.5) * width for i in range(len(METHODS))]
            bars = ax.bar(xs, values, width * 0.92, color=colors[mode],
                          label=mode if metric == metrics[0] else None)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, color=muted)
        ax.set_title(metric, loc="left", fontsize=11, color=ink)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, fontsize=9, color=muted)
        ax.set_ylim(0, 0.6)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    fig.legend(loc="upper right", ncol=2, frameon=False, labelcolor=muted, fontsize=9)
    fig.suptitle("Reverse dictionary: held-out TEST queries", x=0.01, ha="left",
                 fontsize=13, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def write_wins_and_losses(index, path, queries):
    """Qualitative error analysis: where do embeddings beat TF-IDF, and the reverse?"""
    lines = ["# Where embeddings win, and where they lose\n",
             "Computed on all queries with the best configuration. A method 'finds' a query when the "
             "answer is in its top 10 words.\n"]
    groups = {"Embeddings find it, TF-IDF misses": [], "TF-IDF finds it, embeddings miss": [],
              "Both miss": []}
    for q in queries:
        top = {}
        for m in ("tfidf", "glove_idf"):
            top[m] = [r["word"] for r in search(index, q["query"], m, MAX_RANK)]
        rank = {m: answer_rank(top[m], q["answers"]) for m in top}
        hit = {m: rank[m] is not None and rank[m] <= 10 for m in rank}
        entry = (q, rank, top)
        if hit["glove_idf"] and not hit["tfidf"]:
            groups["Embeddings find it, TF-IDF misses"].append(entry)
        elif hit["tfidf"] and not hit["glove_idf"]:
            groups["TF-IDF finds it, embeddings miss"].append(entry)
        elif not hit["tfidf"] and not hit["glove_idf"]:
            groups["Both miss"].append(entry)

    for title, entries in groups.items():
        lines.append(f"\n## {title} ({len(entries)} queries)\n")
        for q, rank, top in entries:
            answer = sorted(q["answers"])[0]
            lines.append(f"- **{q['query']}** -> expected `{'`/`'.join(sorted(q['answers']))}`  "
                         f"(TF-IDF rank: {rank['tfidf']}, GloVe-IDF rank: {rank['glove_idf']})")
            lines.append(f"  - TF-IDF top 5: {', '.join(top['tfidf'][:5])}")
            lines.append(f"  - GloVe-IDF top 5: {', '.join(top['glove_idf'][:5])}")
            if title.startswith("Embeddings"):
                # show the "invisible to TF-IDF" evidence for the best-ranked correct word
                for r in search(index, q["query"], "glove_idf", MAX_RANK):
                    if r["word"] in q["answers"]:
                        links = semantic_links(index, q["query"], r["doc_id"])
                        shared = shared_words(index, q["query"], r["doc_id"])
                        lines.append(f"  - definition: \"{r['gloss']}\"; shared words: "
                                     f"{shared or 'none'}; semantic links: {links or 'none'}")
                        break
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")
    return {title: len(entries) for title, entries in groups.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    RESULTS.mkdir(exist_ok=True)
    glove = load_glove()
    indexes = {mode: build_index(glove, use_related) for mode, use_related in MODES.items()}
    known = {w for d in indexes["gloss"]["docs"] for w in d["words"]}

    queries = load_queries(ROOT / "queries.csv", known)
    dev, test = split_queries(queries)
    print(f"{len(queries)} queries -> {len(dev)} dev / {len(test)} test")

    table, rows, per_query, test_ranks = {}, [], [], {}
    best = {"mrr": -1}
    for mode, index in indexes.items():
        weight, dev_mrr = tune_weights(index, dev)
        print(f"[{mode}] hybrid weights tuned on dev (tfidf, glove_idf, headword) = {weight}")
        if dev_mrr > best["mrr"]:
            best = {"mrr": dev_mrr, "use_related": MODES[mode], "weights": list(weight), "mode": mode}
        for method in METHODS:
            ranks = {"dev": rank_queries(index, dev, method, weight),
                     "test": rank_queries(index, test, method, weight)}
            table[(mode, method)] = {s: summarize(r) for s, r in ranks.items()}
            test_ranks[(mode, method)] = ranks["test"]
            for split, qs in (("dev", dev), ("test", test)):
                for q, rank in zip(qs, ranks[split]):
                    per_query.append({"mode": mode, "method": method, "split": split,
                                      "query": q["query"], "answers": "|".join(sorted(q["answers"])),
                                      "rank": "" if rank is None else rank})
                rows.append({"mode": mode, "method": method, "split": split,
                             "n_queries": len(qs), **{k: round(v, 3) for k, v in
                                                      table[(mode, method)][split].items()}})

    write_csv(RESULTS / "metrics.csv", rows)
    write_csv(RESULTS / "per_query_ranks.csv", per_query)
    plot_results(table, RESULTS / "metrics_test.png")

    chosen = indexes[best["mode"]]
    counts = write_wins_and_losses(chosen, RESULTS / "wins_and_losses.md", queries)
    params = {"use_related": best["use_related"], "weights": best["weights"],
              "chosen_on": "dev MRR of the hybrid", "mode": best["mode"]}
    (RESULTS / "best_params.json").write_text(json.dumps(params, indent=2))

    print("\nTEST results")
    print(f"{'mode':15}{'method':11}{'R@1':>6}{'R@5':>6}{'R@10':>6}{'MRR':>7}")
    for (mode, method), by_split in table.items():
        m = by_split["test"]
        print(f"{mode:15}{method:11}{m['Recall@1']:6.2f}{m['Recall@5']:6.2f}"
              f"{m['Recall@10']:6.2f}{m['MRR']:7.3f}")
    print("\nbest config:", params)

    # Is the hybrid really better than plain TF-IDF, or is it luck of 55 queries?
    mode = best["mode"]
    lines = [f"Paired bootstrap on the {len(test)} TEST queries, mode = {mode}, 2000 resamples", ""]
    for metric in ("Recall@10", "MRR"):
        for other in ("tfidf", "glove_idf"):
            diff, (lo, hi) = bootstrap_difference(test_ranks[(mode, "hybrid")],
                                                  test_ranks[(mode, other)], metric)
            lines.append(f"hybrid - {other:9} {metric:10} {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    (RESULTS / "significance.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print("win/loss counts:", counts)


if __name__ == "__main__":
    main()
