"""Command-line reverse dictionary.

    python search.py "fear of heights"
    python search.py "a doctor who treats children" -k 8 --explain
    python search.py "someone who loves books" --method tfidf
"""
import argparse

from retrieval import (METHODS, build_index, default_settings, load_glove, search,
                       semantic_links, shared_words)


def main():
    parser = argparse.ArgumentParser(description="Describe a word, get the word.")
    parser.add_argument("query", help='description, e.g. "fear of heights"')
    parser.add_argument("-k", type=int, default=10, help="how many words to show")
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--explain", action="store_true",
                        help="show why each word matched (shared words + semantic links)")
    args = parser.parse_args()

    settings = default_settings()   # best config found by evaluate.py
    print("Loading WordNet + GloVe (a few seconds once cached)...")
    index = build_index(load_glove(), settings["use_related"])
    results = search(index, args.query, args.method, args.k, tuple(settings["weights"]))

    print(f"\n{args.method} results for: {args.query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"{rank:>2}. {r['word']:<16} ({r['pos']})  {r['gloss']}")
        if args.explain:
            shared = shared_words(index, args.query, r["doc_id"])
            links = semantic_links(index, args.query, r["doc_id"])
            print(f"      shared words: {shared or '-'}   semantic links: {links or '-'}")


if __name__ == "__main__":
    main()
