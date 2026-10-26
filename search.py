"""Command-line reverse dictionary.

    python search.py "fear of heights"
    python search.py "a doctor who treats children" -k 8 --explain
    python search.py "someone who loves books" --method tfidf
    python search.py "fear of heights" --pos noun
    python search.py -i    # interactive mode (keeps model in memory for instant queries)
"""
import argparse
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from retrieval import (METHODS, build_index, default_settings, load_glove, search,
                       semantic_links, shared_words)


def display_results(index, query, results, method, explain=False):
    """Print formatted search results with optional explainability output."""
    if not results:
        print(f"No matching words found for: {query!r}\n")
        return

    print(f"\n{method.upper()} results for: {query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"{rank:>2}. {r['word']:<18} [{r['pos']:<4}]  {r['gloss']}")
        if explain:
            shared = shared_words(index, query, r["doc_id"])
            links = semantic_links(index, query, r["doc_id"])
            shared_str = ", ".join(shared) if shared else "-"
            # Use ASCII-compatible '~' to avoid cp1252 UnicodeEncodeError on Windows terminals
            links_str = ", ".join(f"{a} ~ {b} ({s})" for a, b, s in links) if links else "-"
            print(f"      shared words: {shared_str}")
            print(f"      semantic links: {links_str}")
    print()


def interactive_mode(index, method, k, weights, pos, explain):
    """Run an interactive prompt that keeps the index loaded in memory."""
    print("=" * 60)
    print("🔎 Tip-of-the-Tongue Reverse Dictionary (Interactive Shell)")
    print("   Type a description or 'exit' / 'quit' to quit.")
    print(f"   Config: method={method}, k={k}, pos={pos or 'all'}, explain={explain}")
    print("=" * 60)

    while True:
        try:
            query = input("\nDescribe a word > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        results = search(index, query, method, k, weights, pos=pos)
        display_results(index, query, results, method, explain)


def main():
    parser = argparse.ArgumentParser(description="Describe a word, get the word.")
    parser.add_argument("query", nargs="?", default=None,
                        help='description, e.g. "fear of heights" (omit for interactive mode)')
    parser.add_argument("-k", type=int, default=10, help="how many candidate words to show")
    parser.add_argument("--method", choices=METHODS, default="hybrid",
                        help=f"retrieval method to use ({', '.join(METHODS)})")
    parser.add_argument("--pos", choices=["noun", "verb", "adj", "adv"], default=None,
                        help="filter candidate words by part of speech")
    parser.add_argument("--explain", action="store_true",
                        help="show why each word matched (shared words + semantic links)")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="launch interactive search prompt")
    args = parser.parse_args()

    settings = default_settings()   # best config found by evaluate.py
    print("Loading WordNet + GloVe (a few seconds once cached)...")
    index = build_index(load_glove(), settings["use_related"])
    weights = tuple(settings["weights"])

    if args.interactive or args.query is None:
        interactive_mode(index, args.method, args.k, weights, args.pos, args.explain)
    else:
        results = search(index, args.query, args.method, args.k, weights, pos=args.pos)
        display_results(index, args.query, results, args.method, args.explain)


if __name__ == "__main__":
    main()
