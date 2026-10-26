"""Streamlit app for Reverse Dictionary: streamlit run app.py"""
import csv
from pathlib import Path
import pandas as pd
import streamlit as st

from retrieval import (
    METHODS,
    build_index,
    default_settings,
    load_glove,
    search,
    semantic_links,
    shared_words,
)

ROOT = Path(__file__).parent
RESULTS_DIR = ROOT / "results"

EXAMPLES = [
    "fear of heights",
    "a doctor who treats children",
    "the meal you eat around noon",
    "someone who walks while asleep",
    "a green spot with water in a desert",
    "a person who eats no meat",
    "a place where bees are kept",
    "an instrument for viewing faraway stars",
    "lasting only a very short time",
    "to make pain less severe",
]

POS_OPTIONS = {"All": None, "Noun": "noun", "Verb": "verb", "Adjective": "adj", "Adverb": "adv"}
METHOD_LABELS = {
    "hybrid": "Hybrid (TF-IDF + GloVe-IDF Fusion)",
    "glove_idf": "GloVe-IDF (Dense Rare-Word Weighted)",
    "glove_avg": "GloVe-Avg (Dense Unweighted Average)",
    "tfidf": "TF-IDF (Sparse Lexical Overlap)",
    "headword": "Headword Vector (Word-Level Embedding)",
}

st.set_page_config(
    page_title="Tip-of-the-Tongue Reverse Dictionary",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading WordNet meanings & GloVe vectors (~5-10s)...")
def get_index():
    settings = default_settings()
    return build_index(load_glove(), settings.get("use_related", False)), settings


@st.cache_data
def get_benchmark_queries():
    path = ROOT / "queries.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf8") as f:
        return list(csv.DictReader(f))


@st.cache_data
def get_metrics_table():
    path = RESULTS_DIR / "metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data
def get_significance_text():
    path = RESULTS_DIR / "significance.txt"
    if path.exists():
        return path.read_text(encoding="utf8")
    return None


@st.cache_data
def get_wins_losses_text():
    path = RESULTS_DIR / "wins_and_losses.md"
    if path.exists():
        return path.read_text(encoding="utf8")
    return None


def render_card(rank, r, query, index):
    """Render a single result card with styling, badge, definition, and explanations."""
    pos_colors = {
        "noun": "#1e88e5",
        "verb": "#43a047",
        "adj": "#fb8c00",
        "adv": "#8e24aa",
    }
    color = pos_colors.get(r["pos"], "#757575")

    with st.container():
        col_title, col_score = st.columns([4, 1])
        with col_title:
            st.markdown(
                f"### {rank}. **{r['word']}** "
                f"<span style='background-color:{color}; color:white; padding:2px 8px; "
                f"border-radius:12px; font-size:12px; font-weight:bold;'>{r['pos']}</span>",
                unsafe_allow_html=True,
            )
        with col_score:
            st.caption(f"Score: `{r['score']:.4f}`")

        st.write(f"*{r['gloss']}*")

        shared = shared_words(index, query, r["doc_id"])
        links = semantic_links(index, query, r["doc_id"])
        reasons = []
        if shared:
            reasons.append(f"**Exact match:** {', '.join(f'`{w}`' for w in shared)}")
        if links:
            reasons.append(f"**Semantic match:** {', '.join(f'`{a}` ≈ `{b}`' for a, b, _ in links)}")

        if reasons:
            st.info(" · ".join(reasons))
        st.divider()


def show_results(index, query, method, k, weights, pos=None):
    results = search(index, query, method, k, weights, pos=pos)
    if not results:
        st.warning(f"No matching words found for '{query}' with the selected filters.")
        return
    for rank, r in enumerate(results, start=1):
        render_card(rank, r, query, index)


# --------------------------------------------------------------------------- #
# Main App Layout
# --------------------------------------------------------------------------- #
index, settings = get_index()
weights = tuple(settings.get("weights", [0.2, 0.8, 0.0]))

st.title("🔎 Tip-of-the-Tongue Reverse Dictionary")
st.markdown(
    "Type a concept or definition, get the exact word. Searches **~75,000 WordNet meanings** "
    "comparing sparse lexical retrieval (TF-IDF) and dense semantic representations (GloVe-100d)."
)

tab_search, tab_eval, tab_theory = st.tabs([
    "🔍 Reverse Search",
    "📊 Benchmark & Evaluation",
    "🧠 Architecture & Theory",
])

# --------------------------------------------------------------------------- #
# Tab 1: Search
# --------------------------------------------------------------------------- #
with tab_search:
    st.sidebar.header("⚙️ Search Settings")
    method_key = st.sidebar.selectbox(
        "Retrieval Method",
        options=METHODS,
        format_func=lambda m: METHOD_LABELS.get(m, m),
        index=0,  # hybrid default
    )
    pos_label = st.sidebar.selectbox("Filter by Part of Speech", list(POS_OPTIONS.keys()), index=0)
    selected_pos = POS_OPTIONS[pos_label]
    k = st.sidebar.slider("Number of results (k)", min_value=3, max_value=20, value=8)

    st.sidebar.markdown("---")
    compare_mode = st.sidebar.toggle("Compare Two Methods Side-by-Side", value=False)
    if compare_mode:
        comp_m1 = st.sidebar.selectbox("Method A", METHODS, index=METHODS.index("tfidf"), format_func=lambda m: METHOD_LABELS[m])
        comp_m2 = st.sidebar.selectbox("Method B", METHODS, index=METHODS.index("glove_idf"), format_func=lambda m: METHOD_LABELS[m])

    # Preset queries selector
    preset = st.selectbox("Quick example queries", [""] + EXAMPLES)
    query_text = st.text_input(
        "Enter your description or definition:",
        value=preset if preset else "",
        placeholder="e.g. fear of heights, someone who bakes bread, or an instrument to view stars",
    )

    if query_text.strip():
        if compare_mode:
            st.markdown(f"#### Comparing Methods for: *\"{query_text}\"*")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Method A: {METHOD_LABELS.get(comp_m1, comp_m1)}")
                show_results(index, query_text, comp_m1, k, weights, pos=selected_pos)
            with col2:
                st.subheader(f"Method B: {METHOD_LABELS.get(comp_m2, comp_m2)}")
                show_results(index, query_text, comp_m2, k, weights, pos=selected_pos)
        else:
            st.markdown(f"#### Results using **{METHOD_LABELS.get(method_key, method_key)}**:")
            show_results(index, query_text, method_key, k, weights, pos=selected_pos)

# --------------------------------------------------------------------------- #
# Tab 2: Evaluation & Benchmark
# --------------------------------------------------------------------------- #
with tab_eval:
    st.header("📊 Benchmark & Empirical Evaluation")
    st.markdown(
        "Evaluation was conducted on **110 human-written queries** split randomly with fixed seed "
        "into **55 Dev** and **55 Test** sets. Hybrid weights were tuned exclusively on Dev, "
        "and numbers below represent performance on the **unseen held-out Test set**."
    )

    metrics_df = get_metrics_table()
    if metrics_df is not None:
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.subheader("Test Set Performance (Held-out 55 queries)")
            test_df = metrics_df[(metrics_df["mode"] == "gloss") & (metrics_df["split"] == "test")]
            st.dataframe(
                test_df[["method", "Recall@1", "Recall@5", "Recall@10", "MRR"]].reset_index(drop=True),
                use_container_width=True,
            )

            st.caption("Recall@k: Ground truth in top k candidates | MRR: Mean Reciprocal Rank")

        with col_m2:
            st.subheader("Dev Set Performance (Parameter Tuning)")
            dev_df = metrics_df[(metrics_df["mode"] == "gloss") & (metrics_df["split"] == "dev")]
            st.dataframe(
                dev_df[["method", "Recall@1", "Recall@5", "Recall@10", "MRR"]].reset_index(drop=True),
                use_container_width=True,
            )

        # Show metrics bar chart if available
        chart_path = RESULTS_DIR / "metrics_test.png"
        if chart_path.exists():
            st.subheader("Test Metric Comparison (Grouped Bar Chart)")
            st.image(str(chart_path), caption="Recall@1, Recall@10, and MRR across retrieval methods on held-out test data.", use_column_width=True)

    sig_text = get_significance_text()
    if sig_text:
        st.subheader("📈 Statistical Significance (Paired Bootstrap, 2000 Resamples)")
        st.code(sig_text, language="text")

    # Interactive Query Explorer
    st.subheader("🧪 Benchmark Dataset Explorer (110 Hand-Written Queries)")
    bench_queries = get_benchmark_queries()
    if bench_queries:
        selected_bench = st.selectbox(
            "Select a benchmark query to test across all methods:",
            [f"{q['query']}  -->  [{q['answers']}]" for q in bench_queries],
        )
        if selected_bench:
            chosen_q = selected_bench.split("  -->  ")[0]
            expected = selected_bench.split("  -->  ")[1]
            st.markdown(f"**Query:** `{chosen_q}` | **Target answers:** `{expected}`")

            b_cols = st.columns(3)
            with b_cols[0]:
                st.markdown("**TF-IDF (Lexical)**")
                for r in search(index, chosen_q, "tfidf", 5, weights):
                    mark = "✅ " if r["word"] in expected else ""
                    st.write(f"- {mark}**{r['word']}** ({r['pos']})")
            with b_cols[1]:
                st.markdown("**GloVe-IDF (Semantic)**")
                for r in search(index, chosen_q, "glove_idf", 5, weights):
                    mark = "✅ " if r["word"] in expected else ""
                    st.write(f"- {mark}**{r['word']}** ({r['pos']})")
            with b_cols[2]:
                st.markdown("**Hybrid Fusion**")
                for r in search(index, chosen_q, "hybrid", 5, weights):
                    mark = "✅ " if r["word"] in expected else ""
                    st.write(f"- {mark}**{r['word']}** ({r['pos']})")

    # Win / Loss analysis
    wins_text = get_wins_losses_text()
    if wins_text:
        with st.expander("🔍 Qualitative Error Analysis: Where Embeddings Win vs. Where TF-IDF Wins"):
            st.markdown(wins_text)

# --------------------------------------------------------------------------- #
# Tab 3: Architecture & Theory
# --------------------------------------------------------------------------- #
with tab_theory:
    st.header("🧠 System Architecture & Theoretical Foundation")
    st.markdown("""
### 1. The Tip-of-the-Tongue Problem
Standard dictionaries take a word and provide a definition. A **Reverse Dictionary** solves the inverse information retrieval problem: given a loose description, concept, or synonym, retrieve the exact target word.

### 2. Retrieval Methods Implemented
1. **TF-IDF (Term Frequency - Inverse Document Frequency)**:
   - Builds a bag-of-words inverted index over WordNet definitions.
   - Measures sparse cosine similarity:
   $$\\text{cosine}(q, d) = \\frac{\\mathbf{q} \\cdot \\mathbf{d}}{\\|\\mathbf{q}\\| \\|\\mathbf{d}\\|}$$
   - **Strength:** Excels when query and definition share exact vocabulary (e.g. *"person who eats no meat"* $\\rightarrow$ *vegetarian*).
   - **Weakness:** Vulnerable to the vocabulary mismatch problem (cannot connect *"noon"* to *"midday"*).

2. **GloVe-Avg (Dense Embedding Average)**:
   - Represents documents and queries as unweighted centroids of 100-dimensional GloVe vectors.
   - Vectors are **mean-centered** by subtracting the top 50,000 word centroid to eliminate common-direction bias.

3. **GloVe-IDF (IDF-Weighted Semantic Embeddings)**:
   - Weights each word vector by its WordNet corpus inverse document frequency:
   $$\\mathbf{v}_{doc} = \\sum_{w \\in doc} \\text{IDF}(w) \\cdot \\mathbf{v}_w$$
   - Prevents non-informative generic words (*"person"*, *"thing"*, *"make"*) from drowning out distinct concept words (*"bees"*, *"molten"*, *"stars"*).

4. **Headword Embeddings**:
   - Compares the query representation directly against the candidate word's own vector.
   - Tested as an empirical negative result: GloVe clusters words by *topical association* (words appearing near each other) rather than *definition equivalence*.

5. **Hybrid Rank Fusion (Reciprocal Rank Fusion - RRF)**:
   - Fuses ranked lists rather than raw scores because sparse TF-IDF cosines and dense GloVe cosines operate on entirely different scales:
   $$RRF(d) = \\sum_{m \\in M} \\frac{w_m}{k_{rrf} + \\text{rank}_m(d)}$$
   - Leverages TF-IDF precision when exact words match, and dense GloVe semantic generalization when phrasing diverges.

### 3. Pipeline Diagram
```
User Description Query
       │
       ▼
 [Preprocessing] (lowercase, regex tokenize, drop stop words keeping negations, lemmatize)
       │
  ┌────┴──────────────────────────┐
  │                               │
  ▼                               ▼
[TF-IDF Sparse Engine]    [GloVe Dense Engine]
  │                               │
  ▼                               ▼
Top-1000 Meanings Rank     Top-1000 Meanings Rank
  │                               │
  └───────────────┬───────────────┘
                  │
                  ▼
   [Reciprocal Rank Fusion (RRF)]
                  │
                  ▼
 [Filter query words & format synsets to words]
                  │
                  ▼
           Ranked Candidates
```
    """)
