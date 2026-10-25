"""Streamlit app:  streamlit run app.py"""
import streamlit as st

from retrieval import (build_index, default_settings, load_glove, search, semantic_links,
                       shared_words)

EXAMPLES = ["fear of heights", "a doctor who treats children", "the meal you eat around noon",
            "someone who walks while asleep", "a green spot with water in a desert"]

st.set_page_config(page_title="Reverse Dictionary", page_icon="🔎")


@st.cache_resource(show_spinner="Loading WordNet + GloVe (first run takes about a minute)...")
def get_index():
    settings = default_settings()
    return build_index(load_glove(), settings["use_related"]), settings


def show_results(index, query, method, k, weights):
    """Render one ranked list: word, part of speech, definition and why it matched."""
    for rank, r in enumerate(search(index, query, method, k, weights), start=1):
        st.markdown(f"**{rank}. {r['word']}** *({r['pos']})*  \n{r['gloss']}")
        shared = shared_words(index, query, r["doc_id"])
        links = semantic_links(index, query, r["doc_id"])
        why = []
        if shared:
            why.append("same words: " + ", ".join(shared))
        if links:
            why.append("similar meaning: " + ", ".join(f"{a} ≈ {b}" for a, b, _ in links))
        if why:
            st.caption(" · ".join(why))


st.title("🔎 Reverse Dictionary")
st.write("Describe a word you can't remember, and get the word back.")

index, settings = get_index()
weights = tuple(settings["weights"])

example = st.selectbox("Try an example", [""] + EXAMPLES)
query = st.text_input("Describe the word", value=example, placeholder="e.g. fear of heights")
k = st.slider("How many words", 3, 15, 8)
compare = st.toggle("Compare TF-IDF with embeddings side by side", value=False)

if query.strip():
    if compare:
        left, right = st.columns(2)
        with left:
            st.subheader("TF-IDF (word overlap)")
            show_results(index, query, "tfidf", k, weights)
        with right:
            st.subheader("GloVe-IDF (meaning)")
            show_results(index, query, "glove_idf", k, weights)
    else:
        show_results(index, query, "hybrid", k, weights)
