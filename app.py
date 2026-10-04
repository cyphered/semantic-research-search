import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Semantic Research Search", page_icon="🔎")
st.title("Semantic Research Search")
st.caption("A lightweight information-retrieval prototype using TF-IDF and cosine similarity.")

DEFAULT_DOCS = {
    "AI in Sustainable Cities": """Machine learning can support sustainable urban planning by predicting energy demand, traffic congestion, air quality, and heat risk. Data-driven systems can help cities allocate resources efficiently.""",
    "Retrieval Augmented Generation": """Retrieval augmented generation combines information retrieval with language models. A retriever finds relevant documents and a generator uses the retrieved context to answer questions.""",
    "Information Retrieval": """Information retrieval is the process of finding relevant information from a large collection of documents. Classical retrieval systems use term weighting methods such as TF-IDF and ranking functions to measure relevance.""",
    "Natural Language Processing": """Natural language processing enables computers to process human language. Common applications include document classification, search, question answering, summarization, and information extraction.""",
    "Machine Learning for Climate": """Machine learning models can analyze environmental observations and satellite data. Applications include climate prediction, wildfire detection, land-cover mapping, and environmental monitoring."""
}

def search(query, docs, top_k=5):
    names = list(docs.keys())
    corpus = list(docs.values())
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(corpus + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = scores.argsort()[::-1][:top_k]
    return [(names[i], corpus[i], float(scores[i])) for i in ranked]

st.sidebar.header("Document collection")
uploaded = st.sidebar.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)

docs = DEFAULT_DOCS.copy()
if uploaded:
    for file in uploaded:
        text = file.read().decode("utf-8", errors="ignore").strip()
        if text:
            docs[file.name] = text

query = st.text_input("Search your research collection",
                      placeholder="e.g. machine learning for environmental monitoring")
top_k = st.slider("Number of results", 1, min(5, len(docs)), min(5, len(docs)))

if query:
    results = search(query, docs, top_k)
    st.subheader("Ranked results")
    for rank, (name, text, score) in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(f"### {rank}. {name}")
            st.write(f"**Relevance score:** {score:.3f}")
            st.write(text)
else:
    st.info("Enter a query to rank the documents by relevance.")

st.divider()
st.markdown("**Method:** TF-IDF term weighting → cosine similarity → ranked retrieval.")
