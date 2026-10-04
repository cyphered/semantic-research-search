# Semantic Research Search

A lightweight Information Retrieval (IR) prototype that ranks research documents against a natural-language query using **TF-IDF** and **cosine similarity**.

## Why this project?

Search systems are a core application of Information Retrieval and an important component of modern AI systems such as retrieval-augmented generation (RAG). This prototype demonstrates a classical retrieval baseline before moving to neural/embedding-based retrieval.

## Features

- Search a research-document collection
- Rank documents by relevance
- TF-IDF term weighting
- Cosine-similarity scoring
- Upload your own `.txt` documents
- Streamlit interface
- Transparent relevance scores

## Architecture

```text
User query
    ↓
TF-IDF vectorization
    ↓
Document-query cosine similarity
    ↓
Ranking
    ↓
Top-k relevant documents
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Example queries

```text
machine learning environmental monitoring
```

```text
how does retrieval augmented generation work
```

## Concepts demonstrated

Information Retrieval, TF-IDF, cosine similarity, ranking, NLP, and search-system design.

## Limitations

This is a classical lexical retrieval baseline. It does not understand semantic similarity when two texts use very different vocabulary.

## Future work

1. Add BM25 retrieval.
2. Add sentence-transformer embeddings.
3. Compare lexical and semantic retrieval.
4. Add precision@k and recall@k evaluation.
5. Extend the system toward a RAG pipeline.

## Academic note

This is intentionally a small, interpretable baseline that can be extended into a rigorous Information Retrieval experiment.
