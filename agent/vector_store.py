"""
vector_store.py
----------------
A lightweight, dependency-free (no external vector DB) implementation of the
retrieval half of Retrieval-Augmented Generation (RAG).

Documents (business context notes, column descriptions, prior Q&A pairs, etc.)
are embedded using TF-IDF and retrieved via cosine similarity. This is the
same retrieval pattern used by heavier stacks (LangChain + FAISS / Pinecone /
Chroma) -- swap this class for one backed by a real vector DB in production
without changing the rest of the agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """In-memory vector store using TF-IDF embeddings and cosine similarity search."""

    def __init__(self) -> None:
        self._documents: List[Document] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None  # sparse TF-IDF matrix, one row per document

    def add_documents(self, documents: List[Document]) -> None:
        self._documents.extend(documents)
        self._reindex()

    def _reindex(self) -> None:
        corpus = [doc.text for doc in self._documents]
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform(corpus)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Return the top_k most relevant documents for a query, with similarity scores."""
        if not self._documents or self._vectorizer is None:
            return []

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            if scores[idx] > 0:
                results.append((self._documents[idx], float(scores[idx])))
        return results

    def __len__(self) -> int:
        return len(self._documents)
