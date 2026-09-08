"""
agent.py
--------
The GenAI Data Analytics Agent. This is the custom orchestration layer that
implements the RAG loop:

    user question
        -> retrieve relevant context (column descriptions, business rules)
        -> build an augmented prompt (retrieved context + schema + question)
        -> call the LLM to translate the question into a pandas expression
        -> execute the expression against the loaded dataset
        -> return a natural-language-friendly result

This mirrors what a LangChain "RetrievalQA"-style chain does internally, just
implemented directly so every step is explicit and inspectable.
"""

from __future__ import annotations

import pandas as pd

from .llm_client import LLMClient
from .query_executor import QueryExecutionError, run_pandas_expression
from .vector_store import Document, VectorStore


class DataAnalyticsAgent:
    def __init__(self, df: pd.DataFrame, llm_client: LLMClient, context_docs: list[Document] | None = None):
        self.df = df
        self.llm_client = llm_client
        self.vector_store = VectorStore()

        # Seed the vector store with schema info so retrieval has something
        # to work with even if the caller doesn't supply extra business docs.
        schema_doc = Document(
            doc_id="schema",
            text=self._build_schema_description(df),
            metadata={"type": "schema"},
        )
        self.vector_store.add_documents([schema_doc] + (context_docs or []))

    @staticmethod
    def _build_schema_description(df: pd.DataFrame) -> str:
        lines = [f"Column '{col}' has dtype {dtype}" for col, dtype in df.dtypes.items()]
        return "Dataset schema: " + "; ".join(lines)

    def _build_prompt(self, question: str, retrieved: list[tuple[Document, float]]) -> str:
        context_block = "\n".join(f"- {doc.text}" for doc, _score in retrieved)
        return (
            "You are a data analytics agent. Given the context and a pandas "
            "DataFrame named `df`, respond with a single pandas expression "
            "(no explanation) that answers the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Pandas expression:"
        )

    def ask(self, question: str) -> dict:
        """Run the full RAG pipeline for a natural-language analytics question."""
        retrieved = self.vector_store.search(question, top_k=3)
        prompt = self._build_prompt(question, retrieved)

        generated_expression = self.llm_client.generate(prompt).strip().strip("`")

        try:
            result = run_pandas_expression(generated_expression, self.df)
            error = None
        except QueryExecutionError as exc:
            result = None
            error = str(exc)

        return {
            "question": question,
            "retrieved_context": [doc.text for doc, _ in retrieved],
            "generated_expression": generated_expression,
            "result": result,
            "error": error,
        }
