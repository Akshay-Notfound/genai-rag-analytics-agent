import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent import DataAnalyticsAgent, Document, MockLLMClient, VectorStore  # noqa: E402


def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"],
            "revenue": [100, 200, 300, 400],
            "profit": [10, 20, 30, 40],
        }
    )


def test_vector_store_retrieves_relevant_document():
    store = VectorStore()
    store.add_documents(
        [
            Document("d1", "Revenue targets are reviewed quarterly per region."),
            Document("d2", "The office coffee machine is broken."),
        ]
    )
    results = store.search("What are the revenue targets?", top_k=1)
    assert len(results) == 1
    assert results[0][0].doc_id == "d1"


def test_agent_answers_average_question():
    df = sample_df()
    agent = DataAnalyticsAgent(df=df, llm_client=MockLLMClient())
    response = agent.ask("What is the average revenue?")
    assert response["error"] is None
    assert response["result"] == df["revenue"].mean()


def test_agent_answers_total_question():
    df = sample_df()
    agent = DataAnalyticsAgent(df=df, llm_client=MockLLMClient())
    response = agent.ask("What is the total profit?")
    assert response["error"] is None
    assert response["result"] == df["profit"].sum()


def test_agent_top_regions_question():
    df = sample_df()
    agent = DataAnalyticsAgent(df=df, llm_client=MockLLMClient())
    response = agent.ask("What are the top regions by revenue?")
    assert response["error"] is None
    assert response["result"].index[0] == "West"  # highest revenue


def test_agent_count_question():
    df = sample_df()
    agent = DataAnalyticsAgent(df=df, llm_client=MockLLMClient())
    response = agent.ask("How many records are there?")
    assert response["error"] is None
    assert response["result"] == len(df)
