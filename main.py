"""
main.py
-------
Demo entry point. Loads the sample sales dataset, spins up the agent with the
offline MockLLMClient by default (so `python main.py` works with zero setup),
and runs a few natural-language analytics questions through the RAG pipeline.

To use a real LLM instead of the mock:
    export OPENAI_API_KEY=sk-...
    python main.py --provider openai

    export ANTHROPIC_API_KEY=sk-ant-...
    python main.py --provider anthropic
"""

from __future__ import annotations

import argparse

import pandas as pd

from agent import AnthropicClient, DataAnalyticsAgent, Document, MockLLMClient, OpenAIClient


def build_llm_client(provider: str):
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    return MockLLMClient()


def main() -> None:
    parser = argparse.ArgumentParser(description="GenAI Data Analytics Agent demo")
    parser.add_argument("--provider", choices=["mock", "openai", "anthropic"], default="mock")
    parser.add_argument("--data", default="sample_data/sales_sample.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    business_context = [
        Document(
            doc_id="ctx-1",
            text="Revenue targets are reviewed per region at the end of each quarter.",
        ),
        Document(
            doc_id="ctx-2",
            text="Profit margin is calculated as profit divided by revenue for each product line.",
        ),
    ]

    llm_client = build_llm_client(args.provider)
    agent = DataAnalyticsAgent(df=df, llm_client=llm_client, context_docs=business_context)

    questions = [
        "What is the average revenue?",
        "What is the total profit?",
        "Which are the top regions by revenue?",
        "How many records are in the dataset?",
    ]

    for question in questions:
        response = agent.ask(question)
        print(f"\nQ: {response['question']}")
        print(f"Generated expression: {response['generated_expression']}")
        if response["error"]:
            print(f"Error: {response['error']}")
        else:
            print(f"Result:\n{response['result']}")


if __name__ == "__main__":
    main()
