# GenAI RAG Data Analytics Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/NumPy-1.24%2B-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/OpenAI-API-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI" />
  <img src="https://img.shields.io/badge/Anthropic-Claude-D97706?style=for-the-badge&logo=anthropic&logoColor=white" alt="Anthropic" />
  <img src="https://img.shields.io/badge/Pytest-Passing-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="Pytest" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License" />
</p>

A custom-orchestrated **Retrieval-Augmented Generation (RAG) Data Analytics Agent** that translates natural language questions into executable Pandas expressions, executes them securely, and returns computed results over your datasets.

---

## 📑 Table of Contents
- [Architecture](#-architecture)
- [Tools & Technologies](#-tools--technologies)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Running the Project](#-running-the-project)
  - [1. Offline Demo (Zero Setup / Mock LLM)](#1-offline-demo-zero-setup--mock-llm)
  - [2. Live OpenAI Integration](#2-live-openai-integration)
  - [3. Live Anthropic Integration](#3-live-anthropic-integration)
- [Running Tests](#-running-tests)
- [How It Works](#-how-it-works)
- [Security & Safe Execution](#-security--safe-execution)
- [Extending This Project](#-extending-this-project)

---

## 🏛 Architecture

```
                       Natural Language Question
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    1. Semantic Retrieval Layer                       │
│  VectorStore.search() (TF-IDF Embeddings + Cosine Similarity)         │
│  - Retrieved Schema & Business Context Documents                     │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    2. Dynamic Prompt Construction                    │
│  Combines: Question + Retrieved Context + DataFrame Column Metadata  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    3. LLM Code Generation Layer                      │
│  LLMClient.generate()                                                │
│  - Supports: OpenAI (GPT-4/3.5), Anthropic (Claude), Offline Mock   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    4. Sandboxed Query Executor                       │
│  query_executor.run_pandas_expression()                              │
│  - Restricted namespace evaluation without arbitrary code execution  │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
                         Real Computed Result
```

This mirrors what LangChain `RetrievalQA` chains execute under the hood, implemented as an inspectable, transparent, and custom orchestration framework.

---

## 🛠 Tools & Technologies

| Tool / Technology | Badge | Purpose |
| :--- | :---: | :--- |
| **Python** | `Python 3.10+` | Core programming language for agent logic, prompt construction, and execution. |
| **Pandas** | `Pandas` | Data manipulation, dataframe aggregations, time-series operations, and statistical calculations. |
| **Scikit-Learn** | `scikit-learn` | TF-IDF vectorization and cosine similarity retrieval for domain context matching. |
| **NumPy** | `NumPy` | Fast numerical operations and vector manipulation. |
| **OpenAI API** | `OpenAI` | State-of-the-art LLM provider (GPT-4 / GPT-3.5) for translating analytics prompts into code. |
| **Anthropic API** | `Claude` | Frontier Claude model integration for precise data query reasoning. |
| **Pytest** | `Pytest` | Comprehensive test runner verifying retrieval, prompt parsing, and analytics output. |
| **Requests** | `Requests` | Resilient HTTP communications with external LLM APIs. |

---

## ✨ Key Features

- **End-to-End RAG Pipeline**: Ingests dataset schemas and domain knowledge (e.g. quarterly business goals, KPI definitions) to answer analytics questions with context.
- **Pluggable LLM Providers**:
  - `MockLLMClient`: Offline, deterministic stand-in for immediate local testing without API keys or costs.
  - `OpenAIClient`: Live code generation using OpenAI models.
  - `AnthropicClient`: Live code generation using Anthropic Claude models.
- **Safe Sandboxed Execution**: Evaluates generated Pandas code in a heavily scoped namespace with stripped builtins and no dangerous OS/import primitives.
- **Self-Contained Vector Store**: Built-in TF-IDF vector search means zero external vector database dependencies required to run.
- **Extensible Architecture**: Ready to swap in FAISS, Chroma, or Pinecone, add conversational history, or wrap in a FastAPI endpoint.

---

## 📂 Project Structure

```
genai-rag-analytics-agent/
├── agent/
│   ├── __init__.py           # Module exports (DataAnalyticsAgent, VectorStore, etc.)
│   ├── agent.py              # Main RAG orchestration layer
│   ├── llm_client.py         # OpenAI, Anthropic, and deterministic Mock clients
│   ├── query_executor.py     # Safe, sandboxed evaluation of generated queries
│   └── vector_store.py       # TF-IDF embedding & cosine similarity retrieval
├── sample_data/
│   └── sales_sample.csv      # Sample sales dataset (region, product, units, revenue, profit)
├── tests/
│   └── test_agent.py         # Unit tests covering vector retrieval and query execution
├── main.py                   # Demo CLI application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10** or higher
- `pip` or `uv` package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Akshay-Notfound/genai-rag-analytics-agent.git
   cd genai-rag-analytics-agent
   ```

2. **(Optional) Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Running the Project

### 1. Offline Demo (Zero Setup / Mock LLM)
Run the agent right away without needing any API key:
```bash
python main.py
```
Or explicitly specify the mock provider:
```bash
python main.py --provider mock
```

**Sample Output:**
```text
Q: What is the average revenue?
Generated expression: df['revenue'].mean()
Result:
3935.0

Q: What is the total profit?
Generated expression: df['profit'].sum()
Result:
9550

Q: Which are the top regions by revenue?
Generated expression: df.groupby('region')['revenue'].sum().sort_values(ascending=False).head(5)
Result:
region
North    11900
South    11000
East     10350
West      6100
Name: revenue, dtype: int64

Q: How many records are in the dataset?
Generated expression: len(df)
Result:
10
```

---

### 2. Live OpenAI Integration
Set your OpenAI API key and run:

**PowerShell (Windows):**
```powershell
$env:OPENAI_API_KEY="sk-..."
python main.py --provider openai
```

**Bash (Linux / macOS):**
```bash
export OPENAI_API_KEY="sk-..."
python main.py --provider openai
```

---

### 3. Live Anthropic Integration
Set your Anthropic API key and run:

**PowerShell (Windows):**
```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
python main.py --provider anthropic
```

**Bash (Linux / macOS):**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py --provider anthropic
```

---

## 🧪 Running Tests

A complete unit test suite validates vector store retrieval, mock LLM generation, and dataframe computation:

```bash
python -m pytest tests/ -v
```

Expected output:
```text
tests/test_agent.py::test_vector_store_retrieves_relevant_document PASSED
tests/test_agent.py::test_agent_answers_average_question PASSED
tests/test_agent.py::test_agent_answers_total_question PASSED
tests/test_agent.py::test_agent_top_regions_question PASSED
tests/test_agent.py::test_agent_count_question PASSED

============================== 5 passed in 1.53s ===============================
```

---

## 🔍 How It Works

1. **Schema Ingestion & Knowledge Indexing:**
   The `VectorStore` indexes dataset schema (columns, datatypes) along with custom business context rules (e.g., target profit margin formula).
2. **Context Retrieval:**
   When a user asks a question, TF-IDF + cosine similarity retrieves the top relevant context snippets.
3. **Augmented Prompting:**
   The system prompts the LLM with the context, schema, and question, requesting a single valid Pandas expression.
4. **Execution:**
   `run_pandas_expression` runs the expression against the in-memory dataframe and returns the real computed output.

---

## 🔒 Security & Safe Execution

Executing LLM-generated code requires strict sandboxing. `query_executor.py` runs expressions with:
- `__builtins__` stripped to safe math/aggregation functions (`len`, `sum`, `min`, `max`, `round`).
- Strict isolation: `import`, `exec`, `open`, `os`, and `sys` are inaccessible.
- Scoped bindings limited strictly to the target `df` and `pd`.

---

## 🔮 Extending This Project

- [ ] **Vector Database**: Connect FAISS, ChromaDB, or Pinecone for scaling to thousands of context documents.
- [ ] **Conversation History**: Add multi-turn dialogue memory for conversational refinements.
- [ ] **REST API**: Wrap in FastAPI or Flask to serve analytics via web endpoints.
- [ ] **Interactive UI**: Connect with Streamlit or Gradio for interactive dataset dashboards.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
