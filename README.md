# Agentic PR Review & Codebase Intelligence

An AI agent system that reviews pull requests and answers questions about a codebase — no human in the loop.

---

## What it does

**PR Review Agent**
Submit a pull request diff and get back a structured review — logic issues, edge cases, naming problems, missing test coverage. Not a linter. Closer to a second pair of eyes that reads the intent, not just the syntax.

**Codebase Q&A Agent**
Ask natural language questions about any codebase. "Where is authentication handled?" "What does this service own?" Answers come from a RAG pipeline over an indexed vector store — not keyword search, not grep.

**Multi-Agent Supervisor**
A LangGraph graph that routes between agents, manages state across turns, and decides when a task is complete. Each agent is independent; the supervisor coordinates. Stateful, not stateless.

---

## Architecture

```
User input
    │
    ▼
Supervisor (LangGraph)
    ├──▶ PR Review Agent     ──▶ structured feedback on diff
    └──▶ Codebase Q&A Agent  ──▶ RAG retrieval + grounded answer
```

---

## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Vector store | ChromaDB |
| LLM inference | Groq (llama3) |
| API | FastAPI |
| UI | Streamlit |
| Language | Python 3.11 |

---

## Getting started

```bash
git clone https://github.com/surabhishail/agentic-pr-review
cd agentic-pr-review

pip install -r requirements.txt

# Add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Start the API
uvicorn api.main:app --reload

# Or run the Streamlit UI
streamlit run ui/app.py
```

---

## Project structure

```
agentic-pr-review/
├── agents/           # PR review + codebase Q&A agent logic
├── rag/              # ChromaDB ingestion and retrieval
├── api/              # FastAPI REST endpoints
├── ui/               # Streamlit chat interface
├── sample_codebase/  # Sample repo for RAG indexing
├── chat.py           # CLI interface
└── run.py            # Entry point
```

---

## Why I built this

Most of my career has been in distributed systems — high-throughput pipelines, multi-region reliability, the kind of work where something always breaks in a way you didn't predict.

Building this was about understanding the equivalent failure modes in agentic systems: where agents hand off state incorrectly, where RAG retrieval silently returns the wrong context, where the supervisor gets stuck in a loop.

The patterns aren't that different from microservice debugging. The tooling is just newer.
