import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict

from agents.qa_agent import run_qa_agent
from agents.reviewer_agent import run_reviewer_agent
from agents.planner_agent import run_planner_agent
from rag.search import search_codebase

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
class SupervisorState(TypedDict):
    question: str
    context:  str   # RAG context injected here
    route:    str
    answer:   str
    agent:    str

# ─────────────────────────────────────────────
# NODE 1: supervisor — routes + fetches RAG context
# ─────────────────────────────────────────────
def supervisor_node(state: SupervisorState) -> SupervisorState:
    print(f"\n🧠 [Supervisor] Analysing question...")

    prompt = f"""You are a supervisor routing questions to specialist agents.

Agents available:
- qa        → answers questions about how code works
- reviewer  → reviews code quality, finds bugs, security issues
- planner   → plans refactors, architectural changes, step-by-step improvements

Question: "{state['question']}"

Reply with ONLY one word — the agent name: qa, reviewer, or planner"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5
    )

    route = response.choices[0].message.content.strip().lower()
    if route not in ["qa", "reviewer", "planner"]:
        route = "qa"

    print(f"   → Routing to: [{route.upper()} AGENT]")
    print(f"   → Fetching codebase context via RAG...")

    # RAG: pull relevant chunks from ChromaDB
    chunks = search_codebase(state["question"], n_results=3)
    context = "\n\n".join([
        f"# File: {c['source']} | Function: {c['function']}\n{c['text']}"
        for c in chunks
    ])

    state["route"]   = route
    state["context"] = context
    return state

# ─────────────────────────────────────────────
# NODES 2-4: specialist agents (now receive context)
# ─────────────────────────────────────────────
def qa_node(state: SupervisorState) -> SupervisorState:
    state["answer"] = run_qa_agent(state["question"], state["context"])
    state["agent"]  = "qa"
    return state

def reviewer_node(state: SupervisorState) -> SupervisorState:
    state["answer"] = run_reviewer_agent(state["question"], state["context"])
    state["agent"]  = "reviewer"
    return state

def planner_node(state: SupervisorState) -> SupervisorState:
    state["answer"] = run_planner_agent(state["question"], state["context"])
    state["agent"]  = "planner"
    return state

# ─────────────────────────────────────────────
# ROUTING FUNCTION
# ─────────────────────────────────────────────
def route_to_agent(state: SupervisorState) -> str:
    return state["route"]

# ─────────────────────────────────────────────
# BUILD GRAPH
# ─────────────────────────────────────────────
def build_supervisor_graph():
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("qa",         qa_node)
    graph.add_node("reviewer",   reviewer_node)
    graph.add_node("planner",    planner_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges("supervisor", route_to_agent, {
        "qa":       "qa",
        "reviewer": "reviewer",
        "planner":  "planner"
    })

    graph.add_edge("qa",       END)
    graph.add_edge("reviewer", END)
    graph.add_edge("planner",  END)

    return graph.compile()

# ─────────────────────────────────────────────
# PUBLIC RUNNER — used by FastAPI
# ─────────────────────────────────────────────
def run_supervisor(question: str) -> dict:
    graph = build_supervisor_graph()
    result = graph.invoke({
        "question": question,
        "context":  "",
        "route":    "",
        "answer":   "",
        "agent":    ""
    })
    return {
        "answer":   result["answer"],
        "agent":    result["agent"],
        "question": question
    }

# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "how does user login work?",
        "are there any security issues in the auth code?",
        "how would I refactor the database connection to use a connection pool?",
    ]
    for question in questions:
        print(f"\n{'='*60}")
        print(f"❓ {question}")
        print(f"{'='*60}")
        result = run_supervisor(question)
        print(f"\n📝 ANSWER:\n{result['answer']}")
        print(f"\n📊 Routed to: [{result['agent'].upper()} AGENT]")