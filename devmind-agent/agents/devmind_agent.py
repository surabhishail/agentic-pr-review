import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from groq import Groq
from rag.search import search_codebase
from agents.state import DevMindState

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def receive_input(state: DevMindState) -> DevMindState:
    print(f"\n🧠 [Agent] Question: '{state['question']}'")
    state["search_query"] = state["question"]
    state["retry_count"] = 0
    state["enough_context"] = False
    return state

def search_codebase_node(state: DevMindState) -> DevMindState:
    print(f"\n🔍 [Agent] Searching for: '{state['search_query']}'")
    chunks = search_codebase(state["search_query"], n_results=3)
    state["retrieved_chunks"] = chunks
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Chunk {i+1} from {chunk['source']} ---\n"
        context += chunk["text"] + "\n"
    state["context"] = context
    for c in chunks:
        print(f"   · {c['source']} | distance: {c['distance']:.4f}")
    return state

def evaluate_context(state: DevMindState) -> DevMindState:
    chunks = state["retrieved_chunks"]
    if not chunks:
        state["enough_context"] = False
        return state
    top_distance = chunks[0]["distance"]
    print(f"\n⚖️  [Agent] Top chunk distance: {top_distance:.4f}")
    if top_distance < 1.3:
        state["enough_context"] = True
        print(f"   → ✅ Good context — answering")
    elif state["retry_count"] >= 2:
        state["enough_context"] = True
        print(f"   → ⚠️  Max retries — answering with what we have")
    else:
        state["enough_context"] = False
        print(f"   → ❌ Weak context — retrying")
    return state

def search_broader(state: DevMindState) -> DevMindState:
    state["retry_count"] += 1
    print(f"\n🔄 [Agent] Retry #{state['retry_count']} — rephrasing query...")
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f'Rephrase "{state["question"]}" into 2-3 broader keywords for searching a Python codebase. Return ONLY the keywords.'}],
        max_tokens=50
    )
    state["search_query"] = response.choices[0].message.content.strip()
    print(f"   → New query: '{state['search_query']}'")
    return search_codebase_node(state)

def generate_answer(state: DevMindState) -> DevMindState:
    print(f"\n💬 [Agent] Generating answer...")
    prompt = f"""You are DevMind, an AI assistant that answers questions about a codebase.
Use ONLY the code chunks below. If the answer isn't there, say "I don't see that in the codebase."

CODE CHUNKS:
{state['context']}

QUESTION: {state['question']}
ANSWER:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    state["answer"] = response.choices[0].message.content
    return state

def route_after_evaluation(state: DevMindState) -> str:
    return "generate_answer" if state["enough_context"] else "search_broader"

def build_devmind_graph():
    graph = StateGraph(DevMindState)
    graph.add_node("receive_input", receive_input)
    graph.add_node("search_codebase", search_codebase_node)
    graph.add_node("evaluate_context", evaluate_context)
    graph.add_node("search_broader", search_broader)
    graph.add_node("generate_answer", generate_answer)
    graph.set_entry_point("receive_input")
    graph.add_edge("receive_input", "search_codebase")
    graph.add_edge("search_codebase", "evaluate_context")
    graph.add_conditional_edges("evaluate_context", route_after_evaluation, {
        "generate_answer": "generate_answer",
        "search_broader": "search_broader"
    })
    graph.add_edge("search_broader", "evaluate_context")
    graph.add_edge("generate_answer", END)
    return graph.compile()

if __name__ == "__main__":
    agent = build_devmind_graph()
    questions = [
        "how does user login work?",
        "how are passwords stored?",
        "what happens when a user registers?",
    ]
    for question in questions:
        print(f"\n{'='*60}")
        print(f"❓ {question}")
        print(f"{'='*60}")
        final_state = agent.invoke({
            "question": question,
            "search_query": "",
            "retrieved_chunks": [],
            "context": "",
            "answer": "",
            "retry_count": 0,
            "enough_context": False
        })
        print(f"\n📝 ANSWER:\n{final_state['answer']}")
        print(f"📊 {len(final_state['retrieved_chunks'])} chunks | {final_state['retry_count']} retries")
