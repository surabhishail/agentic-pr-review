"""
RAG Pipeline — Step 3: Ask
Combines search + LLM to answer questions about the codebase.
Run: python3 -m rag.ask
"""

import os
from groq import Groq
from dotenv import load_dotenv
from rag.search import search_codebase

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_codebase(question):
    # 1. Retrieve relevant chunks
    chunks = search_codebase(question, n_results=3)

    # 2. Build context from chunks
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n--- Chunk {i+1} from {chunk['source']} ---\n"
        context += chunk["text"]
        context += "\n"

    # 3. Build prompt — this is RAG's core
    prompt = f"""You are DevMind, an AI assistant that answers questions about a codebase.

Use ONLY the code chunks below to answer. Do not guess or use outside knowledge.
If the answer isn't in the chunks, say "I don't see that in the codebase."

CODE CHUNKS:
{context}

QUESTION: {question}

ANSWER:"""

    # 4. Call LLM with context + question
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    questions = [
        "how does user login work?",
        "how are passwords stored?",
        "what happens when a user registers?",
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"Q: {q}")
        print(f"{'='*50}")
        print(ask_codebase(q))