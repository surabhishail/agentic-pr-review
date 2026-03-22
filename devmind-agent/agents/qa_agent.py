import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_qa_agent(question: str, context: str = "") -> str:
    prompt = f"""You are a senior developer explaining code clearly.
Use the codebase context below to answer specifically — reference actual
function names, files, and flow. Do not guess.

Codebase context:
{context if context else "No context available."}

Question: {question}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response.choices[0].message.content