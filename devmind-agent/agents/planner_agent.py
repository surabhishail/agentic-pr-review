import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_planner_agent(question: str, context: str = "") -> str:
    prompt = f"""You are a principal engineer planning code changes.
Produce a clear numbered refactor plan:
- Each step must be actionable (file to edit, what to change, why)
- Include test checkpoints after risky steps
- Flag dependencies between steps

Codebase context:
{context if context else "No context available."}

Request: {question}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response.choices[0].message.content