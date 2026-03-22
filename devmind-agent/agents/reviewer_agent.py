import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_reviewer_agent(question: str, context: str = "") -> str:
    prompt = f"""You are a senior security-focused code reviewer.
Analyse the code context and find:
- Bugs and logic errors
- Security vulnerabilities (injection, auth issues, hardcoded secrets)
- Missing error handling
- Performance problems

Be specific — name the file, line pattern, and the fix.

Codebase context:
{context if context else "No context available."}

Review request: {question}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )
    return response.choices[0].message.content