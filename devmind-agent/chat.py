from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# This list IS the memory — grows with every message
history = [
    {
        "role": "system",
        "content": "You are DevMind, an AI assistant that helps developers understand codebases. Be concise and technical."
    }
]

print("\n🧠 DevMind — Chat")
print("Type 'quit' to exit | 'history' to see message count\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("Bye!")
        break

    if user_input.lower() == "history":
        print(f"  → {len(history)} messages in context\n")
        continue

    # 1. Add user message to history
    history.append({
        "role": "user",
        "content": user_input
    })

    # 2. Send ENTIRE history to Groq every time
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=history        # ← full history, not just latest message
    )

    assistant_reply = response.choices[0].message.content

    # 3. Add assistant reply to history too
    history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    print(f"\nDevMind: {assistant_reply}\n")