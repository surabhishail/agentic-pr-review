import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from api.schemas import AskRequest, AskResponse
from agents.supervisor_agent import run_supervisor

app = FastAPI(
    title="DevMind",
    description="Agentic code intelligence — routes to specialist agents with RAG context",
    version="0.1.0"
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "devmind"}

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = run_supervisor(request.question)
    return AskResponse(
        answer=result["answer"],
        agent=result["agent"],
        question=result["question"]
    )