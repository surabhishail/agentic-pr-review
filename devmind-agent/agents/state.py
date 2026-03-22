from typing import TypedDict, List

class DevMindState(TypedDict):
    question: str
    search_query: str
    retrieved_chunks: List[dict]
    context: str
    answer: str
    retry_count: int
    enough_context: bool
