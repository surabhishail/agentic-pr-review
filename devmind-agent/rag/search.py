"""
RAG Pipeline — Step 2: Search
Given a query, find the most relevant code chunks from ChromaDB.
"""

import chromadb
from chromadb.utils import embedding_functions

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "devmind_codebase"

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def search_codebase(query, n_results=3):
    """
    Search ChromaDB for chunks most relevant to the query.
    Returns top n_results chunks with their source file info.
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "function": results["metadatas"][0][i]["function"],
            "distance": results["distances"][0][i]  # lower = more similar
        })

    return chunks


if __name__ == "__main__":
    # Test it directly
    query = "how does user login work?"
    print(f"\n🔍 Query: '{query}'\n")

    chunks = search_codebase(query)

    for i, chunk in enumerate(chunks):
        print(f"--- Result {i+1} ---")
        print(f"File:     {chunk['source']}")
        print(f"Function: {chunk['function']}")
        print(f"Distance: {chunk['distance']:.4f}  (lower = more relevant)")
        print(f"Code:\n{chunk['text'][:300]}...")
        print()