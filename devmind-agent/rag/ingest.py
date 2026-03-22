"""
RAG Pipeline — Step 1: Ingest
Reads all .py files from sample_codebase,
chunks them by function, embeds and stores in ChromaDB.
Run once: python3 -m rag.ingest
"""

import os
import chromadb
from chromadb.utils import embedding_functions

# ── Point to your codebase ────────────────────────────────────────────────────
CODEBASE_PATH = "sample_codebase"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "devmind_codebase"

# ── Embedding function (runs locally, no API key needed) ──────────────────────
# Uses all-MiniLM-L6-v2 — small, fast, free model (~80MB, downloads once)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def read_python_files(path):
    """Walk the codebase and return list of (filepath, content)"""
    files = []
    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".py"):
                filepath = os.path.join(root, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                files.append((filepath, content))
    return files

def chunk_by_function(filepath, content):
    chunks = []
    lines = content.split("\n")
    current_chunk = []
    current_name = "module_level"

    for line in lines:
        # Only split on def/class, NOT on decorators
        # Decorators stay attached to their function
        stripped = line.strip()
        if (line.startswith("def ") or line.startswith("class ")) and current_chunk:
            # Save previous chunk
            chunks.append({
                "text": "\n".join(current_chunk),
                "source": filepath,
                "function": current_name
            })
            current_name = stripped
            current_chunk = [line]
        else:
            current_chunk.append(line)

    # Last chunk
    if current_chunk:
        chunks.append({
            "text": "\n".join(current_chunk),
            "source": filepath,
            "function": current_name
        })

    return chunks
def ingest():
    # 1. Set up ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete collection if it exists (fresh start)
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    # 2. Read all Python files
    files = read_python_files(CODEBASE_PATH)
    print(f"\n📁 Found {len(files)} Python files")

    # 3. Chunk all files
    all_chunks = []
    for filepath, content in files:
        chunks = chunk_by_function(filepath, content)
        all_chunks.extend(chunks)
        print(f"  → {filepath}: {len(chunks)} chunks")

    print(f"\n✂️  Total chunks: {len(all_chunks)}")

    # 4. Store in ChromaDB (embeds automatically)
    collection.add(
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"], "function": c["function"]} for c in all_chunks],
        ids=[f"chunk_{i}" for i in range(len(all_chunks))]
    )

    print(f"\n✅ Stored {len(all_chunks)} chunks in ChromaDB at {CHROMA_DB_PATH}")
    print("Ready to search!\n")

if __name__ == "__main__":
    ingest()