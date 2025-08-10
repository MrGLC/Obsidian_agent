"""FastAPI server exposing Obsidian vector store operations.

The server indexes an Obsidian vault into a Chroma vector store on startup.
Endpoints are decorated for Model Context Protocol (MCP) compatibility so that
LLM agents can call them as tools.
"""

import os
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from obsidian_vectorstore import build_vector_store, delete_note

try:  # Optional MCP decorator for tool exposure
    from mcp import tool  # type: ignore
except Exception:  # pragma: no cover - decorator is optional
    def tool(*args, **kwargs):  # type: ignore
        def decorator(func):
            return func
        return decorator


VAULT_PATH = os.environ.get("OBSIDIAN_VAULT")
if not VAULT_PATH:
    raise RuntimeError("OBSIDIAN_VAULT environment variable not set")

store = build_vector_store(VAULT_PATH)

app = FastAPI(title="Obsidian Vector Store Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    uuid: str
    source: str
    content: str


@app.get("/search", response_model=List[SearchResult])
@tool()
def search(q: str, k: int = 4) -> List[SearchResult]:
    """Return the top ``k`` chunks most similar to ``q``."""
    results = store.similarity_search(q, k=k)
    return [
        SearchResult(
            uuid=doc.metadata.get("uuid", ""),
            source=doc.metadata.get("source", ""),
            content=doc.page_content,
        )
        for doc in results
    ]


@app.post("/reindex")
@tool()
def reindex() -> dict:
    """Rebuild the vector store from the vault."""
    global store
    store = build_vector_store(VAULT_PATH)
    ids = store.get(include=[])["ids"]
    return {"documents": len(ids)}


@app.delete("/note/{note_uuid}")
@tool()
def remove_note(note_uuid: str) -> dict:
    """Delete all embeddings associated with ``note_uuid``."""
    delete_note(store, note_uuid)
    return {"deleted": note_uuid}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
