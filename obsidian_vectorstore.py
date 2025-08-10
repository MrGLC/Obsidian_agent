"""Utilities for indexing an Obsidian vault in a Chroma vector store.

This module extracts the prototyping code that lived in the notebooks and
turns it into reusable functions. Notes from an Obsidian vault are loaded,
optionally split into chunks, and written to a persistent Chroma vector store.
Each note is associated with a UUID that is stored both in the note's
frontmatter and in the vector store so the note can later be updated or
removed.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ObsidianLoader


def _ensure_uuid(file_path: Path, metadata: dict) -> str:
    """Return the note UUID, inserting one into the file if missing."""
    if "uuid" in metadata and metadata["uuid"]:
        return metadata["uuid"]

    note_uuid = str(uuid.uuid4())
    text = file_path.read_text(encoding="utf-8")

    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            frontmatter = yaml.safe_load(text[3:end]) or {}
            frontmatter["uuid"] = note_uuid
            new_frontmatter = yaml.safe_dump(frontmatter, sort_keys=False)
            updated = f"---\n{new_frontmatter}---{text[end+4:]}"
        else:
            # Malformed frontmatter; prepend a new block
            updated = f"---\nuuid: {note_uuid}\n---\n{text}"
    else:
        updated = f"---\nuuid: {note_uuid}\n---\n{text}"

    file_path.write_text(updated, encoding="utf-8")
    return note_uuid


def load_and_split(vault_path: str) -> List[Tuple[str, Document]]:
    """Load markdown notes from an Obsidian vault and split them.

    Returns a list of tuples ``(chunk_id, Document)`` where ``chunk_id`` is
    unique and encodes the originating note UUID and chunk index.
    """
    loader = ObsidianLoader(vault_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )

    processed: List[Tuple[str, Document]] = []
    for doc in docs:
        source_path = doc.metadata.get("source")
        if not source_path:
            # Skip documents without a file source
            continue
        file_path = Path(source_path)
        note_uuid = _ensure_uuid(file_path, doc.metadata)

        for idx, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata["uuid"] = note_uuid
            chunk.metadata["source"] = source_path
            chunk_id = f"{note_uuid}:{idx}"
            processed.append((chunk_id, chunk))
    return processed


def build_vector_store(
    vault_path: str,
    persist_directory: str = "./chroma_obsidian_db",
    collection_name: str = "obsidian_docs",
) -> Chroma:
    """Create (or update) a Chroma vector store from an Obsidian vault."""
    docs_with_ids = load_and_split(vault_path)
    if not docs_with_ids:
        raise ValueError("No documents were loaded from the vault")

    ids, documents = zip(*docs_with_ids)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    store.add_documents(list(documents), ids=list(ids))
    return store


def delete_note(store: Chroma, note_uuid: str) -> None:
    """Remove all vector entries associated with ``note_uuid``."""
    store.delete(where={"uuid": note_uuid})


def index_note(store: Chroma, file_path: Path) -> str:
    """Index a single markdown note into the vector store.

    The note's frontmatter is ensured to contain a UUID. Any existing
    embeddings for that UUID are expected to be removed by the caller.
    Returns the note UUID.
    """

    text = file_path.read_text(encoding="utf-8")
    metadata = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            metadata = yaml.safe_load(text[3:end]) or {}
            body = text[end + 4 :]

    note_uuid = _ensure_uuid(file_path, metadata)

    doc = Document(page_content=body, metadata={"source": str(file_path), "uuid": note_uuid})
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = splitter.split_documents([doc])
    ids = [f"{note_uuid}:{i}" for i in range(len(chunks))]
    for chunk in chunks:
        chunk.metadata["uuid"] = note_uuid
        chunk.metadata["source"] = str(file_path)
    store.add_documents(chunks, ids=ids)
    return note_uuid


__all__ = [
    "build_vector_store",
    "delete_note",
    "index_note",
    "load_and_split",
]
