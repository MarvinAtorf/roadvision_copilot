import json
import re
from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

SIGN_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "stvo_sign_catalog.json"
SIGN_COLLECTION_NAME = "stvo_signs"

# Load the embedding model once at import time (same reasoning as rag/index.py:
# loading it per-request blocked the whole API during the HuggingFace download).

SIGN_REFERENCE_PATTERN = re.compile(r"(?:Zeichen|Schild)\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def load_sign_catalog() -> list[dict]:
    """
    Loads a catalog of signs from a predefined file path.

    This function reads a JSON file containing a catalog of signs and parses it
    into a list of dictionaries. The file path is defined by the constant
    SIGN_CATALOG_PATH.

    :return: A list of dictionaries representing the sign catalog.
    :rtype: list[dict]
    """
    with SIGN_CATALOG_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def build_sign_documents() -> list[Document]:
    catalog = load_sign_catalog()
    documents = []

    for entry in catalog:
        text = f"Zeichen {entry['sign_number']}: {entry['official_name']}. {entry['explanation']}"

        documents.append(
            Document(
                text=text,
                metadata={"sign_number": entry["sign_number"], "source": "StVO-Anlage"},
                doc_id=entry["sign_number"],
            )
        )

    return documents


def build_sign_index(chroma_client: chromadb.ClientAPI) -> VectorStoreIndex:
    documents = build_sign_documents()
    collection = chroma_client.get_or_create_collection(SIGN_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)


def _find_explicit_sign_matches(chroma_client: chromadb.ClientAPI, question: str) -> list[dict]:
    """If the question explicitly names a sign number (e.g. "Zeichen 274"),
    look it up directly via metadata instead of relying on semantic similarity."""
    referenced_numbers = SIGN_REFERENCE_PATTERN.findall(question)
    if not referenced_numbers:
        return []

    collection = chroma_client.get_or_create_collection(SIGN_COLLECTION_NAME)
    all_entries = collection.get(include=["metadatas", "documents"])

    matches = []
    for metadata, doc_text in zip(all_entries["metadatas"], all_entries["documents"], strict=True):
        sign_number = metadata.get("sign_number", "")
        if sign_number in referenced_numbers:
            matches.append({"sign_number": sign_number, "text": doc_text, "score": 1.0})

    return matches


def retrieve_sign_context(
    chroma_client: chromadb.ClientAPI, question: str, top_k: int = 3
) -> list[dict]:
    collection = chroma_client.get_or_create_collection(SIGN_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=top_k)
    semantic_results = retriever.retrieve(question)
    semantic_matches = [
        {"sign_number": r.metadata.get("sign_number"), "text": r.text, "score": r.score}
        for r in semantic_results
    ]

    explicit_matches = _find_explicit_sign_matches(chroma_client, question)

    seen_signs = set()
    combined = []
    for match in explicit_matches + semantic_matches:
        if match["sign_number"] in seen_signs:
            continue
        seen_signs.add(match["sign_number"])
        combined.append(match)

    return combined
