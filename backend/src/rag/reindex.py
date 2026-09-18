"""
Shared registry of RAG collections and a guard that builds only the ones
still empty — used by main.py's startup hook (skip already-built
collections) and by scripts/reindex_all.py's manual "rebuild everything"
CLI entry point.
"""

import chromadb

from rag.bussgeldkatalog_index import BUSSGELDKATALOG_COLLECTION_NAME, build_bussgeldkatalog_index
from rag.index import CHROMA_COLLECTION_NAME as STVO_COLLECTION_NAME
from rag.index import build_index
from rag.sign_index import SIGN_COLLECTION_NAME, build_sign_index

COLLECTION_BUILDERS = {
    STVO_COLLECTION_NAME: build_index,
    SIGN_COLLECTION_NAME: build_sign_index,
    BUSSGELDKATALOG_COLLECTION_NAME: build_bussgeldkatalog_index,
}


def ensure_indexes(chroma_client: chromadb.ClientAPI) -> None:
    """
    Build any of the three RAG collections that don't already have data.

    Runs on every API startup, so a completely fresh ChromaDB volume ends
    up populated without a manual step — but an already-indexed collection
    is left untouched, since re-embedding everything on every container
    restart would needlessly slow down every `docker compose up`.
    """
    for name, build_fn in COLLECTION_BUILDERS.items():
        collection = chroma_client.get_or_create_collection(name)
        if collection.count() > 0:
            print(f"{name}: already indexed ({collection.count()} entries), skipping.")
            continue
        print(f"{name}: empty, indexing now...")
        build_fn(chroma_client)
        print(f"{name}: done.")
