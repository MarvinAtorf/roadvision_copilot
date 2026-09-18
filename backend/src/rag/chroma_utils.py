import contextlib

import chromadb


def reset_collection(chroma_client: chromadb.ClientAPI, name: str) -> chromadb.Collection:
    """
    Delete a ChromaDB collection if it exists, then create it fresh.

    Every build_*_index() function must call this before adding documents —
    get_or_create_collection() alone only ever ADDS to what's there, so
    re-running an indexer without this duplicates every entry instead of
    replacing them.
    """
    with contextlib.suppress(Exception):
        chroma_client.delete_collection(name)
    return chroma_client.get_or_create_collection(name)
