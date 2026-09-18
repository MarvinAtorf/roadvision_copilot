"""
Rebuild all three RAG collections (StVO paragraphs, StVO signs, and the
Bußgeldkatalog) in one run — unconditionally, regardless of whether a
collection already has data. Use this after changing a source PDF.

For the guarded "only if missing" version that runs automatically on API
startup, see rag.reindex.ensure_indexes().

Usage: python -m scripts.reindex_all
"""

import os

import chromadb
from rag.reindex import COLLECTION_BUILDERS


def main():
    chroma_host = os.getenv("CHROMA_HOST", "chromadb")
    chroma_port = int(os.getenv("CHROMA_PORT", 8000))
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    for name, build_fn in COLLECTION_BUILDERS.items():
        print(f"Indexing {name}...")
        build_fn(client)
        print("Done.")

    print("All three collections reindexed.")


if __name__ == "__main__":
    main()
