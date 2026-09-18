import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

import rag.embeddings  # noqa: F401 - import side effect sets Settings.embed_model
from rag.chroma_utils import reset_collection
from rag.ingest_bussgeldkatalog import (
    DEFAULT_PDF_PATH,
    chunk_by_bussgeld_category,
    extract_bussgeld_text,
)

BUSSGELDKATALOG_COLLECTION_NAME = "bussgeldkatalog_pkw"


def build_bussgeldkatalog_documents() -> list[Document]:
    """
    Extract and chunk the Pkw section of the Bußgeldkatalog PDF into one
    Document per category (Abstand, Rote Ampel, Vorfahrt, ...).

    :return: A list of Documents, one per Bußgeldkatalog category.
    :rtype: list[Document]
    """
    text = extract_bussgeld_text(DEFAULT_PDF_PATH)
    chunks = chunk_by_bussgeld_category(text)

    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                text=chunk["text"],
                metadata={"category": chunk["category"], "source": "Bußgeldkatalog Pkw"},
                # doc_id is the category name itself: unique across the 17
                # categories, and re-indexing overwrites rather than duplicates.
                doc_id=chunk["category"],
            )
        )

    return documents


def build_bussgeldkatalog_index(chroma_client: chromadb.ClientAPI) -> VectorStoreIndex:
    documents = build_bussgeldkatalog_documents()
    collection = reset_collection(chroma_client, BUSSGELDKATALOG_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Documents ARE TextNodes in LlamaIndex — passing them directly as nodes
    # (instead of from_documents()) skips the default SentenceSplitter
    # transformation, so each category stays exactly one embedded chunk
    # instead of being auto-split at ~1024 tokens. That split had silently
    # turned our 17 hand-built category chunks into 23 stored entries, some
    # missing half their fine schedule — bad for the report's "hypothetical
    # fine" lookup, which needs a category's complete text in one retrieval.
    return VectorStoreIndex(nodes=documents, storage_context=storage_context)


def retrieve_bussgeldkatalog_context(
    chroma_client: chromadb.ClientAPI, query: str, top_k: int = 3
) -> list[dict]:
    """
    Retrieve the most relevant Bußgeldkatalog categories for a query.

    Purely semantic — unlike retrieve_sign_context, there is no numbered
    anchor (like a sign number) to look up explicitly in the source text,
    so similarity search over the whole query is the only retrieval path.

    :param chroma_client: Shared ChromaDB client (from app.state).
    :param query: Free-text query (e.g. the user's scenario description).
    :param top_k: Number of categories to retrieve.
    :return: A list of {"category", "text", "score"} dicts.
    :rtype: list[dict]
    """
    collection = chroma_client.get_or_create_collection(BUSSGELDKATALOG_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query)

    return [{"category": r.metadata["category"], "text": r.text, "score": r.score} for r in results]


def main():
    import os

    chroma_host = os.getenv("CHROMA_HOST", "chromadb")
    chroma_port = int(os.getenv("CHROMA_PORT", 8000))
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    print(
        f"Indexing Bußgeldkatalog into ChromaDB collection '{BUSSGELDKATALOG_COLLECTION_NAME}'..."
    )
    build_bussgeldkatalog_index(client)
    print("Done.")

    results = retrieve_bussgeldkatalog_context(client, "Rechts vor links missachtet")
    for r in results:
        print(f"[{r['score']:.3f}] {r['category']}")


if __name__ == "__main__":
    main()
