import re

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.ingest_stvo import DEFAULT_PDF_PATH, chunk_by_paragraph, extract_stvo_text

CHROMA_COLLECTION_NAME = "stvo_full"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Matches explicit paragraph references in a question, e.g. "§ 53" or "§53a"
PARAGRAPH_REFERENCE_PATTERN = re.compile(
    r"(?:§|[Pp]aragraph(?:en)?|[Pp]aragraf(?:en)?)\s*(\d+[a-z]?)"
)

# Load the embedding model once at import time instead of on every request
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)


def build_documents(pdf_path=DEFAULT_PDF_PATH) -> list[Document]:
    text = extract_stvo_text(pdf_path)
    chunks = chunk_by_paragraph(text)
    return [
        Document(
            text=chunk["text"],
            metadata={"paragraph": chunk["header"], "source": "StVO"},
            doc_id=chunk["header"],
        )
        for chunk in chunks
    ]


def build_index(chroma_client: chromadb.ClientAPI) -> VectorStoreIndex:
    documents = build_documents()
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)


def _find_explicit_paragraph_matches(
    chroma_client: chromadb.ClientAPI, question: str
) -> list[dict]:
    """If the question explicitly names a paragraph number (e.g. "§ 53"),
    look it up directly via metadata instead of relying on semantic similarity,
    which can miss administrative/procedural paragraphs that rank poorly."""
    referenced_numbers = PARAGRAPH_REFERENCE_PATTERN.findall(question)
    if not referenced_numbers:
        return []

    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    all_entries = collection.get(include=["metadatas", "documents"])

    matches = []
    for metadata, doc_text in zip(all_entries["metadatas"], all_entries["documents"], strict=True):
        paragraph = metadata.get("paragraph", "")
        # normalize non-breaking spaces from the PDF extraction
        normalized = paragraph.replace("\xa0", " ")
        for number in referenced_numbers:
            if normalized.startswith((f"§ {number} ", f"§ {number},")):
                matches.append({"paragraph": paragraph, "text": doc_text, "score": 1.0})
    return matches


def retrieve_context(
    chroma_client: chromadb.ClientAPI, question: str, top_k: int = 3
) -> list[dict]:
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=top_k)
    semantic_results = retriever.retrieve(question)
    semantic_matches = [
        {"paragraph": r.metadata.get("paragraph"), "text": r.text, "score": r.score}
        for r in semantic_results
    ]

    explicit_matches = _find_explicit_paragraph_matches(chroma_client, question)

    # merge, explicit matches first, no duplicate paragraphs
    seen_paragraphs = set()
    combined = []
    for match in explicit_matches + semantic_matches:
        if match["paragraph"] in seen_paragraphs:
            continue
        seen_paragraphs.add(match["paragraph"])
        combined.append(match)

    return combined


def query_index(chroma_client: chromadb.ClientAPI, question: str, top_k: int = 3):
    results = retrieve_context(chroma_client, question, top_k=top_k)
    for r in results:
        print(f"[{r['paragraph']}] score={r['score']:.3f}")
        print(r["text"][:200])
        print("---")


def main():
    import os

    chroma_host = os.getenv("CHROMA_HOST", "chromadb")
    chroma_port = int(os.getenv("CHROMA_PORT", 8000))
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

    print(f"Indexing StVO into ChromaDB collection '{CHROMA_COLLECTION_NAME}'...")
    build_index(client)
    print("Done.")
    query_index(client, "Wie schnell darf ich innerorts fahren?")


if __name__ == "__main__":
    main()
