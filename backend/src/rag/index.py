import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from rag.ingest_stvo import DEFAULT_PDF_PATH, chunk_by_paragraph, extract_stvo_text

CHROMA_COLLECTION_NAME = "stvo_full"
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


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
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    documents = build_documents()
    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)


def query_index(chroma_client: chromadb.ClientAPI, question: str, top_k: int = 3):
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(question)
    for r in results:
        print(f"[{r.metadata.get('paragraph')}] score={r.score:.3f}")
        print(r.text[:200])
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


def retrieve_context(
    chroma_client: chromadb.ClientAPI, question: str, top_k: int = 3
) -> list[dict]:
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

    collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(question)
    return [
        {"paragraph": r.metadata.get("paragraph"), "text": r.text, "score": r.score}
        for r in results
    ]


if __name__ == "__main__":
    main()
