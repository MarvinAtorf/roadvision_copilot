from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
