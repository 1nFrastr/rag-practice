"""Document indexing module using LlamaIndex and pgvector."""

from pathlib import Path
from typing import List

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from .config import config


def get_embed_model() -> OpenAIEmbedding:
    """Create OpenAI embedding model via OpenRouter."""
    return OpenAIEmbedding(
        model=config.EMBEDDING_MODEL,
        api_base=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
    )


def get_vector_store() -> PGVectorStore:
    """Create pgvector store connection."""
    return PGVectorStore.from_params(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        database=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        table_name="documents",
        embed_dim=config.EMBEDDING_DIM,
    )


def load_document(file_path: str) -> Document:
    """Load a single document from file path."""
    path = Path(file_path)
    content = path.read_text(encoding="utf-8")
    return Document(
        text=content,
        metadata={"filename": path.name, "filepath": str(path)},
    )


def index_documents(file_paths: List[str]) -> VectorStoreIndex:
    """
    Index documents into pgvector store.

    Args:
        file_paths: List of file paths to index

    Returns:
        VectorStoreIndex for querying
    """
    config.validate()

    # Load documents
    documents = [load_document(fp) for fp in file_paths]

    # Create text splitter
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # Get vector store and embed model
    vector_store = get_vector_store()
    embed_model = get_embed_model()

    # Create storage context with pgvector
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    return index


def get_existing_index() -> VectorStoreIndex:
    """Load existing index from pgvector store."""
    config.validate()

    vector_store = get_vector_store()
    embed_model = get_embed_model()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    return index
