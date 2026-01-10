"""Document indexing module using LlamaIndex and pgvector."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import psycopg2
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from .config import config


@dataclass
class IndexResult:
    """Result of document indexing operation."""
    total_chunks: int
    indexed_chunks: int
    skipped_chunks: int
    files_processed: List[str]


def get_db_connection():
    """Create a database connection."""
    return psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        database=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
    )


def compute_chunk_hash(content: str) -> str:
    """Compute SHA-256 hash of chunk content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def ensure_chunk_hash_index():
    """Create index on chunk_hash if it doesn't exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunk_hash 
                ON data_documents ((metadata_->>'chunk_hash'))
            """)
            conn.commit()
    except Exception:
        # Table might not exist yet, index will be created later
        pass
    finally:
        conn.close()


def get_existing_chunk_hashes() -> Set[str]:
    """Retrieve existing chunk hashes from pgvector table metadata."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Query chunk_hash from metadata_ JSON field (uses idx_chunk_hash index)
            cur.execute("""
                SELECT metadata_->>'chunk_hash' 
                FROM data_documents 
                WHERE metadata_->>'chunk_hash' IS NOT NULL
            """)
            return {row[0] for row in cur.fetchall()}
    except Exception:
        # Table might not exist yet
        return set()
    finally:
        conn.close()


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


def index_documents(file_paths: List[str]) -> IndexResult:
    """
    Index documents into pgvector store with chunk-level deduplication.

    Args:
        file_paths: List of file paths to index

    Returns:
        IndexResult with details about indexed and skipped chunks
    """
    config.validate()

    # Ensure index exists for fast hash lookups
    ensure_chunk_hash_index()

    # Get existing chunk hashes for deduplication
    existing_hashes = get_existing_chunk_hashes()

    # Create text splitter
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # Process files and collect chunks
    all_nodes = []
    total_chunks = 0
    skipped_chunks = 0
    files_processed = []

    for fp in file_paths:
        path = Path(fp)
        filename = path.name
        files_processed.append(filename)

        content = path.read_text(encoding="utf-8")
        doc = Document(
            text=content,
            metadata={"filename": filename, "filepath": str(path)},
        )

        # Split document into nodes (chunks)
        nodes = splitter.get_nodes_from_documents([doc])
        total_chunks += len(nodes)

        # Filter chunks by hash
        for node in nodes:
            chunk_hash = compute_chunk_hash(node.text)

            if chunk_hash in existing_hashes:
                skipped_chunks += 1
            else:
                # Add hash to node metadata (will be saved with the node)
                node.metadata["chunk_hash"] = chunk_hash
                all_nodes.append(node)
                # Add to existing_hashes to handle duplicates within batch
                existing_hashes.add(chunk_hash)

    indexed_chunks = len(all_nodes)

    # If no new chunks, return early
    if not all_nodes:
        return IndexResult(
            total_chunks=total_chunks,
            indexed_chunks=0,
            skipped_chunks=skipped_chunks,
            files_processed=files_processed,
        )

    # Get vector store and embed model
    vector_store = get_vector_store()
    embed_model = get_embed_model()

    # Create storage context with pgvector
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Build index from nodes directly (chunk_hash is saved in metadata automatically)
    VectorStoreIndex(
        nodes=all_nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    return IndexResult(
        total_chunks=total_chunks,
        indexed_chunks=indexed_chunks,
        skipped_chunks=skipped_chunks,
        files_processed=files_processed,
    )


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
