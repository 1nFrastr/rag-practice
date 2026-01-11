"""Hybrid search module implementing BM25 + Vector search with RRF fusion."""

from collections import defaultdict
from typing import Dict, List, Tuple

import psycopg2
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .config import config


def get_db_connection():
    """Create a database connection."""
    return psycopg2.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        database=config.POSTGRES_DB,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
    )


def ensure_fulltext_index():
    """Create GIN index for fulltext search if it doesn't exist."""
    # PGVectorStore uses "data_<table_name>" as the actual table name
    table_name = "data_documents"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Create tsvector column if it doesn't exist
            cur.execute(f"""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name = '{table_name}' 
                        AND column_name = 'text_tsvector'
                    ) THEN
                        ALTER TABLE {table_name} 
                        ADD COLUMN text_tsvector tsvector;
                    END IF;
                END $$;
            """)
            
            # Create GIN index for fulltext search
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_documents_text_tsvector 
                ON {table_name} USING GIN (text_tsvector);
            """)
            
            conn.commit()
    except Exception as e:
        conn.rollback()
        # Table might not exist yet, will be created by PGVectorStore
        print(f"Warning: Could not create fulltext index: {e}")
    finally:
        conn.close()


def update_fulltext_index():
    """Update tsvector column for all documents."""
    # PGVectorStore uses "data_<table_name>" as the actual table name
    table_name = "data_documents"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Update tsvector column using to_tsvector
            # Using 'simple' configuration for better Chinese support
            cur.execute(f"""
                UPDATE {table_name} 
                SET text_tsvector = to_tsvector('simple', COALESCE(text, ''))
                WHERE text_tsvector IS NULL OR text_tsvector != to_tsvector('simple', COALESCE(text, ''));
            """)
            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Warning: Could not update fulltext index: {e}")
    finally:
        conn.close()


def bm25_search(query: str, top_k: int = 20) -> List[Tuple[str, float]]:
    """
    Perform BM25 search using PostgreSQL fulltext search.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        
    Returns:
        List of (node_id, score) tuples
    """
    # PGVectorStore uses "data_<table_name>" as the actual table name
    table_name = "data_documents"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Use ts_rank_cd for BM25-like ranking
            # ts_rank_cd uses cover density ranking which is similar to BM25
            cur.execute(f"""
                SELECT node_id, ts_rank_cd(text_tsvector, query, 32) as rank
                FROM {table_name}, 
                     to_tsquery('simple', regexp_replace(%s, '\s+', ' & ', 'g')) query
                WHERE text_tsvector @@ query
                ORDER BY rank DESC
                LIMIT %s;
            """, (query, top_k))
            
            results = cur.fetchall()
            return [(row[0], float(row[1])) for row in results]
    except Exception as e:
        print(f"Error in BM25 search: {e}")
        return []
    finally:
        conn.close()


def reciprocal_rank_fusion(
    vector_results: List[NodeWithScore],
    bm25_results: List[Tuple[str, float]],
    k: int = 60
) -> List[NodeWithScore]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion (RRF).
    
    Args:
        vector_results: List of NodeWithScore from vector search
        bm25_results: List of (node_id, score) tuples from BM25 search
        k: RRF constant (typically 60)
        
    Returns:
        Combined and ranked list of NodeWithScore
    """
    # Create node_id to NodeWithScore mapping from vector results
    node_map = {node.node.node_id: node for node in vector_results}
    
    # Build RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)
    
    # Add vector search scores
    for rank, node in enumerate(vector_results, start=1):
        node_id = node.node.node_id
        rrf_scores[node_id] += 1.0 / (k + rank)
    
    # Add BM25 search scores
    for rank, (node_id, _) in enumerate(bm25_results, start=1):
        rrf_scores[node_id] += 1.0 / (k + rank)
        # If node_id not in vector results, we need to fetch it from DB
        if node_id not in node_map:
            node_map[node_id] = _fetch_node_from_db(node_id)
    
    # Sort by RRF score (descending)
    sorted_nodes = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Return NodeWithScore objects with RRF scores
    result = []
    for node_id, rrf_score in sorted_nodes:
        if node_id in node_map:
            node_with_score = node_map[node_id]
            # Create new NodeWithScore with RRF score
            result.append(
                NodeWithScore(
                    node=node_with_score.node,
                    score=rrf_score
                )
            )
    
    return result


def _fetch_node_from_db(node_id: str) -> NodeWithScore:
    """Fetch a node from database by node_id."""
    # PGVectorStore uses "data_<table_name>" as the actual table name
    table_name = "data_documents"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT text, metadata_
                FROM {table_name}
                WHERE node_id = %s;
            """, (node_id,))
            
            row = cur.fetchone()
            if row:
                from llama_index.core.schema import TextNode
                import json
                
                text = row[0]
                metadata = json.loads(row[1]) if row[1] else {}
                
                node = TextNode(
                    text=text,
                    metadata=metadata,
                    id_=node_id
                )
                
                return NodeWithScore(node=node, score=0.0)
    except Exception as e:
        print(f"Error fetching node from DB: {e}")
    finally:
        conn.close()
    
    # Return empty node if not found
    from llama_index.core.schema import TextNode
    return NodeWithScore(
        node=TextNode(text="", metadata={}, id_=node_id),
        score=0.0
    )


def hybrid_search(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 20,
    use_hybrid: bool = True
) -> List[NodeWithScore]:
    """
    Perform hybrid search (BM25 + Vector) with RRF fusion.
    
    Args:
        index: VectorStoreIndex instance
        query: Search query
        top_k: Number of results to return
        use_hybrid: If True, use hybrid search; if False, use vector search only
        
    Returns:
        List of NodeWithScore objects
    """
    # Note: fulltext index should be created/updated during indexing, not during query
    # This avoids the expensive update_fulltext_index() call on every query
    
    # Get vector store retriever
    vector_retriever = index.as_retriever(similarity_top_k=top_k)
    vector_results = vector_retriever.retrieve(query)
    
    if not use_hybrid:
        return vector_results
    
    # Perform BM25 search
    bm25_results = bm25_search(query, top_k=top_k)
    
    if not bm25_results:
        # If BM25 returns no results, fall back to vector search
        return vector_results
    
    # Combine results using RRF
    combined_results = reciprocal_rank_fusion(
        vector_results=vector_results,
        bm25_results=bm25_results,
        k=60
    )
    
    # Return top_k results
    return combined_results[:top_k]
