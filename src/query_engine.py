"""Query engine module for RAG-based question answering."""

from dataclasses import dataclass
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openai_like import OpenAILike

from .config import config
from .indexer import get_existing_index

# Minimum score threshold for reranked results
# Results below this threshold will be filtered out before sending to LLM
MIN_RERANK_SCORE_THRESHOLD = 0.3  # 30%


@dataclass
class DebugInfo:
    """Container for debugging information."""
    
    # Hybrid search results (before reranking)
    hybrid_results: List[dict]  # [{filename, score, text_preview}]
    # Reranked results
    reranked_results: List[dict]  # [{filename, score, text_preview}]
    # Final results sent to LLM (after filtering)
    final_results: List[dict]  # [{filename, score, text_preview}]
    # The actual prompt/context sent to LLM
    llm_input: str


@dataclass
class QueryResult:
    """Container for query results."""

    answer: str
    source_chunks: List[str]
    source_files: List[str]
    scores: List[float]
    debug_info: DebugInfo = None  # Optional debug information


def get_llm() -> OpenAILike:
    """Create LLM via OpenRouter."""
    return OpenAILike(
        model=config.LLM_MODEL,
        api_base=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        is_chat_model=True,
    )


def create_query_engine(index: VectorStoreIndex, similarity_top_k: int = 3):
    """Create a query engine from the index."""
    llm = get_llm()

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=similarity_top_k,
        response_mode=ResponseMode.COMPACT,
    )

    return query_engine


def query(
    question: str,
    use_hybrid: bool = False,
    use_rerank: bool = False
) -> QueryResult:
    """
    Query the indexed documents.

    Args:
        question: User's question
        use_hybrid: If True, use hybrid search (BM25 + Vector)
        use_rerank: If True, use Cohere reranking

    Returns:
        QueryResult with answer and source information
    """
    config.validate()

    # Get existing index
    index = get_existing_index()

    # Helper function to extract node info for debug
    def _node_to_debug_dict(node_with_score, max_preview_len=100):
        node = node_with_score.node
        text = node.text
        preview = text[:max_preview_len] + "..." if len(text) > max_preview_len else text
        filename = node.metadata.get("filename", "Unknown")
        score = node_with_score.score if node_with_score.score is not None else 0.0
        return {"filename": filename, "score": score, "text_preview": preview}
    
    # Initialize debug info
    debug_hybrid_results = []
    debug_reranked_results = []
    debug_final_results = []
    debug_llm_input = ""
    
    # Determine retrieval strategy
    if use_hybrid:
        # Use hybrid search
        from .hybrid_search import hybrid_search
        from .reranker import rerank_results
        
        # Perform hybrid search (top_k=20 for reranking)
        top_k = 20 if use_rerank else 3
        retrieved_nodes = hybrid_search(
            index=index,
            query=question,
            top_k=top_k,
            use_hybrid=True
        )
        
        # Collect hybrid search debug info
        debug_hybrid_results = [_node_to_debug_dict(n) for n in retrieved_nodes]
        
        # Apply reranking if enabled
        if use_rerank:
            retrieved_nodes = rerank_results(
                query=question,
                nodes=retrieved_nodes,
                top_n=3
            )
            # Collect reranked debug info
            debug_reranked_results = [_node_to_debug_dict(n) for n in retrieved_nodes]
        
        # Use retrieved nodes to create context
        from llama_index.core import get_response_synthesizer
        
        llm = get_llm()
        synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT, llm=llm)
        
        # Filter out low-score results when reranking is enabled
        # Rerank scores are relevance scores (0-1), low scores indicate irrelevant documents
        if use_rerank:
            filtered_nodes = [
                node for node in retrieved_nodes
                if node.score is not None and node.score >= MIN_RERANK_SCORE_THRESHOLD
            ]
            # Always keep at least the top result even if below threshold
            if not filtered_nodes and retrieved_nodes:
                filtered_nodes = [retrieved_nodes[0]]
            retrieved_nodes = filtered_nodes
        
        # Collect final results debug info
        debug_final_results = [_node_to_debug_dict(n) for n in retrieved_nodes]
        
        # Build LLM input context for debug display
        context_parts = []
        for i, node_with_score in enumerate(retrieved_nodes):
            context_parts.append(f"[Document {i+1}]\n{node_with_score.node.text}")
        debug_llm_input = f"Question: {question}\n\n---\nContext:\n" + "\n\n---\n".join(context_parts)
        
        # Create response from retrieved nodes
        response = synthesizer.synthesize(
            query=question,
            nodes=retrieved_nodes
        )
        
        # Extract source information from retrieved nodes
        source_chunks = []
        source_files = []
        scores = []
        
        for node_with_score in retrieved_nodes:
            node = node_with_score.node
            source_chunks.append(node.text)
            scores.append(node_with_score.score if node_with_score.score is not None else 0.0)
            if "filename" in node.metadata:
                source_files.append(node.metadata["filename"])
            else:
                source_files.append("Unknown")
        
        # Create debug info object
        debug_info = DebugInfo(
            hybrid_results=debug_hybrid_results,
            reranked_results=debug_reranked_results,
            final_results=debug_final_results,
            llm_input=debug_llm_input
        )
        
        return QueryResult(
            answer=str(response),
            source_chunks=source_chunks,
            source_files=source_files,
            scores=scores,
            debug_info=debug_info,
        )
    else:
        # Use standard vector search
        query_engine = create_query_engine(index, similarity_top_k=3)
        response = query_engine.query(question)

        # Extract source information
        source_chunks = []
        source_files = []
        scores = []

        for node in response.source_nodes:
            source_chunks.append(node.text)
            scores.append(node.score if node.score is not None else 0.0)
            if "filename" in node.metadata:
                source_files.append(node.metadata["filename"])
            else:
                source_files.append("Unknown")
        
        # Build debug info for vector-only search
        debug_vector_results = []
        for node in response.source_nodes:
            text = node.text
            preview = text[:100] + "..." if len(text) > 100 else text
            filename = node.metadata.get("filename", "Unknown")
            score = node.score if node.score is not None else 0.0
            debug_vector_results.append({"filename": filename, "score": score, "text_preview": preview})
        
        # Build LLM input for debug
        context_parts = []
        for i, node in enumerate(response.source_nodes):
            context_parts.append(f"[Document {i+1}]\n{node.text}")
        debug_llm_input = f"Question: {question}\n\n---\nContext:\n" + "\n\n---\n".join(context_parts)
        
        debug_info = DebugInfo(
            hybrid_results=[],  # No hybrid search in this mode
            reranked_results=[],  # No reranking in this mode
            final_results=debug_vector_results,
            llm_input=debug_llm_input
        )

        return QueryResult(
            answer=str(response),
            source_chunks=source_chunks,
            source_files=source_files,
            scores=scores,
            debug_info=debug_info,
        )
