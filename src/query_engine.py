"""Query engine module for RAG-based question answering."""

from dataclasses import dataclass
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openai_like import OpenAILike

from .config import config
from .indexer import get_existing_index
from .tracing import traced_operation, add_span_attributes, set_span_output, SpanKind

# Minimum score threshold for reranked results
# Results below this threshold will be filtered out before sending to LLM
MIN_RERANK_SCORE_THRESHOLD = 0.3  # 30%


@dataclass
class DebugInfo:
    """Container for debugging information."""
    
    # Vector search results (before reranking, when not using hybrid)
    vector_results: List[dict]  # [{filename, score, text_preview, chunk_id}]
    # Hybrid search results (before reranking)
    hybrid_results: List[dict]  # [{filename, score, text_preview, chunk_id}]
    # Reranked results
    reranked_results: List[dict]  # [{filename, score, text_preview, chunk_id}]
    # Final results sent to LLM (after filtering)
    final_results: List[dict]  # [{filename, score, text_preview, chunk_id}]
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
    with traced_operation(
        "rag_query",
        span_kind=SpanKind.CHAIN,
        input_value=question,
        attributes={
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "search_mode": "hybrid" if use_hybrid else "vector"
        }
    ) as parent_span:
        config.validate()

        # Get existing index
        index = get_existing_index()

        # Helper function to extract node info for debug
        def _node_to_debug_dict(node_with_score, max_preview_len=100):
            node = node_with_score.node
            text = node.text
            preview = text[:max_preview_len] + "..." if len(text) > max_preview_len else text
            filename = node.metadata.get("filename", "Unknown")
            chunk_hash = node.metadata.get("chunk_hash", "")
            # 只取前8位作为短标识符
            chunk_id = chunk_hash[:8] if chunk_hash else node.node_id[:8]
            score = node_with_score.score if node_with_score.score is not None else 0.0
            return {"filename": filename, "score": score, "text_preview": preview, "chunk_id": chunk_id}
        
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
            
            # Always retrieve 20 candidates for better debug visibility
            retrieved_nodes = hybrid_search(
                index=index,
                query=question,
                top_k=20,
                use_hybrid=True
            )
            
            # Collect hybrid search debug info (all 20 results)
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
            else:
                # Without reranking, take top 3 from hybrid results
                retrieved_nodes = retrieved_nodes[:3]
            
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
            
            # Add final metrics to parent span
            add_span_attributes(parent_span, {
                "retrieved_count": len(debug_hybrid_results),
                "final_count": len(source_chunks),
                "source_files": ", ".join(set(source_files)),
                "answer_length": len(str(response)),
            })
            
            # Set output for Phoenix UI
            answer_preview = str(response)[:200] + "..." if len(str(response)) > 200 else str(response)
            set_span_output(parent_span, answer_preview)
            
            # Create debug info object
            debug_info = DebugInfo(
                vector_results=[],  # No separate vector search in hybrid mode
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
            # Use standard vector search (with optional reranking)
            from .reranker import rerank_results
            from llama_index.core import get_response_synthesizer
            
            # Always retrieve 20 candidates for better debug visibility
            vector_retriever = index.as_retriever(similarity_top_k=20)
            retrieved_nodes = vector_retriever.retrieve(question)
            
            # Collect vector search debug info (all 20 results)
            debug_vector_results = [_node_to_debug_dict(n) for n in retrieved_nodes]
            
            # Apply reranking if enabled
            if use_rerank:
                retrieved_nodes = rerank_results(
                    query=question,
                    nodes=retrieved_nodes,
                    top_n=3
                )
                # Collect reranked debug info
                debug_reranked_results = [_node_to_debug_dict(n) for n in retrieved_nodes]
                
                # Filter out low-score results
                filtered_nodes = [
                    node for node in retrieved_nodes
                    if node.score is not None and node.score >= MIN_RERANK_SCORE_THRESHOLD
                ]
                # Always keep at least the top result even if below threshold
                if not filtered_nodes and retrieved_nodes:
                    filtered_nodes = [retrieved_nodes[0]]
                retrieved_nodes = filtered_nodes
            else:
                # Without reranking, take top 3 from vector results
                retrieved_nodes = retrieved_nodes[:3]
            
            # Collect final results debug info
            debug_final_results = [_node_to_debug_dict(n) for n in retrieved_nodes]
            
            # Build LLM input context for debug display
            context_parts = []
            for i, node_with_score in enumerate(retrieved_nodes):
                context_parts.append(f"[Document {i+1}]\n{node_with_score.node.text}")
            debug_llm_input = f"Question: {question}\n\n---\nContext:\n" + "\n\n---\n".join(context_parts)
            
            # Create response from retrieved nodes
            llm = get_llm()
            synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT, llm=llm)
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
            
            # Add final metrics to parent span
            add_span_attributes(parent_span, {
                "retrieved_count": len(debug_vector_results),
                "final_count": len(source_chunks),
                "source_files": ", ".join(set(source_files)),
                "answer_length": len(str(response)),
            })
            
            # Set output for Phoenix UI
            answer_preview = str(response)[:200] + "..." if len(str(response)) > 200 else str(response)
            set_span_output(parent_span, answer_preview)
            
            debug_info = DebugInfo(
                vector_results=debug_vector_results,  # Initial vector search results
                hybrid_results=[],  # No hybrid search in this mode
                reranked_results=debug_reranked_results if use_rerank else [],
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
