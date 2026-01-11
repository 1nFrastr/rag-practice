"""Reranking module using Cohere Rerank API."""

import json
from typing import Any, Dict, List, Optional

import cohere
from llama_index.core.schema import NodeWithScore
from openinference.instrumentation import get_reranker_attributes
from openinference.instrumentation._types import Document
from .config import config
from .tracing import traced_operation, add_span_attributes, set_span_output, SpanKind


def _nodes_to_documents(
    nodes: List[NodeWithScore],
    include_score: bool = False
) -> List[Document]:
    """
    Convert NodeWithScore objects to OpenInference Document format.
    
    Document TypedDict has keys: content, id, metadata, score
    """
    documents: List[Document] = []
    for i, node in enumerate(nodes):
        doc: Document = {
            "content": node.node.text[:500],  # Truncate for display
            "id": node.node.node_id or f"doc_{i}",
        }
        if include_score and node.score is not None:
            doc["score"] = float(node.score)
        
        # Include key metadata
        if node.node.metadata:
            filename = node.node.metadata.get("filename", "")
            if filename:
                doc["metadata"] = json.dumps({"filename": filename})
        
        documents.append(doc)
    return documents


def rerank_results(
    query: str,
    nodes: List[NodeWithScore],
    top_n: int = 3,
    model: str = "rerank-multilingual-v3.0"
) -> List[NodeWithScore]:
    """
    Rerank search results using Cohere Rerank API.
    
    Args:
        query: Search query
        nodes: List of NodeWithScore objects to rerank
        top_n: Number of top results to return after reranking
        model: Cohere rerank model name
        
    Returns:
        Reranked list of NodeWithScore objects (top_n results)
    """
    # Build input summary for tracing
    input_summary = f"Query: {query}\nCandidates: {len(nodes)} docs"
    
    # Format input documents for Phoenix tracing (using OpenInference format)
    input_docs = _nodes_to_documents(nodes, include_score=True)
    
    # Use get_reranker_attributes to generate proper flattened attributes
    reranker_attrs = get_reranker_attributes(
        query=query,
        model_name=model,
        top_k=top_n,
        input_documents=input_docs,
    )
    
    with traced_operation(
        "cohere_rerank",
        span_kind=SpanKind.RERANKER,
        input_value=input_summary,
        attributes=reranker_attrs,
    ) as span:
        if not nodes:
            add_span_attributes(span, {"status": "empty_input"})
            set_span_output(span, "No candidates to rerank")
            return []
        
        if not config.COHERE_API_KEY:
            # If API key is not set, return original results
            fallback_results = nodes[:top_n]
            add_span_attributes(span, {"status": "no_api_key_fallback"})
            set_span_output(span, f"No API key - returning top {len(fallback_results)} original")
            return fallback_results
        
        try:
            # Initialize Cohere client
            client = cohere.Client(api_key=config.COHERE_API_KEY)
            
            # Extract texts from nodes
            documents = [node.node.text for node in nodes]
            
            # Call Cohere Rerank API
            response = client.rerank(
                model=model,
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False
            )
            
            # Map reranked results back to NodeWithScore objects
            reranked_nodes = []
            scores = []
            rank_changes = []
            output_details = []
            
            for new_rank, result in enumerate(response.results):
                old_index = result.index
                if 0 <= old_index < len(nodes):
                    original_node = nodes[old_index]
                    relevance_score = float(result.relevance_score)
                    scores.append(relevance_score)
                    
                    # Track rank change
                    rank_change = old_index - new_rank  # positive = moved up
                    rank_changes.append(f"{old_index}→{new_rank}")
                    
                    # Build output detail
                    filename = original_node.node.metadata.get("filename", "unknown")
                    score_pct = relevance_score * 100
                    text_preview = original_node.node.text[:40].replace("\n", " ")
                    output_details.append(
                        f"[{new_rank+1}] {filename} ({score_pct:.1f}%): {text_preview}..."
                    )
                    
                    # Create new NodeWithScore with rerank score
                    reranked_nodes.append(
                        NodeWithScore(
                            node=original_node.node,
                            score=relevance_score
                        )
                    )
            
            # Format output documents for Phoenix tracing (using OpenInference format)
            output_docs = _nodes_to_documents(reranked_nodes, include_score=True)
            
            # Use get_reranker_attributes to generate proper flattened output document attributes
            output_attrs = get_reranker_attributes(output_documents=output_docs)
            
            # Add detailed rerank metrics to span
            add_span_attributes(span, {
                "status": "success",
                "result_count": len(reranked_nodes),
                "top_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "rank_changes": str(rank_changes),
                **output_attrs,
            })
            
            # Set output for Phoenix UI
            output_summary = "\n".join(output_details)
            set_span_output(span, output_summary)
            
            return reranked_nodes
        
        except Exception as e:
            print(f"Error in Cohere reranking: {e}")
            fallback_results = nodes[:top_n]
            add_span_attributes(span, {
                "status": "error",
                "error": str(e),
            })
            set_span_output(span, f"Error: {str(e)} - fallback to original order")
            # On error, return original top_n results
            return fallback_results
