"""Reranking module using Cohere Rerank API."""

from typing import List, Optional

import cohere
from llama_index.core.schema import NodeWithScore

from .config import config


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
    if not nodes:
        return []
    
    if not config.COHERE_API_KEY:
        # If API key is not set, return original results
        return nodes[:top_n]
    
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
        for result in response.results:
            index = result.index
            if 0 <= index < len(nodes):
                original_node = nodes[index]
                # Create new NodeWithScore with rerank score
                reranked_nodes.append(
                    NodeWithScore(
                        node=original_node.node,
                        score=float(result.relevance_score)
                    )
                )
        
        return reranked_nodes
    
    except Exception as e:
        print(f"Error in Cohere reranking: {e}")
        # On error, return original top_n results
        return nodes[:top_n]
