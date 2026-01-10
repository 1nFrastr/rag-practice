"""Query engine module for RAG-based question answering."""

from dataclasses import dataclass
from typing import List

from llama_index.core import VectorStoreIndex
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.llms.openai_like import OpenAILike

from .config import config
from .indexer import get_existing_index


@dataclass
class QueryResult:
    """Container for query results."""

    answer: str
    source_chunks: List[str]
    source_files: List[str]
    scores: List[float]


def get_llm() -> OpenAILike:
    """Create LLM via OpenRouter."""
    return OpenAILike(
        model=config.LLM_MODEL,
        api_base=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
        is_chat_model=True,
    )


def create_query_engine(index: VectorStoreIndex):
    """Create a query engine from the index."""
    llm = get_llm()

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        response_mode=ResponseMode.COMPACT,
    )

    return query_engine


def query(question: str) -> QueryResult:
    """
    Query the indexed documents.

    Args:
        question: User's question

    Returns:
        QueryResult with answer and source information
    """
    config.validate()

    # Get existing index
    index = get_existing_index()

    # Create query engine
    query_engine = create_query_engine(index)

    # Execute query
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

    return QueryResult(
        answer=str(response),
        source_chunks=source_chunks,
        source_files=source_files,
        scores=scores,
    )
