"""Tracing utilities for custom observability with OpenTelemetry."""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span

# OpenInference semantic conventions for Phoenix
# See: https://github.com/Arize-ai/openinference
OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
INPUT_MIME_TYPE = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME_TYPE = "output.mime_type"

# Span kinds recognized by Phoenix
class SpanKind:
    CHAIN = "CHAIN"
    RETRIEVER = "RETRIEVER"
    RERANKER = "RERANKER"
    LLM = "LLM"
    EMBEDDING = "EMBEDDING"
    TOOL = "TOOL"

# Tracer instance for custom spans
# Uses the same tracer provider configured in main.py via Phoenix
_tracer: Optional[trace.Tracer] = None


def get_tracer() -> trace.Tracer:
    """
    Get the OpenTelemetry tracer instance.
    
    Returns:
        Tracer instance for creating spans
    """
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer("rag-notes.custom", "1.0.0")
    return _tracer


@contextmanager
def traced_operation(
    name: str,
    span_kind: str = SpanKind.CHAIN,
    input_value: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True
):
    """
    Context manager for tracing a custom operation.
    
    Args:
        name: Name of the operation (will appear in Phoenix UI)
        span_kind: OpenInference span kind (CHAIN, RETRIEVER, RERANKER, etc.)
        input_value: Input value to display in Phoenix UI
        attributes: Optional dictionary of span attributes
        record_exception: Whether to record exceptions on the span
        
    Yields:
        The current span for adding additional attributes
        
    Example:
        with traced_operation("bm25_search", SpanKind.RETRIEVER, query) as span:
            results = do_search()
            set_span_output(span, f"Found {len(results)} results")
    """
    tracer = get_tracer()
    start_time = time.perf_counter()
    
    with tracer.start_as_current_span(name) as span:
        # Set OpenInference span kind for Phoenix UI
        span.set_attribute(OPENINFERENCE_SPAN_KIND, span_kind)
        
        # Set input value for Phoenix UI
        if input_value:
            span.set_attribute(INPUT_VALUE, input_value)
            span.set_attribute(INPUT_MIME_TYPE, "text/plain")
        
        # Set additional attributes
        if attributes:
            for key, value in attributes.items():
                _set_span_attribute(span, key, value)
        
        try:
            yield span
            # Record execution time in milliseconds
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("duration_ms", round(elapsed_ms, 2))
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            span.set_attribute("duration_ms", round(elapsed_ms, 2))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            if record_exception:
                span.record_exception(e)
            raise


def set_span_output(span: Span, output_value: str) -> None:
    """
    Set the output value for a span (displayed in Phoenix UI output column).
    
    Args:
        span: The span to set output on
        output_value: The output value to display
    """
    span.set_attribute(OUTPUT_VALUE, output_value)
    span.set_attribute(OUTPUT_MIME_TYPE, "text/plain")


def _set_span_attribute(span: Span, key: str, value: Any) -> None:
    """Set a single attribute on a span, handling type conversion."""
    if value is None:
        return
    if isinstance(value, (str, int, float, bool)):
        span.set_attribute(key, value)
    elif isinstance(value, (list, tuple)):
        # For lists, try to keep as list if all elements are same primitive type
        if len(value) > 0 and all(isinstance(v, str) for v in value):
            span.set_attribute(key, list(value))
        elif len(value) > 0 and all(isinstance(v, (int, float)) for v in value):
            span.set_attribute(key, list(value))
        else:
            span.set_attribute(key, str(value))
    else:
        span.set_attribute(key, str(value))


def add_span_attributes(span: Span, attributes: Dict[str, Any]) -> None:
    """
    Add attributes to an existing span.
    
    Args:
        span: The span to add attributes to
        attributes: Dictionary of attributes to add
    """
    for key, value in attributes.items():
        _set_span_attribute(span, key, value)


def format_results_for_trace(
    nodes: List[Any],
    max_items: int = 5,
    max_text_len: int = 100
) -> Dict[str, Any]:
    """
    Format search results for tracing display.
    
    Args:
        nodes: List of NodeWithScore objects
        max_items: Maximum number of items to include details for
        max_text_len: Maximum text length for preview
        
    Returns:
        Dictionary with formatted result information
    """
    if not nodes:
        return {
            "output.result_count": 0,
            "output.results": "[]"
        }
    
    result_summaries = []
    filenames = []
    scores = []
    
    for i, node in enumerate(nodes[:max_items]):
        if hasattr(node, 'node'):
            # NodeWithScore object
            text = node.node.text[:max_text_len] + "..." if len(node.node.text) > max_text_len else node.node.text
            filename = node.node.metadata.get("filename", "unknown")
            score = node.score if node.score is not None else 0.0
        else:
            # Tuple (node_id, score) for BM25 results
            text = str(node)
            filename = "unknown"
            score = node[1] if isinstance(node, tuple) and len(node) > 1 else 0.0
        
        result_summaries.append(f"[{i+1}] {filename} (score={score:.4f}): {text[:50]}...")
        filenames.append(filename)
        scores.append(round(score, 4))
    
    return {
        "output.result_count": len(nodes),
        "output.top_results": "\n".join(result_summaries),
        "output.filenames": filenames,
        "output.scores": scores,
        "output.top_score": max(scores) if scores else 0.0
    }
