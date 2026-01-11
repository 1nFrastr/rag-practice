"""Tracing utilities for custom observability with OpenTelemetry."""

from contextlib import contextmanager
from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span

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
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True
):
    """
    Context manager for tracing a custom operation.
    
    Args:
        name: Name of the operation (will appear in Phoenix UI)
        attributes: Optional dictionary of span attributes
        record_exception: Whether to record exceptions on the span
        
    Yields:
        The current span for adding additional attributes
        
    Example:
        with traced_operation("bm25_search", {"query": query, "top_k": 10}) as span:
            results = do_search()
            span.set_attribute("result_count", len(results))
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span(name) as span:
        # Set initial attributes
        if attributes:
            for key, value in attributes.items():
                # Only set serializable values
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
                elif isinstance(value, (list, tuple)):
                    # Convert lists/tuples to string for display
                    span.set_attribute(key, str(value))
        
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            if record_exception:
                span.record_exception(e)
            raise


def add_span_attributes(span: Span, attributes: Dict[str, Any]) -> None:
    """
    Add attributes to an existing span.
    
    Args:
        span: The span to add attributes to
        attributes: Dictionary of attributes to add
    """
    for key, value in attributes.items():
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)
        elif isinstance(value, (list, tuple)):
            span.set_attribute(key, str(value))
