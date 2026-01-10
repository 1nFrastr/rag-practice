"""Entry point for RAG Notes application."""

import os
import sys
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

from .app import create_app


def setup_observability():
    """Initialize Phoenix observability platform."""
    # Use a persistent directory for Phoenix data
    phoenix_dir = Path(__file__).parent.parent / ".phoenix"
    phoenix_dir.mkdir(exist_ok=True)
    os.environ["PHOENIX_WORKING_DIR"] = str(phoenix_dir)
    
    # Launch Phoenix in the background
    # Access the UI at http://localhost:6006
    px.launch_app()
    
    # Register Phoenix as the trace provider
    # This connects OpenTelemetry to Phoenix
    tracer_provider = register(
        project_name="rag-notes",
        endpoint="http://localhost:6006/v1/traces",
    )
    
    # Instrument LlamaIndex to automatically trace all calls
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    
    print("[Phoenix] Observability started at http://localhost:6006")


def main():
    """Launch the Gradio application."""
    # Setup observability first
    setup_observability()
    
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
