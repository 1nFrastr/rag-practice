"""Entry point for RAG Notes application."""

import atexit
import os
import signal
import sys
import threading
from pathlib import Path

# Fix Windows encoding issues - do this early
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Print immediately so user knows app is starting
print("[App] 启动中...")

# Global reference to Phoenix session for cleanup
_phoenix_session = None


def cleanup():
    """Clean up Phoenix and other resources."""
    global _phoenix_session
    if _phoenix_session is not None:
        print("\n[Phoenix] Shutting down...")
        try:
            import phoenix as px
            px.close_app()
        except Exception:
            pass
        _phoenix_session = None
    print("[App] Goodbye!")


def signal_handler(signum, frame):
    """Handle interrupt signals for clean shutdown."""
    cleanup()
    sys.exit(0)


def setup_observability():
    """Initialize Phoenix observability platform (runs in background thread)."""
    global _phoenix_session
    
    print("[Phoenix] 正在后台加载 observability...")
    
    # Lazy import heavy modules
    import phoenix as px
    from phoenix.otel import register
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    
    # Use a persistent directory for Phoenix data
    phoenix_dir = Path(__file__).parent.parent / ".phoenix"
    phoenix_dir.mkdir(exist_ok=True)
    os.environ["PHOENIX_WORKING_DIR"] = str(phoenix_dir)
    
    # Launch Phoenix in the background
    # Access the UI at http://localhost:6006
    _phoenix_session = px.launch_app()
    
    # Register Phoenix as the trace provider
    # This connects OpenTelemetry to Phoenix
    tracer_provider = register(
        project_name="rag-notes",
        endpoint="http://localhost:6006/v1/traces",
    )
    
    # Instrument LlamaIndex to automatically trace all calls
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    
    print("[Phoenix] Observability 已启动: http://localhost:6006")


def main():
    """Launch the Gradio application."""
    # Register cleanup handlers in main thread (signal only works in main thread)
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start observability in background thread (non-blocking)
    observability_thread = threading.Thread(target=setup_observability, daemon=True)
    observability_thread.start()
    
    print("[Gradio] 正在加载 Web 界面...")
    
    # Lazy import app module (which imports gradio and llama_index)
    from .app import create_app
    
    app = create_app()
    
    print("[Gradio] 启动完成！")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    main()
