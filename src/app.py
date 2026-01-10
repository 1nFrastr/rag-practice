"""Gradio web interface for RAG Notes application."""

from typing import List, Tuple

import gradio as gr

from .indexer import index_documents
from .query_engine import query


def handle_upload(files: List) -> str:
    """
    Handle file upload and indexing.

    Args:
        files: List of uploaded file objects from Gradio

    Returns:
        Status message
    """
    if not files:
        return "请至少上传一个文件。"

    try:
        # Get file paths from uploaded files
        file_paths = [f.name for f in files]

        # Index documents with chunk-level deduplication
        result = index_documents(file_paths)

        # Build status message
        messages = [
            f"📄 处理文件: {', '.join(result.files_processed)}",
            f"📊 总 chunks: {result.total_chunks}",
        ]

        if result.indexed_chunks > 0:
            messages.append(f"✅ 新索引: {result.indexed_chunks} 个 chunks")

        if result.skipped_chunks > 0:
            messages.append(f"⏭️ 跳过重复: {result.skipped_chunks} 个 chunks")

        if result.indexed_chunks == 0 and result.skipped_chunks > 0:
            messages.append("ℹ️ 所有内容已存在，无需重复索引")

        return "\n".join(messages)
    except Exception as e:
        return f"索引文件时出错: {str(e)}"


def handle_query(question: str) -> Tuple[str, str]:
    """
    Handle user question.

    Args:
        question: User's question

    Returns:
        Tuple of (answer, source_chunks)
    """
    if not question.strip():
        return "Please enter a question.", ""

    try:
        result = query(question)

        # Format source chunks with collapsible details
        sources_text = ""
        if result.source_chunks:
            sources_parts = []
            for i, chunk in enumerate(result.source_chunks):
                filename = result.source_files[i] if i < len(result.source_files) else "Unknown"
                score = result.scores[i] if i < len(result.scores) else 0.0
                score_percent = score * 100
                sources_parts.append(
                    f"<details>\n"
                    f"<summary><strong>Source {i+1}</strong> - {filename} "
                    f"(Match: {score_percent:.1f}%)</summary>\n\n"
                    f"{chunk}\n\n"
                    f"</details>"
                )
            sources_text = "\n\n".join(sources_parts)

        return result.answer, sources_text
    except Exception as e:
        return f"Error: {str(e)}", ""


def create_app() -> gr.Blocks:
    """Create and configure the Gradio application."""
    with gr.Blocks(title="RAG Notes - Q&A Assistant") as app:
        gr.Markdown("# RAG Notes Q&A Assistant")
        gr.Markdown("Upload your Markdown or TXT notes, then ask questions about them.")

        with gr.Tab("Upload Documents"):
            file_input = gr.File(
                label="Upload Files",
                file_count="multiple",
                file_types=[".md", ".txt"],
            )
            upload_btn = gr.Button("Index Documents", variant="primary")
            upload_status = gr.Textbox(label="Status", interactive=False, lines=6)

            upload_btn.click(
                fn=handle_upload,
                inputs=[file_input],
                outputs=[upload_status],
            )

        with gr.Tab("Ask Questions"):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Enter your question here...",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")

            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=5,
            )
            sources_output = gr.Markdown(label="Related Sources")

            ask_btn.click(
                fn=handle_query,
                inputs=[question_input],
                outputs=[answer_output, sources_output],
            )

            # Also trigger on Enter key
            question_input.submit(
                fn=handle_query,
                inputs=[question_input],
                outputs=[answer_output, sources_output],
            )

    return app
