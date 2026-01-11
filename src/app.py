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


def handle_query(
    question: str,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    show_debug: bool = False
) -> Tuple[str, str, str]:
    """
    Handle user question.

    Args:
        question: User's question
        use_hybrid: If True, use hybrid search (BM25 + Vector)
        use_rerank: If True, use Cohere reranking
        show_debug: If True, return debug information

    Returns:
        Tuple of (answer, source_chunks, debug_info)
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    try:
        result = query(question, use_hybrid=use_hybrid, use_rerank=use_rerank)

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

        # Format debug information
        debug_text = ""
        if show_debug and result.debug_info:
            debug_parts = []
            
            # Hybrid search results
            if result.debug_info.hybrid_results:
                debug_parts.append("## 🔍 混合检索结果 (Hybrid Search)")
                debug_parts.append(f"共 {len(result.debug_info.hybrid_results)} 个结果（显示前10）：\n")
                for i, r in enumerate(result.debug_info.hybrid_results[:10]):  # Show top 10
                    debug_parts.append(f"{i+1}. **{r['filename']}** — RRF score: `{r['score']:.4f}`")
                debug_parts.append("")
            
            debug_parts.append("---")
            debug_parts.append("")
            
            # Reranked results
            if result.debug_info.reranked_results:
                debug_parts.append("## 🎯 重排序结果 (After Rerank)")
                debug_parts.append("阈值: < 30% 将被过滤\n")
                for i, r in enumerate(result.debug_info.reranked_results):
                    score_pct = r['score'] * 100
                    status = "✅" if score_pct >= 30 else "❌"
                    debug_parts.append(f"{i+1}. {status} **{r['filename']}** — `{score_pct:.1f}%`")
                debug_parts.append("")
            
            debug_parts.append("---")
            debug_parts.append("")
            
            # Final results sent to LLM
            debug_parts.append("## ✅ 最终传给 LLM 的结果")
            if result.debug_info.final_results:
                debug_parts.append(f"共 {len(result.debug_info.final_results)} 个文档：\n")
                for i, r in enumerate(result.debug_info.final_results):
                    score_pct = r['score'] * 100
                    debug_parts.append(f"{i+1}. **{r['filename']}** — `{score_pct:.1f}%`")
                debug_parts.append("")
            else:
                debug_parts.append("（无结果通过过滤）")
                debug_parts.append("")
            
            debug_parts.append("---")
            debug_parts.append("")
            
            # LLM Input
            debug_parts.append("## 📝 LLM 输入上下文")
            debug_parts.append(f"```\n{result.debug_info.llm_input}\n```")
            
            debug_text = "\n".join(debug_parts)

        return result.answer, sources_text, debug_text
    except Exception as e:
        return f"Error: {str(e)}", "", ""


def create_app() -> gr.Blocks:
    """Create and configure the Gradio application."""
    with gr.Blocks(title="RAG Notes - Q&A Assistant") as app:
        gr.Markdown("# RAG Notes Q&A Assistant")
        gr.Markdown("Upload your documents (Markdown, TXT, PDF, Word, Excel, PowerPoint, HTML, etc.), then ask questions about them.")

        with gr.Tab("Upload Documents"):
            file_input = gr.File(
                label="Upload Files",
                file_count="multiple",
                file_types=[
                    ".md", ".txt",           # Plain text
                    ".pdf",                  # PDF
                    ".docx",                 # Word (new format only)
                    ".xls", ".xlsx",         # Excel
                    ".ppt", ".pptx",         # PowerPoint
                    ".html", ".htm",         # HTML
                    ".csv", ".json", ".xml", # Data formats
                ],
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
            
            with gr.Row():
                hybrid_search_checkbox = gr.Checkbox(
                    label="混合检索 (BM25 + Vector)",
                    value=False,
                    info="开启后使用BM25全文检索与向量检索的混合检索，使用RRF融合结果"
                )
                rerank_checkbox = gr.Checkbox(
                    label="重排序 (Cohere Rerank)",
                    value=False,
                    info="开启后使用Cohere Rerank API对检索结果进行重排序（需要COHERE_API_KEY）"
                )
                debug_checkbox = gr.Checkbox(
                    label="🐛 显示调试信息",
                    value=False,
                    info="显示检索过程的中间结果"
                )
            
            ask_btn = gr.Button("Ask", variant="primary")

            answer_output = gr.Textbox(
                label="Answer",
                interactive=False,
                lines=5,
            )
            sources_output = gr.Markdown(label="Related Sources")
            
            # Debug information section (collapsible)
            with gr.Accordion("🔧 调试信息 (Debug Info)", open=False, visible=True) as debug_accordion:
                debug_output = gr.Markdown(
                    label="Debug Information",
                    value="开启「显示调试信息」选项后，这里会显示检索过程的详细信息。"
                )

            ask_btn.click(
                fn=handle_query,
                inputs=[question_input, hybrid_search_checkbox, rerank_checkbox, debug_checkbox],
                outputs=[answer_output, sources_output, debug_output],
            )

            # Also trigger on Enter key
            question_input.submit(
                fn=handle_query,
                inputs=[question_input, hybrid_search_checkbox, rerank_checkbox, debug_checkbox],
                outputs=[answer_output, sources_output, debug_output],
            )

    return app
