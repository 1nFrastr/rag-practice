"""Gradio web interface for RAG Notes application."""

from typing import List, Tuple, Generator

import gradio as gr

from .indexer import index_documents
from .query_engine import query, query_stream
from .chat_engine import ChatHistory, chat_stream, clear_session, get_or_create_session


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


def _format_sources(result) -> str:
    """Format source chunks with collapsible details."""
    if not result.source_chunks:
        return ""
    
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
    return "\n\n".join(sources_parts)


def _format_debug_info(result, show_debug: bool) -> str:
    """Format debug information."""
    if not show_debug or not result.debug_info:
        return ""
    
    debug_parts = []
    
    # First stage: Hybrid search or Vector search results
    if result.debug_info.hybrid_results:
        debug_parts.append("## 🔍 混合检索结果 (Hybrid Search)")
        debug_parts.append(f"共 {len(result.debug_info.hybrid_results)} 个结果（显示前10）：\n")
        for i, r in enumerate(result.debug_info.hybrid_results[:10]):
            chunk_id = r.get('chunk_id', '')
            debug_parts.append(f"{i+1}. **{r['filename']}** `[{chunk_id}]` — RRF score: `{r['score']:.4f}`")
        debug_parts.append("")
        debug_parts.append("---")
        debug_parts.append("")
    elif result.debug_info.vector_results:
        debug_parts.append("## 🔍 向量检索结果 (Vector Search)")
        debug_parts.append(f"共 {len(result.debug_info.vector_results)} 个结果（显示前10）：\n")
        for i, r in enumerate(result.debug_info.vector_results[:10]):
            chunk_id = r.get('chunk_id', '')
            debug_parts.append(f"{i+1}. **{r['filename']}** `[{chunk_id}]` — similarity: `{r['score']:.4f}`")
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
            chunk_id = r.get('chunk_id', '')
            debug_parts.append(f"{i+1}. {status} **{r['filename']}** `[{chunk_id}]` — `{score_pct:.1f}%`")
        debug_parts.append("")
        debug_parts.append("---")
        debug_parts.append("")
    
    # Final results sent to LLM
    if result.debug_info.hybrid_results or result.debug_info.reranked_results:
        debug_parts.append("## ✅ 最终传给 LLM 的结果")
        if result.debug_info.final_results:
            debug_parts.append(f"共 {len(result.debug_info.final_results)} 个文档：\n")
            for i, r in enumerate(result.debug_info.final_results):
                score_pct = r['score'] * 100
                chunk_id = r.get('chunk_id', '')
                debug_parts.append(f"{i+1}. **{r['filename']}** `[{chunk_id}]` — `{score_pct:.1f}%`")
            debug_parts.append("")
        else:
            debug_parts.append("（无结果通过过滤）")
            debug_parts.append("")
        debug_parts.append("---")
        debug_parts.append("")
    
    # LLM Input
    debug_parts.append("## 📝 LLM 输入上下文")
    debug_parts.append(f"```\n{result.debug_info.llm_input}\n```")
    
    return "\n".join(debug_parts)


def handle_query(
    question: str,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    show_debug: bool = False
) -> Tuple[str, str, str]:
    """
    Handle user question (non-streaming version).

    Args:
        question: User's question
        use_hybrid: If True, use hybrid search (BM25 + Vector)
        use_rerank: If True, use Cohere reranking
        show_debug: If True, return debug information

    Returns:
        Tuple of (answer, source_chunks, debug_info)
    """
    if not question.strip():
        return "请输入问题。", "", ""

    try:
        result = query(question, use_hybrid=use_hybrid, use_rerank=use_rerank)
        sources_text = _format_sources(result)
        debug_text = _format_debug_info(result, show_debug)
        return result.answer, sources_text, debug_text
    except Exception as e:
        return f"Error: {str(e)}", "", ""


def handle_query_stream(
    question: str,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    show_debug: bool = False
) -> Generator[Tuple[str, str, str], None, None]:
    """
    Handle user question with streaming response.

    Args:
        question: User's question
        use_hybrid: If True, use hybrid search (BM25 + Vector)
        use_rerank: If True, use Cohere reranking
        show_debug: If True, return debug information

    Yields:
        Tuple of (answer, source_chunks, debug_info)
    """
    if not question.strip():
        yield "请输入问题。", "", ""
        return

    try:
        sources_text = ""
        debug_text = ""
        
        for partial_answer, final_result in query_stream(
            question, 
            use_hybrid=use_hybrid, 
            use_rerank=use_rerank
        ):
            if final_result is not None:
                # Final result - format sources and debug info
                sources_text = _format_sources(final_result)
                debug_text = _format_debug_info(final_result, show_debug)
            
            yield partial_answer, sources_text, debug_text
            
    except Exception as e:
        yield f"Error: {str(e)}", "", ""


def create_app() -> gr.Blocks:
    """Create and configure the Gradio application."""
    with gr.Blocks(title="RAG Notes - Q&A Assistant") as app:
        # Header with title and tracing link
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("# RAG Notes Q&A Assistant")
            with gr.Column(scale=1, min_width=150):
                gr.HTML(
                    '<a href="http://localhost:6006" target="_blank" '
                    'style="display: inline-flex; align-items: center; gap: 6px; '
                    'padding: 8px 16px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
                    'color: white; text-decoration: none; border-radius: 8px; '
                    'font-weight: 500; font-size: 14px; float: right; margin-top: 8px;">'
                    '🔭 Phoenix Tracing</a>'
                )
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
                    value=True,
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
            with gr.Accordion("🔧 调试信息 (Debug Info)", open=False, visible=True):
                debug_output = gr.Markdown(
                    label="Debug Information",
                    value="开启「显示调试信息」选项后，这里会显示检索过程的详细信息。"
                )

            # Use streaming response for better UX
            ask_btn.click(
                fn=handle_query_stream,
                inputs=[question_input, hybrid_search_checkbox, rerank_checkbox, debug_checkbox],
                outputs=[answer_output, sources_output, debug_output],
            )

            # Also trigger on Enter key with streaming
            question_input.submit(
                fn=handle_query_stream,
                inputs=[question_input, hybrid_search_checkbox, rerank_checkbox, debug_checkbox],
                outputs=[answer_output, sources_output, debug_output],
            )

        with gr.Tab("💬 Chat (多轮对话)"):
            gr.Markdown("""
### 多轮对话模式
支持上下文理解，可以进行连续追问。系统会自动改写查询以解决代词指代问题（如"它"、"这个"等）。
            """)
            
            # Chat configuration
            with gr.Row():
                chat_hybrid_checkbox = gr.Checkbox(
                    label="混合检索 (BM25 + Vector)",
                    value=False,
                    info="使用BM25全文检索与向量检索的混合检索"
                )
                chat_rerank_checkbox = gr.Checkbox(
                    label="重排序 (Cohere Rerank)",
                    value=False,
                    info="使用Cohere Rerank API对检索结果进行重排序"
                )
                chat_rewrite_checkbox = gr.Checkbox(
                    label="🔄 查询改写",
                    value=True,
                    info="基于对话历史自动改写查询，解决代词指代问题"
                )
            
            # State to store chat history
            chat_history_state = gr.State(value=None)
            
            # Chatbot component
            chatbot = gr.Chatbot(
                label="对话",
                height=400,
            )
            
            # Input area
            with gr.Row():
                chat_input = gr.Textbox(
                    label="输入消息",
                    placeholder="输入你的问题...",
                    lines=1,
                    scale=4,
                )
                chat_submit_btn = gr.Button("发送", variant="primary", scale=1)
            
            # Show rewritten query
            with gr.Accordion("🔍 查询改写信息", open=False):
                rewrite_info = gr.Markdown("查询改写信息将在这里显示。")
            
            # Clear button
            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
            
            def handle_chat_message(
                message: str,
                history: list,
                chat_hist_obj: ChatHistory,
                use_hybrid: bool,
                use_rerank: bool,
                enable_rewrite: bool
            ) -> Generator[Tuple[list, ChatHistory, str], None, None]:
                """Handle chat message with streaming."""
                if not message.strip():
                    yield history or [], chat_hist_obj, ""
                    return
                
                # Initialize chat history if needed
                if chat_hist_obj is None:
                    chat_hist_obj = ChatHistory()
                
                # Ensure history is a list
                if history is None:
                    history = []
                
                try:
                    # Add user message to display (messages format for new Gradio)
                    history = history + [
                        {"role": "user", "content": message},
                        {"role": "assistant", "content": ""}
                    ]
                    rewrite_text = ""
                    
                    for partial_answer, final_result, rewritten_query in chat_stream(
                        message,
                        chat_hist_obj,
                        use_hybrid=use_hybrid,
                        use_rerank=use_rerank,
                        enable_rewrite=enable_rewrite
                    ):
                        # Update the last message with streaming content
                        history[-1] = {"role": "assistant", "content": partial_answer}
                        
                        # Update rewrite info if query was rewritten
                        if rewritten_query and rewritten_query != message:
                            rewrite_text = f"**原始查询:** {message}\n\n**改写后查询:** {rewritten_query}"
                        elif not rewritten_query:
                            rewrite_text = "（无需改写，使用原始查询）"
                        
                        yield history, chat_hist_obj, rewrite_text
                    
                    # Add sources info to the response if available
                    if final_result and final_result.source_files:
                        sources_summary = ", ".join(set(final_result.source_files))
                        current_response = history[-1]["content"]
                        history[-1] = {"role": "assistant", "content": current_response + f"\n\n📚 *来源: {sources_summary}*"}
                        yield history, chat_hist_obj, rewrite_text
                        
                except Exception as e:
                    history[-1] = {"role": "assistant", "content": f"❌ 错误: {str(e)}"}
                    yield history, chat_hist_obj, ""
            
            def clear_chat(chat_hist_obj: ChatHistory) -> Tuple[list, ChatHistory, str]:
                """Clear chat history."""
                if chat_hist_obj is not None:
                    chat_hist_obj.clear()
                return [], ChatHistory(), ""
            
            # Wire up events
            chat_submit_btn.click(
                fn=handle_chat_message,
                inputs=[chat_input, chatbot, chat_history_state, chat_hybrid_checkbox, chat_rerank_checkbox, chat_rewrite_checkbox],
                outputs=[chatbot, chat_history_state, rewrite_info],
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=chat_input,
            )
            
            chat_input.submit(
                fn=handle_chat_message,
                inputs=[chat_input, chatbot, chat_history_state, chat_hybrid_checkbox, chat_rerank_checkbox, chat_rewrite_checkbox],
                outputs=[chatbot, chat_history_state, rewrite_info],
            ).then(
                fn=lambda: "",
                inputs=None,
                outputs=chat_input,
            )
            
            clear_btn.click(
                fn=clear_chat,
                inputs=[chat_history_state],
                outputs=[chatbot, chat_history_state, rewrite_info],
            )

    return app
