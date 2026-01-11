"""Chat engine module for multi-turn conversation support."""

from dataclasses import dataclass, field
from typing import List, Generator, Tuple, Optional
from datetime import datetime

from .config import config
from .query_engine import get_llm, query_stream, QueryResult
from .tracing import traced_operation, SpanKind


@dataclass
class ChatMessage:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class ChatHistory:
    """Manages conversation history for multi-turn chat."""
    
    def __init__(self, max_turns: int = 10):
        """
        Initialize chat history.
        
        Args:
            max_turns: Maximum number of conversation turns to keep
        """
        self.messages: List[ChatMessage] = []
        self.max_turns = max_turns
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.messages.append(ChatMessage(role="user", content=content))
        self._trim_history()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to history."""
        self.messages.append(ChatMessage(role="assistant", content=content))
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Keep only the last max_turns * 2 messages (user + assistant pairs)."""
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
    
    def get_context_string(self) -> str:
        """Get conversation history as a formatted string."""
        if not self.messages:
            return ""
        
        parts = []
        for msg in self.messages:
            role_label = "用户" if msg.role == "user" else "助手"
            parts.append(f"{role_label}: {msg.content}")
        
        return "\n".join(parts)
    
    def get_messages_for_rewrite(self) -> List[dict]:
        """Get recent messages for query rewriting context."""
        # Return last few turns for context
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages[-6:]  # Last 3 turns
        ]
    
    def clear(self) -> None:
        """Clear all conversation history."""
        self.messages = []
    
    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self.messages) == 0


# Query rewriting prompt template
QUERY_REWRITE_PROMPT = """你是一个查询改写助手。你的任务是将用户的当前问题改写成一个独立的、完整的搜索查询。

对话历史：
{history}

用户当前问题：{question}

改写规则：
1. 如果当前问题包含代词（如"它"、"这个"、"那个"、"他们"等），根据对话历史将其替换为具体的实体名称
2. 如果当前问题是对之前话题的追问，补充必要的上下文使其成为独立问题
3. 如果当前问题已经是完整独立的，直接返回原问题
4. 只返回改写后的问题，不要添加任何解释

改写后的问题："""


def rewrite_query_with_history(
    question: str,
    chat_history: ChatHistory
) -> str:
    """
    Rewrite the user's question based on conversation history.
    
    This resolves pronouns and references to make the query self-contained.
    
    Args:
        question: The user's current question
        chat_history: The conversation history
        
    Returns:
        The rewritten query
    """
    # If no history, return the question as-is
    if chat_history.is_empty():
        return question
    
    with traced_operation(
        "query_rewrite",
        span_kind=SpanKind.LLM,
        input_value=question,
        attributes={"history_turns": len(chat_history.messages) // 2}
    ):
        llm = get_llm()
        
        # Format the prompt
        history_str = chat_history.get_context_string()
        prompt = QUERY_REWRITE_PROMPT.format(
            history=history_str,
            question=question
        )
        
        # Call LLM for rewriting
        response = llm.complete(prompt)
        rewritten = str(response).strip()
        
        # If the response is empty or too long, use original
        if not rewritten or len(rewritten) > len(question) * 3:
            return question
        
        return rewritten


def chat_stream(
    question: str,
    chat_history: ChatHistory,
    use_hybrid: bool = False,
    use_rerank: bool = False,
    enable_rewrite: bool = True
) -> Generator[Tuple[str, QueryResult | None, str | None], None, None]:
    """
    Chat with streaming response, supporting multi-turn conversation.
    
    Args:
        question: User's current question
        chat_history: The conversation history
        use_hybrid: If True, use hybrid search (BM25 + Vector)
        use_rerank: If True, use Cohere reranking
        enable_rewrite: If True, rewrite query based on history
        
    Yields:
        Tuple of (partial_answer, final_result, rewritten_query)
        - During streaming: (partial_text, None, rewritten_query)
        - At the end: (full_text, QueryResult, rewritten_query)
    """
    with traced_operation(
        "chat_stream",
        span_kind=SpanKind.CHAIN,
        input_value=question,
        attributes={
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "enable_rewrite": enable_rewrite,
            "history_turns": len(chat_history.messages) // 2
        }
    ):
        # Step 1: Rewrite query if enabled and has history
        rewritten_query = None
        search_query = question
        
        if enable_rewrite and not chat_history.is_empty():
            rewritten_query = rewrite_query_with_history(question, chat_history)
            if rewritten_query != question:
                search_query = rewritten_query
        
        # Step 2: Add user message to history
        chat_history.add_user_message(question)
        
        # Step 3: Stream the response using the rewritten query
        full_response = ""
        final_result = None
        
        for partial_answer, result in query_stream(
            search_query,
            use_hybrid=use_hybrid,
            use_rerank=use_rerank
        ):
            full_response = partial_answer
            if result is not None:
                final_result = result
            yield (partial_answer, result, rewritten_query)
        
        # Step 4: Add assistant response to history
        if full_response:
            chat_history.add_assistant_message(full_response)


# Global chat history storage (for simple in-memory session management)
# In production, you might want to use a database or session-based storage
_chat_sessions: dict[str, ChatHistory] = {}


def get_or_create_session(session_id: str = "default") -> ChatHistory:
    """Get or create a chat session."""
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = ChatHistory()
    return _chat_sessions[session_id]


def clear_session(session_id: str = "default") -> None:
    """Clear a chat session."""
    if session_id in _chat_sessions:
        _chat_sessions[session_id].clear()


def delete_session(session_id: str) -> None:
    """Delete a chat session."""
    if session_id in _chat_sessions:
        del _chat_sessions[session_id]
