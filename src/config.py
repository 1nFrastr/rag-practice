"""Configuration module for RAG Notes application."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""

    # OpenRouter API
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Cohere API (for reranking)
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

    # Model configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI model name (without openai/ prefix)
    LLM_MODEL: str = "openai/gpt-4o-mini"
    EMBEDDING_DIM: int = 1536  # text-embedding-3-small dimension

    # PostgreSQL configuration
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "rag_notes")

    # Document processing
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required. Please set it in .env file.")
        return True

    @classmethod
    def validate_rerank(cls) -> bool:
        """Validate that Cohere API key is present for reranking."""
        if not cls.COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is required for reranking. Please set it in .env file.")
        return True


config = Config()
