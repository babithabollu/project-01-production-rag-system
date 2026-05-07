"""Application configuration loaded from environment variables."""
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All runtime configuration for the RAG system."""

    # API Keys
    openai_api_key: str
    cohere_api_key: str

    @field_validator("openai_api_key", "cohere_api_key", mode="before")
    @classmethod
    def strip_key(cls, v: str) -> str:
        return v.strip()

    # ChromaDB connection
    chroma_host: str = "localhost"
    chroma_port: int = 8001

    # Chunking
    chunk_size: int = 700
    chunk_overlap: int = 100

    # Retrieval
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "gpt-4o-mini"
    rerank_model: str = "rerank-english-v3.0"

    # Evaluation thresholds
    min_faithfulness: float = 0.85
    min_precision_at_5: float = 0.70

    class Config:
        env_file = ".env"


settings = Settings()
