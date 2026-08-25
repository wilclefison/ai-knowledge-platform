import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Knowledge Platform"
    VERSION: str = "0.2.0"
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/knowledge_db",
        env="DATABASE_URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL"
    )
    
    # AI & Embeddings
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    
    # Chunking Parameters
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    
    # Hybrid Search Weights
    RRF_K: int = 60  # Smoothing constant for Reciprocal Rank Fusion
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
