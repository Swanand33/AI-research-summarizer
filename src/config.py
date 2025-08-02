"""Configuration management for AI Research Summarizer."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings  # Changed this line!


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Model Configuration
    default_model: str = "gpt-3.5-turbo"
    embedding_model: str = "all-MiniLM-L6-v2"
    use_local_models: bool = False
    
    # ArXiv Settings
    arxiv_max_results: int = 10
    arxiv_sort_by: str = "submittedDate"
    arxiv_sort_order: str = "descending"
    
    # Summary Settings
    summary_max_length: int = 500
    summary_style: str = "technical"  # technical, simple, bullet_points
    
    # Vector Store
    chroma_persist_directory: Path = Path("./data/chroma_db")
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Paths
    data_dir: Path = Path("./data")
    papers_dir: Path = Path("./data/papers")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.papers_dir.mkdir(exist_ok=True)
        self.chroma_persist_directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()