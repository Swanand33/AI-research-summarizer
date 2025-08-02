"""Utility functions for the AI Research Summarizer."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\-.,!?;:()\[\]{}\'"]', '', text)
    return text.strip()


def generate_paper_id(title: str, authors: List[str]) -> str:
    """Generate a unique ID for a paper."""
    content = f"{title}_{','.join(authors)}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(filepath: Path) -> Any:
    """Load data from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_paper_metadata(paper: Dict[str, Any]) -> Dict[str, Any]:
    """Format paper metadata for display."""
    return {
        "id": paper.get("id", ""),
        "title": paper.get("title", ""),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract", ""),
        "published": paper.get("published", ""),
        "categories": paper.get("categories", []),
        "pdf_url": paper.get("pdf_url", ""),
        "arxiv_url": paper.get("arxiv_url", "")
    }


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at a sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size - 200:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


logger = setup_logging()