"""AI Research Paper Summarizer package."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .scraper import ArxivScraper
from .summarizer import create_summarizer, MultiModelSummarizer
from .rag_engine import RAGEngine
from .config import settings

__all__ = [
    "ArxivScraper",
    "create_summarizer",
    "MultiModelSummarizer",
    "RAGEngine",
    "settings"
]