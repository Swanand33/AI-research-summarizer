"""Tests for the RAG engine module."""

import pytest
from unittest.mock import Mock, patch
import json

from src.rag_engine import RAGEngine


class TestRAGEngine:
    """Test cases for RAG engine."""
    
    @pytest.fixture
    def rag_engine(self, tmp_path):
        """Create a RAG engine instance with temporary storage."""
        return RAGEngine(persist_directory=tmp_path / "test_chroma")
    
    def test_add_paper(self, rag_engine):
        """Test adding a paper to the vector store."""
        paper = {
            "paper_id": "test123",
            "title": "Test Paper",
            "authors": ["Author 1", "Author 2"],
            "abstract": "Test abstract",
            "published": "2024-01-01",
            "categories": ["cs.AI"],
            "arxiv_url": "http://arxiv.org/abs/test123",
            "pdf_url": "http://example.com/pdf"
        }
        
        result = rag_engine.add_paper(paper, "Test summary")
        assert result is True
    
    def test_search_empty_collection(self, rag_engine):
        """Test searching an empty collection."""
        results = rag_engine.search("test query")
        assert results == []
    
    def test_get_statistics(self, rag_engine):
        """Test getting statistics."""
        stats = rag_engine.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_chunks' in stats
        assert 'unique_papers' in stats
        assert stats['total_chunks'] >= 0