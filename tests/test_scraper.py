"""Tests for the ArXiv scraper module."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.scraper import ArxivScraper


class TestArxivScraper:
    """Test cases for ArxivScraper."""
    
    @pytest.fixture
    def scraper(self):
        """Create a scraper instance."""
        return ArxivScraper()
    
    def test_search_papers_success(self, scraper):
        """Test successful paper search."""
        with patch.object(scraper, 'client') as mock_client:
            # Mock ArXiv results
            mock_result = Mock()
            mock_result.title = "Test Paper"
            mock_result.authors = [Mock(name="Author 1"), Mock(name="Author 2")]
            mock_result.summary = "Test abstract"
            mock_result.published = datetime.now()
            mock_result.updated = datetime.now()
            mock_result.categories = ["cs.AI", "cs.LG"]
            mock_result.pdf_url = "http://example.com/pdf"
            mock_result.entry_id = "http://arxiv.org/abs/1234.5678"
            
            mock_client.results.return_value = [mock_result]
            
            papers = scraper.search_papers("test query", max_results=1)
            
            assert len(papers) == 1
            assert papers[0]['title'] == "Test Paper"
            assert len(papers[0]['authors']) == 2
            assert papers[0]['arxiv_id'] == "1234.5678"
    
    def test_search_papers_empty_results(self, scraper):
        """Test search with no results."""
        with patch.object(scraper, 'client') as mock_client:
            mock_client.results.return_value = []
            
            papers = scraper.search_papers("nonexistent topic")
            assert papers == []
    
    def test_trending_topics(self, scraper):
        """Test getting trending topics."""
        topics = scraper.get_trending_topics()
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        assert "large language models" in topics