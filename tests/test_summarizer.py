"""Tests for the summarizer module."""

import pytest
from unittest.mock import Mock, patch

from src.summarizer import OpenAISummarizer, LocalSummarizer, create_summarizer


class TestOpenAISummarizer:
    """Test cases for OpenAI summarizer."""
    
    def test_summarize_without_api_key(self):
        """Test summarization without API key."""
        summarizer = OpenAISummarizer(api_key=None)
        result = summarizer.summarize("Test text")
        
        assert "OpenAI API key not configured" in result
    
    @patch('openai.OpenAI')
    def test_summarize_with_api_key(self, mock_openai):
        """Test successful summarization."""
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test summary"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        summarizer = OpenAISummarizer(api_key="test-key")
        result = summarizer.summarize("Test abstract text")
        
        assert result == "Test summary"
        mock_client.chat.completions.create.assert_called_once()


class TestLocalSummarizer:
    """Test cases for local summarizer."""
    
    def test_extract_key_points(self):
        """Test key point extraction."""
        summarizer = LocalSummarizer()
        text = "We propose a new method. Our contribution is significant. Results show improvement."
        
        points = summarizer.extract_key_points(text)
        
        assert isinstance(points, list)
        assert len(points) > 0
        assert any("propose" in point.lower() for point in points)