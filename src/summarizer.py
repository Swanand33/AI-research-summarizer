"""LLM-based paper summarization module."""

import os
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging

from .config import settings
from .utils import clean_text, chunk_text, logger


class BaseSummarizer(ABC):
    """Base class for paper summarizers."""
    
    @abstractmethod
    def summarize(self, text: str, max_length: int = 500) -> str:
        """Generate summary of text."""
        pass
    
    @abstractmethod
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text."""
        pass


class OpenAISummarizer(BaseSummarizer):
    """OpenAI-based summarizer."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        if self.api_key:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("OpenAI API key not provided")
    
    def summarize(self, text: str, max_length: int = 500) -> str:
        """Generate summary using OpenAI."""
        if not self.client:
            return "OpenAI API key not configured. Please set OPENAI_API_KEY."
        
        try:
            prompt = f"""Summarize this research paper abstract in {max_length} characters or less.
Focus on the main contribution, methodology, and key findings.

Abstract: {text[:3000]}

Summary:"""
            
            response = self.client.chat.completions.create(
                model=settings.default_model,
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing AI/ML research papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length // 4,  # Rough token estimation
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error in OpenAI summarization: {e}")
            return f"Error generating summary: {str(e)}"
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points using OpenAI."""
        if not self.client:
            return ["OpenAI API key not configured"]
        
        try:
            prompt = f"""Extract 3-5 key points from this research paper abstract.
Each point should be a complete sentence.

Abstract: {text[:3000]}

Key Points:"""
            
            response = self.client.chat.completions.create(
                model=settings.default_model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing AI/ML research papers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            # Parse bullet points
            points = [line.strip('- •*').strip() 
                     for line in content.split('\n') 
                     if line.strip() and line.strip()[0] in '-•*123456789']
            
            return points[:5]  # Limit to 5 points
            
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            return [f"Error: {str(e)}"]


class LocalSummarizer(BaseSummarizer):
    """Local model summarizer using Hugging Face."""
    
    def __init__(self):
        try:
            from transformers import pipeline
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # CPU
            )
        except Exception as e:
            logger.error(f"Error loading local model: {e}")
            self.summarizer = None
    
    def summarize(self, text: str, max_length: int = 500) -> str:
        """Generate summary using local model."""
        if not self.summarizer:
            return "Local model not available. Install transformers library."
        
        try:
            # Chunk text if too long
            chunks = chunk_text(text, chunk_size=1024, overlap=100)
            summaries = []
            
            for chunk in chunks[:3]:  # Limit chunks
                summary = self.summarizer(
                    chunk,
                    max_length=max_length // len(chunks[:3]),
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                summaries.append(summary)
            
            return " ".join(summaries)[:max_length]
            
        except Exception as e:
            logger.error(f"Error in local summarization: {e}")
            return f"Error generating summary: {str(e)}"
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points using pattern matching."""
        # Simple implementation - could be enhanced
        sentences = text.split('. ')
        
        # Look for sentences with key phrases
        key_phrases = [
            "we propose", "we present", "we introduce",
            "our contribution", "we demonstrate", "we show",
            "results show", "experiments demonstrate",
            "outperforms", "achieves", "improves"
        ]
        
        key_points = []
        for sentence in sentences:
            if any(phrase in sentence.lower() for phrase in key_phrases):
                key_points.append(sentence.strip() + '.')
        
        return key_points[:5]


class MultiModelSummarizer:
    """Combine multiple summarizers for comparison."""
    
    def __init__(self):
        self.summarizers = {
            "openai": OpenAISummarizer(),
            "local": LocalSummarizer()
        }
    
    def summarize_all(self, text: str, max_length: int = 500) -> Dict[str, str]:
        """Get summaries from all available models."""
        results = {}
        
        for name, summarizer in self.summarizers.items():
            try:
                results[name] = summarizer.summarize(text, max_length)
            except Exception as e:
                results[name] = f"Error with {name}: {str(e)}"
        
        return results
    
    def get_best_summary(self, text: str, max_length: int = 500) -> str:
        """Get summary from the best available model."""
        # Try OpenAI first, fall back to local
        if settings.openai_api_key:
            return self.summarizers["openai"].summarize(text, max_length)
        else:
            return self.summarizers["local"].summarize(text, max_length)


def create_summarizer(model_type: Optional[str] = None) -> BaseSummarizer:
    """Factory function to create appropriate summarizer."""
    model_type = model_type or ("openai" if settings.openai_api_key else "local")
    
    if model_type == "openai":
        return OpenAISummarizer()
    elif model_type == "local":
        return LocalSummarizer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")