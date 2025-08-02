"""RAG (Retrieval-Augmented Generation) engine for semantic search."""

from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import settings
from .utils import chunk_text, logger, load_json, save_json


class RAGEngine:
    """Semantic search and retrieval engine."""
    
    def __init__(self, persist_directory: Optional[Path] = None):
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            metadata={"description": "AI/ML research paper embeddings"}
        )
        
        logger.info(f"RAG engine initialized with {self.collection.count()} documents")
    
    def add_paper(self, paper: Dict[str, Any], summary: str) -> bool:
        """Add a paper to the vector store."""
        try:
            # Prepare text for embedding
            text_content = f"""
Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Abstract: {paper['abstract']}
Summary: {summary}
Categories: {', '.join(paper['categories'])}
            """.strip()
            
            # Generate chunks
            chunks = chunk_text(text_content, chunk_size=500, overlap=50)
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{paper['paper_id']}_{i}"
                
                documents.append(chunk)
                ids.append(chunk_id)
                metadatas.append({
                    "paper_id": paper['paper_id'],
                    "title": paper['title'],
                    "authors": json.dumps(paper['authors']),
                    "published": paper['published'],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "arxiv_url": paper['arxiv_url'],
                    "pdf_url": paper['pdf_url']
                })
            
            # Add to collection
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added paper '{paper['title']}' with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper to RAG: {e}")
            return False
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Semantic search for papers."""
        try:
            # Perform search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 3,  # Get more to deduplicate
                where=filter_dict
            )
            
            # Deduplicate by paper_id
            seen_papers = set()
            unique_results = []
            
            for i, metadata in enumerate(results['metadatas'][0]):
                paper_id = metadata['paper_id']
                
                if paper_id not in seen_papers:
                    seen_papers.add(paper_id)
                    
                    # Parse authors from JSON
                    authors = json.loads(metadata['authors'])
                    
                    result = {
                        'paper_id': paper_id,
                        'title': metadata['title'],
                        'authors': authors,
                        'published': metadata['published'],
                        'arxiv_url': metadata['arxiv_url'],
                        'pdf_url': metadata['pdf_url'],
                        'relevance_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                        'matched_text': results['documents'][0][i][:200] + '...'
                    }
                    unique_results.append(result)
                    
                    if len(unique_results) >= n_results:
                        break
            
            return unique_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def find_similar_papers(self, paper_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Find papers similar to a given paper."""
        try:
            # Get the paper's chunks
            paper_results = self.collection.get(
                where={"paper_id": paper_id},
                limit=1
            )
            
            if not paper_results['documents']:
                logger.warning(f"Paper {paper_id} not found in collection")
                return []
            
            # Use the first chunk as query
            query_text = paper_results['documents'][0]
            
            # Search for similar papers, excluding the source paper
            results = self.search(
                query_text,
                n_results=n_results + 1,
                filter_dict={"paper_id": {"$ne": paper_id}}
            )
            
            return results[:n_results]
            
        except Exception as e:
            logger.error(f"Error finding similar papers: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            # Get unique paper count
            all_metadata = self.collection.get()['metadatas']
            unique_papers = set(m['paper_id'] for m in all_metadata)
            
            return {
                'total_chunks': self.collection.count(),
                'unique_papers': len(unique_papers),
                'embedding_model': settings.embedding_model,
                'persist_directory': str(self.persist_directory)
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def clear_collection(self) -> bool:
        """Clear all data from the collection."""
        try:
            # Delete and recreate collection
            self.client.delete_collection("research_papers")
            self.collection = self.client.create_collection(
                name="research_papers",
                metadata={"description": "AI/ML research paper embeddings"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False