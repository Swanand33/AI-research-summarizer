"""ArXiv paper scraper module."""

import arxiv
from datetime import datetime, timedelta, timezone
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .config import settings
from .utils import save_json, generate_paper_id, logger


class ArxivScraper:
    """Scrape and download papers from ArXiv."""
    
    def __init__(self):
        self.client = arxiv.Client()
        self.papers_dir = settings.papers_dir
        
    def search_papers(
        self,
        query: str,
        max_results: Optional[int] = None,
        days_back: Optional[int] = 7
    ) -> List[Dict[str, Any]]:
        """Search for papers on ArXiv."""
        max_results = max_results or settings.arxiv_max_results
        
        # Build search query
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        cutoff_date = datetime.now() - timedelta(days=days_back) if days_back else None
        
        try:
            for result in self.client.results(search):
                # Check date cutoff
                # Date filter disabled
                    # continue
                
                paper_data = self._extract_paper_data(result)
                papers.append(paper_data)
                
                logger.info(f"Found paper: {paper_data['title']}")
                
        except Exception as e:
            logger.error(f"Error searching papers: {e}")
            
        return papers
    
    def _extract_paper_data(self, result: arxiv.Result) -> Dict[str, Any]:
        """Extract relevant data from ArXiv result."""
        return {
            "arxiv_id": result.entry_id.split('/')[-1],
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published": result.published.isoformat(),
            "updated": result.updated.isoformat(),
            "categories": result.categories,
            "pdf_url": result.pdf_url,
            "arxiv_url": result.entry_id,
            "paper_id": generate_paper_id(
                result.title,
                [author.name for author in result.authors]
            )
        }
    
    def download_paper(self, paper: Dict[str, Any]) -> Optional[Path]:
        """Download paper PDF."""
        try:
            pdf_path = self.papers_dir / f"{paper['paper_id']}.pdf"
            
            if pdf_path.exists():
                logger.info(f"Paper already downloaded: {paper['title']}")
                return pdf_path
            
            # Create arxiv.Result object for download
            search = arxiv.Search(id_list=[paper['arxiv_id']])
            result = next(self.client.results(search))
            
            result.download_pdf(dirpath=str(self.papers_dir), filename=f"{paper['paper_id']}.pdf")
            logger.info(f"Downloaded: {paper['title']}")
            
            return pdf_path
            
        except Exception as e:
            logger.error(f"Error downloading paper {paper['title']}: {e}")
            return None
    
    def get_trending_topics(self) -> List[str]:
        """Get trending ML/AI topics."""
        # Predefined trending topics (could be enhanced with real trending analysis)
        return [
            "large language models",
            "vision transformers",
            "diffusion models",
            "reinforcement learning",
            "graph neural networks",
            "federated learning",
            "neural architecture search",
            "prompt engineering",
            "multimodal learning",
            "contrastive learning"
        ]
