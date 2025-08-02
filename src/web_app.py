"""FastAPI web application for AI Research Summarizer."""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uvicorn

from .scraper import ArxivScraper
from .summarizer import create_summarizer, MultiModelSummarizer
from .rag_engine import RAGEngine
from .config import settings
from .utils import logger


# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query for ArXiv papers")
    max_results: int = Field(default=10, ge=1, le=50)
    days_back: Optional[int] = Field(default=7, ge=1, le=30)
    include_summaries: bool = Field(default=True)


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    limit: int = Field(default=5, ge=1, le=20)


class PaperResponse(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    categories: List[str]
    arxiv_url: str
    pdf_url: str
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    papers: List[PaperResponse]


# Initialize FastAPI app
app = FastAPI(
    title="AI Research Paper Summarizer",
    description="Fetch, summarize, and search AI/ML research papers from ArXiv",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
scraper = ArxivScraper()
rag_engine = RAGEngine()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Research Summarizer</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
            code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>🧠 AI Research Paper Summarizer API</h1>
        <p>Welcome to the AI Research Paper Summarizer API. Use the endpoints below:</p>
        
        <div class="endpoint">
            <h3>📚 Fetch Papers</h3>
            <code>POST /papers/search</code>
            <p>Search and fetch recent papers from ArXiv</p>
        </div>
        
        <div class="endpoint">
            <h3>🔍 Semantic Search</h3>
            <code>POST /papers/semantic-search</code>
            <p>Search through indexed papers using semantic similarity</p>
        </div>
        
        <div class="endpoint">
            <h3>📊 Statistics</h3>
            <code>GET /stats</code>
            <p>Get database statistics</p>
        </div>
        
        <div class="endpoint">
            <h3>🔥 Trending Topics</h3>
            <code>GET /topics</code>
            <p>Get trending AI/ML research topics</p>
        </div>
        
        <div class="endpoint">
            <h3>📖 API Documentation</h3>
            <code>GET /docs</code>
            <p>Interactive API documentation (Swagger UI)</p>
        </div>
    </body>
    </html>
    """


@app.post("/papers/search", response_model=SearchResponse)
async def search_papers(
    request: SearchRequest,
    background_tasks: BackgroundTasks
):
    """Search for papers on ArXiv and optionally generate summaries."""
    try:
        # Search papers
        papers = scraper.search_papers(
            query=request.query,
            max_results=request.max_results,
            days_back=request.days_back
        )
        
        if not papers:
            raise HTTPException(status_code=404, detail="No papers found")
        
        # Prepare response
        paper_responses = []
        
        for paper in papers:
            paper_resp = PaperResponse(**paper)
            
            # Generate summaries if requested
            if request.include_summaries:
                summarizer = create_summarizer()
                paper_resp.summary = summarizer.summarize(paper['abstract'])
                paper_resp.key_points = summarizer.extract_key_points(paper['abstract'])
                
                # Add to RAG engine in background
                background_tasks.add_task(
                    rag_engine.add_paper,
                    paper,
                    paper_resp.summary
                )
            
            paper_responses.append(paper_resp)
        
        return SearchResponse(
            query=request.query,
            total_results=len(papers),
            papers=paper_responses
        )
        
    except Exception as e:
        logger.error(f"Error in search_papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/papers/semantic-search")
async def semantic_search(request: SemanticSearchRequest):
    """Search through indexed papers using semantic similarity."""
    try:
        results = rag_engine.search(request.query, n_results=request.limit)
        
        if not results:
            raise HTTPException(status_code=404, detail="No matching papers found")
        
        return {
            "query": request.query,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in semantic_search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers/{paper_id}/similar")
async def find_similar_papers(
    paper_id: str,
    limit: int = Query(default=5, ge=1, le=20)
):
    """Find papers similar to a given paper."""
    try:
        results = rag_engine.find_similar_papers(paper_id, n_results=limit)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No similar papers found for ID: {paper_id}")
        
        return {
            "paper_id": paper_id,
            "similar_papers": results
        }
        
    except Exception as e:
        logger.error(f"Error in find_similar_papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_statistics():
    """Get database statistics."""
    return rag_engine.get_statistics()


@app.get("/topics")
async def get_trending_topics():
    """Get trending AI/ML research topics."""
    return {
        "topics": scraper.get_trending_topics(),
        "updated": datetime.now().isoformat()
    }


@app.post("/papers/summarize")
async def summarize_text(
    text: str = Query(..., description="Text to summarize"),
    model: str = Query(default="auto", description="Model to use: openai, local, or auto"),
    max_length: int = Query(default=500, ge=100, le=2000)
):
    """Generate a summary for provided text."""
    try:
        summarizer = create_summarizer(model if model != "auto" else None)
        summary = summarizer.summarize(text, max_length)
        key_points = summarizer.extract_key_points(text)
        
        return {
            "summary": summary,
            "key_points": key_points,
            "model": model
        }
        
    except Exception as e:
        logger.error(f"Error in summarize_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_stats": rag_engine.get_statistics()
    }


if __name__ == "__main__":
    uvicorn.run(
        "src.web_app:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )