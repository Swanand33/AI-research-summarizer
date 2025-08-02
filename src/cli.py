"""Command-line interface for AI Research Summarizer."""

import click
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List
import sys

from .scraper import ArxivScraper
from .summarizer import create_summarizer, MultiModelSummarizer
from .rag_engine import RAGEngine
from .config import settings
from .utils import save_json, load_json, logger, format_paper_metadata


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """AI Research Paper Summarizer - Fetch, summarize, and search AI/ML papers."""
    pass


@cli.command()
@click.option('--topic', '-t', required=True, help='Search topic (e.g., "large language models")')
@click.option('--count', '-c', default=5, help='Number of papers to fetch')
@click.option('--days', '-d', default=7, help='Papers from last N days')
@click.option('--summarize', '-s', is_flag=True, help='Generate summaries')
@click.option('--model', '-m', type=click.Choice(['openai', 'local', 'all']), default='local', help='Summarization model')
def fetch(topic: str, count: int, days: int, summarize: bool, model: str):
    """Fetch recent papers from ArXiv."""
    click.echo(f"🔍 Searching for papers on '{topic}'...")
    
    scraper = ArxivScraper()
    papers = scraper.search_papers(topic, max_results=count, days_back=days)
    
    if not papers:
        click.echo("❌ No papers found matching your criteria.")
        return
    
    click.echo(f"✅ Found {len(papers)} papers\n")
    
    # Display papers
    for i, paper in enumerate(papers, 1):
        click.echo(f"{i}. {paper['title']}")
        click.echo(f"   Authors: {', '.join(paper['authors'][:3])}{' et al.' if len(paper['authors']) > 3 else ''}")
        click.echo(f"   Published: {paper['published'][:10]}")
        click.echo(f"   ArXiv: {paper['arxiv_url']}\n")
    
    # Save metadata
    output_file = Path(f"data/papers_{topic.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_json([format_paper_metadata(p) for p in papers], output_file)
    click.echo(f"💾 Saved paper metadata to {output_file}")
    
    # Generate summaries if requested
    if summarize:
        click.echo("\n📝 Generating summaries...")
        
        if model == 'auto':
            summarizer = create_summarizer()
        elif model == 'all':
            summarizer = MultiModelSummarizer()
        else:
            summarizer = create_summarizer(model)
        
        rag_engine = RAGEngine()
        summaries = []
        
        with click.progressbar(papers, label='Summarizing papers') as bar:
            for paper in bar:
                if isinstance(summarizer, MultiModelSummarizer):
                    summary_results = summarizer.summarize_all(paper['abstract'])
                    summary = summary_results.get('openai', summary_results.get('local', 'No summary available'))
                else:
                    summary = summarizer.summarize(paper['abstract'])
                
                summaries.append({
                    'paper_id': paper['paper_id'],
                    'title': paper['title'],
                    'summary': summary,
                    'key_points': summarizer.extract_key_points(paper['abstract']) if hasattr(summarizer, 'extract_key_points') else []
                })
                
                # Add to RAG engine
                rag_engine.add_paper(paper, summary)
        
        # Save summaries
        summary_file = output_file.with_name(output_file.stem + '_summaries.json')
        save_json(summaries, summary_file)
        click.echo(f"\n✅ Summaries saved to {summary_file}")


@cli.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--limit', '-l', default=5, help='Number of results')
def search(query: str, limit: int):
    """Search through indexed papers using semantic search."""
    click.echo(f"🔎 Searching for: '{query}'...")
    
    rag_engine = RAGEngine()
    stats = rag_engine.get_statistics()
    
    if stats['unique_papers'] == 0:
        click.echo("❌ No papers in the database. Run 'fetch' command first.")
        return
    
    results = rag_engine.search(query, n_results=limit)
    
    if not results:
        click.echo("❌ No matching papers found.")
        return
    
    click.echo(f"\n✅ Found {len(results)} relevant papers:\n")
    
    for i, result in enumerate(results, 1):
        click.echo(f"{i}. {result['title']}")
        click.echo(f"   Authors: {', '.join(result['authors'][:3])}{' et al.' if len(result['authors']) > 3 else ''}")
        click.echo(f"   Relevance: {result['relevance_score']:.2%}")
        click.echo(f"   Match: {result['matched_text']}")
        click.echo(f"   ArXiv: {result['arxiv_url']}\n")


@cli.command()
@click.option('--paper-id', '-p', required=True, help='Paper ID to find similar papers')
@click.option('--limit', '-l', default=5, help='Number of results')
def similar(paper_id: str, limit: int):
    """Find papers similar to a given paper."""
    rag_engine = RAGEngine()
    results = rag_engine.find_similar_papers(paper_id, n_results=limit)
    
    if not results:
        click.echo(f"❌ No similar papers found for ID: {paper_id}")
        return
    
    click.echo(f"\n✅ Papers similar to {paper_id}:\n")
    
    for i, result in enumerate(results, 1):
        click.echo(f"{i}. {result['title']}")
        click.echo(f"   Authors: {', '.join(result['authors'][:3])}{' et al.' if len(result['authors']) > 3 else ''}")
        click.echo(f"   Similarity: {result['relevance_score']:.2%}")
        click.echo(f"   ArXiv: {result['arxiv_url']}\n")


@cli.command()
def stats():
    """Show statistics about the paper database."""
    rag_engine = RAGEngine()
    stats = rag_engine.get_statistics()
    
    click.echo("\n📊 Database Statistics:")
    click.echo(f"   Total chunks: {stats.get('total_chunks', 0)}")
    click.echo(f"   Unique papers: {stats.get('unique_papers', 0)}")
    click.echo(f"   Embedding model: {stats.get('embedding_model', 'N/A')}")
    click.echo(f"   Storage location: {stats.get('persist_directory', 'N/A')}\n")


@cli.command()
def topics():
    """Show trending AI/ML research topics."""
    scraper = ArxivScraper()
    topics = scraper.get_trending_topics()
    
    click.echo("\n🔥 Trending AI/ML Topics:")
    for i, topic in enumerate(topics, 1):
        click.echo(f"   {i}. {topic}")
    click.echo()


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear the database?')
def clear():
    """Clear the paper database."""
    rag_engine = RAGEngine()
    if rag_engine.clear_collection():
        click.echo("✅ Database cleared successfully.")
    else:
        click.echo("❌ Failed to clear database.")


if __name__ == "__main__":
    cli()
