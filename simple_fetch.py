import arxiv
import json
from datetime import datetime

# Fetch 5 papers about transformers
papers = []
search = arxiv.Search(query="transformer", max_results=5)

print("Fetching papers from ArXiv...")
for result in search.results():
    paper = {
        'title': result.title,
        'authors': [a.name for a in result.authors],
        'date': str(result.published),
        'url': result.entry_id
    }
    papers.append(paper)
    print(f" {paper['title']}")

# Save to file
with open('data/my_papers.json', 'w') as f:
    json.dump(papers, f, indent=2)

print(f"\n✅ Saved {len(papers)} papers to data/my_papers.json")
