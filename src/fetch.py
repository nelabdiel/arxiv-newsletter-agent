# src/fetch.py

from datetime import datetime, timedelta, timezone
import arxiv


# Some arXiv API pages occasionally return an empty feed when paginating to
# start>=100. The official client raises UnexpectedEmptyPageError in that case.
# For our use (rolling last 24h), we treat that as "end of results" instead of
# failing the whole run. Also cap max_results<=100 to avoid a second page.


def fetch_arxiv(query: str, since_hours: int = 24, max_results: int = 20):
    """Fetch arXiv entries updated in the last N hours (rolling window).
    
    
    We keep only: id, title, authors, summary, categories, links, updated.
    """
    # Guardrail: the python-arxiv client paginates in chunks of page_size (default 100).
    # We explicitly set page_size=100 and cap max_results to avoid page 2 unless you
    # raise max_results consciously.
    page_size = 100
    max_results = min(max_results, page_size)
    
    
    search = arxiv.Search(
    query=query,
    max_results=max_results,
    sort_by=arxiv.SortCriterion.LastUpdatedDate,
    )
    
    
    client = arxiv.Client(page_size=page_size, delay_seconds=0.0, num_retries=2)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    
    
    papers = []
    try:
        for r in client.results(search):
            updated = r.updated or r.published
            if not updated:
                continue
            if updated < cutoff:
                # Remaining results will only get older; stop early
                break
            papers.append(
                {
                "id": r.get_short_id(),
                "title": (r.title or "").strip().replace("", " "),
                "authors": [a.name for a in r.authors],
                "summary": (r.summary or "").strip(),
                "categories": r.categories,
                "links": {"abs": r.entry_id, "pdf": r.pdf_url},
                "updated": updated.isoformat(),
                }
                )
    except arxiv.UnexpectedEmptyPageError:
        # Treat as end-of-results; keep what we already collected
        pass
    
    
    return papers
