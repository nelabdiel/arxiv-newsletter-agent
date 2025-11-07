# src/pipeline.py
"""End-to-end runner: fetch → embed → cluster → (optional editor selects) → summarize → render."""
import argparse
from collections import OrderedDict

from src.config import Settings
from src.fetch import fetch_arxiv
from src.embed import embed
from src.cluster import cluster_embeddings, group_by_label
from src.summarizers import make_llm, topic_label, summarize_science, summarize_comms
from src.render import render_newsletter, save_output

# Editor is optional; guard the import so the pipeline still runs without it.
try:
    from src.editor import choose_clusters as _choose_clusters
except Exception:
    _choose_clusters = None


def run(args=None):
    cfg = Settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=cfg.query)
    parser.add_argument("--since-hours", type=int, default=cfg.since_hours)
    parser.add_argument("--threshold", type=float, default=cfg.cluster_threshold)
    parser.add_argument("--max-items", type=int, default=cfg.max_papers_per_cluster)
    parser.add_argument("--use-mcp", action="store_true",
                        help="Fetch via MCP server instead of python-arxiv client")
    parser.add_argument("--min-cluster-size", type=int, default=2,
                        help="Drop clusters smaller than this size")
    parser.add_argument("--max-clusters", type=int, default=8,
                        help="Keep only the top-K clusters after selection")
    parser.add_argument("--editor", action="store_true",
                        help="Enable LLM editor to pick clusters")
    parser.add_argument("--cohesion-mode", choices=["tight", "diverse"], default="tight",
                        help="Editor preference: 'tight' (high cohesion) or 'diverse' (low cohesion)")
    ns = parser.parse_args(args=args)

    # 1) Fetch
    if ns.use_mcp:
        try:
            from src.mcp_fetch import fetch_via_mcp
            papers = fetch_via_mcp(ns.query, ns.since_hours)
        except Exception as e:
            print(f"[warn] MCP fetch failed ({e}). Falling back to direct arXiv client.")
            papers = fetch_arxiv(query=ns.query, since_hours=ns.since_hours)
    else:
        papers = fetch_arxiv(query=ns.query, since_hours=ns.since_hours)

    if not papers:
        md = render_newsletter([], since_hours=ns.since_hours)
        path = save_output(md)
        print(f"No papers found. Wrote placeholder newsletter → {path}")
        return

    # 2) Embed
    embs = embed(papers, cfg.embed_model)

    # 3) Cluster
    labels = cluster_embeddings(embs, threshold=ns.threshold)
    raw_clusters = group_by_label(papers, labels)

    # NOTE: distance = 1 - cosine_similarity. Larger threshold ⇒ more merging (fewer, larger clusters).
    filtered = {k: v for k, v in raw_clusters.items() if len(v) >= ns.min_cluster_size}

    # If filtering nukes all clusters (common when threshold is strict), fall back to raw (include singletons).
    clusters = filtered if filtered else raw_clusters

    # Sort by size descending
    clusters = OrderedDict(sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True))

    # 3.5) Editor selection (optional); else top-K by size
    if ns.max_clusters and ns.max_clusters > 0:
        if ns.editor and _choose_clusters and len(clusters) > ns.max_clusters:
            llm = make_llm(cfg.ollama_base_url, cfg.ollama_model)
            try:
                chosen_ids = _choose_clusters(
                    llm, clusters, embs, labels, ns.max_clusters, cohesion_mode=ns.cohesion_mode
                )
                clusters = OrderedDict((cid, clusters[cid]) for cid in chosen_ids if cid in clusters)
            except Exception as e:
                print(f"[warn] Editor selection failed ({e}). Falling back to top-{ns.max_clusters} by size.")
                clusters = OrderedDict(list(clusters.items())[:ns.max_clusters])
        else:
            clusters = OrderedDict(list(clusters.items())[:ns.max_clusters])

    # 4) Summaries
    llm = make_llm(cfg.ollama_base_url, cfg.ollama_model)
    rendered = []
    for cid, plist in clusters.items():
        titles = [p["title"] for p in plist]
        topic = topic_label(llm, titles)
        sci, sci_cites = summarize_science(llm, plist, ns.max_items)
        comm, comm_cites = summarize_comms(llm, plist, ns.max_items)
        citations = sci_cites if sci_cites == comm_cites else f"{sci_cites}; {comm_cites}"
        rendered.append({
            "topic": topic,
            "science": sci.strip(),
            "comms": comm.strip(),
            "citations": citations,
        })

    # 5) Render
    md = render_newsletter(rendered, since_hours=ns.since_hours)
    path = save_output(md)
    print(f"Wrote newsletter → {path}")


if __name__ == "__main__":
    run()
