# src/editor.py
import json, math, re
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from langchain_ollama import ChatOllama
from langchain.schema import SystemMessage, HumanMessage


def _iso_to_dt(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def cluster_features(
    clusters: Dict[int, List[dict]],
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, dict]:
    feats = {}
    max_size = max((len(v) for v in clusters.values()), default=1)
    label_to_idxs = {}
    for idx, lab in enumerate(labels.tolist()):
        label_to_idxs.setdefault(int(lab), []).append(idx)

    for cid, plist in clusters.items():
        idxs = label_to_idxs.get(int(cid), [])
        if not idxs:
            size, cohesion = len(plist), 0.0
        else:
            em = embeddings[idxs]
            centroid = em.mean(axis=0, keepdims=True)
            sims = cosine_similarity(em, centroid).ravel()
            cohesion = float(np.clip(np.mean(sims), 0.0, 1.0))
            size = len(idxs)

        hours, cats = [], []
        for p in plist:
            dt = _iso_to_dt(p.get("updated") or p.get("published") or "")
            dh = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
            hours.append(dh)
            for c in p.get("categories", []):
                cats.append(c.split(".")[0] if "." in c else c)
        mean_hours = float(np.mean(hours) if hours else 24.0)
        recency = math.exp(-mean_hours / 48.0)
        top_category = (Counter(cats).most_common(1)[0][0] if cats else "unknown")

        feats[int(cid)] = {
            "size": size,
            "size_norm": size / max_size if max_size else 0.0,
            "cohesion": cohesion,
            "recency": recency,
            "top_category": top_category,
        }
    return feats


def heuristic_rank(feats: Dict[int, dict], k: int) -> List[int]:
    scored = []
    for cid, f in feats.items():
        s = 0.5 * f["size_norm"] + 0.3 * f["cohesion"] + 0.2 * f["recency"]
        scored.append((cid, s))
    scored.sort(key=lambda t: t[1], reverse=True)
    return [cid for cid, _ in scored[:k]]


def _extract_json(s: str) -> dict:
    try:
        lo = s.index("{")
        hi = s.rindex("}") + 1
        return json.loads(s[lo:hi])
    except Exception:
        return {}


def llm_select(
    llm: ChatOllama,
    feats: Dict[int, dict],
    clusters: Dict[int, List[dict]],
    k: int,
) -> List[int]:
    lines = []
    for cid, plist in clusters.items():
        titles = "; ".join(p["title"] for p in plist[:3])
        f = feats[cid]
        lines.append(
            f"{cid}|size={f['size']}|cohesion={f['cohesion']:.2f}|recency={f['recency']:.2f}|cat={f['top_category']}|titles={titles}"
        )
    evidence = "\n".join(lines)

    prompt = (
        "You are the editor of a daily quantum arXiv brief. "
        f"Select up to {k} clusters to include. Prefer larger size, higher cohesion, higher recency, and diversify categories. "
        "Return STRICT JSON only in this schema: "
        '{"selected":[{"id":<int>,"reason":"short phrase (<=20 words)"}]} '
        "No other text. Evidence lines use 'cid|feature=...|...|titles=...'.\n\n"
        "Evidence:\n" + evidence
    )
    out = llm.invoke([
        SystemMessage(content="Be terse and obey the schema."),
        HumanMessage(content=prompt),
    ]).content
    data = _extract_json(out)
    sel = []
    try:
        for item in data.get("selected", []):
            cid = int(item["id"])
            if cid in clusters and cid not in sel:
                sel.append(cid)
            if len(sel) >= k:
                break
    except Exception:
        sel = []
    return sel


def choose_clusters(
    llm: ChatOllama,
    clusters: Dict[int, List[dict]],
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
) -> List[int]:
    feats = cluster_features(clusters, embeddings, labels)
    chosen = llm_select(llm, feats, clusters, k)
    if chosen:
        return chosen
    return heuristic_rank(feats, k)
