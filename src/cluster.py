# src/cluster.py

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


def cluster_embeddings(embeddings: np.ndarray, threshold: float = 0.35):
    """Agglomerative clustering on cosine distance with a merge threshold.
    Smaller threshold ⇒ larger clusters (more merging).
    """
    if len(embeddings) == 0:
        return np.array([])
    dist = 1.0 - cosine_similarity(embeddings)
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, linkage="average", metric="precomputed"
    )
    return model.fit_predict(dist)


def group_by_label(papers, labels):
    clusters = {}
    for p, lab in zip(papers, labels.tolist()):
        clusters.setdefault(int(lab), []).append(p)
    # sort each cluster by title for determinism
    for k in clusters:
        clusters[k] = sorted(clusters[k], key=lambda x: x["title"])
    return clusters
