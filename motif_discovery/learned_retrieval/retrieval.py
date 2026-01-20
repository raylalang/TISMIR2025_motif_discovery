"""
Within-piece retrieval for LR_V0.

Build a similarity graph:
- cosine similarity on normalized embeddings
- optional tempo penalty on similarity
- edges added if sim >= sim_threshold or in top-k
- optional time IoU suppression to avoid trivial overlaps
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .segments import Segment


@dataclass
class RetrievalConfig:
    top_k: int = 5
    sim_threshold: float = 0.6
    max_time_iou: float = 0.8  # skip edges if time IoU >= this
    tempo_alpha: float = 0.5   # penalty factor for |g_i - g_j|
    same_scale_only: bool = False


def time_iou(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def build_similarity_matrix(embeddings: np.ndarray, tempos: np.ndarray, cfg: RetrievalConfig) -> np.ndarray:
    if embeddings.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    sims = embeddings @ embeddings.T
    if tempos.size and cfg.tempo_alpha > 0:
        g_diff = np.abs(tempos[:, None] - tempos[None, :])
        penalty = np.exp(-cfg.tempo_alpha * g_diff)
        sims = sims * penalty
    np.fill_diagonal(sims, -1.0)  # exclude self
    return sims.astype(np.float32)


def build_graph(
    segments: Sequence[Segment],
    embeddings: np.ndarray,
    tempos: np.ndarray,
    cfg: RetrievalConfig,
) -> Dict[int, List[int]]:
    sims = build_similarity_matrix(embeddings, tempos, cfg)
    n = sims.shape[0]
    graph: Dict[int, List[int]] = {i: [] for i in range(n)}
    if n == 0:
        return graph

    top_k = max(1, cfg.top_k)
    for i in range(n):
        row = sims[i]
        # top-k indices (excluding self already set to -1)
        top_idx = np.argpartition(-row, top_k - 1)[:top_k]
        for j in range(n):
            if i == j:
                continue
            if cfg.same_scale_only and segments[i].scale_id != segments[j].scale_id:
                continue
            if cfg.max_time_iou is not None and cfg.max_time_iou >= 0.0:
                if time_iou(segments[i], segments[j]) >= cfg.max_time_iou:
                    continue
            if row[j] >= cfg.sim_threshold or j in top_idx:
                graph[i].append(j)
    return graph
