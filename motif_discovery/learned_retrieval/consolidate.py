"""
NMS-style consolidation of segments into occurrences within each cluster.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .segments import Segment
from .retrieval import time_iou


def overlap_ratio(a: Segment, b: Segment) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    denom = min(a.end - a.start, b.end - b.start)
    return inter / denom if denom > 0 else 0.0


@dataclass
class ConsolidateConfig:
    iou_threshold: float = 0.3
    overlap_threshold: float | None = None
    score_mode: str = "centrality"  # or "note_count"


def consolidate_cluster(
    cluster: Sequence[int],
    segments: Sequence[Segment],
    sims: np.ndarray,
    cfg: ConsolidateConfig,
) -> List[Segment]:
    if not cluster:
        return []
    # Scores
    scores = []
    for idx in cluster:
        if cfg.score_mode == "centrality" and sims.size:
            # mean similarity to others in cluster (excluding self)
            mask = [j for j in cluster if j != idx]
            if mask:
                scores.append((idx, float(np.mean(sims[idx, mask]))))
            else:
                scores.append((idx, 0.0))
        else:  # note_count or fallback
            scores.append((idx, len(segments[idx].note_indices)))

    # Sort descending by score
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    kept: List[int] = []
    for idx, _ in scores:
        keep = True
        for k in kept:
            iou = time_iou(segments[idx], segments[k])
            contain = overlap_ratio(segments[idx], segments[k])
            if iou >= cfg.iou_threshold:
                keep = False
                break
            if cfg.overlap_threshold is not None and contain >= cfg.overlap_threshold:
                keep = False
                break
        if keep:
            kept.append(idx)
    return [segments[i] for i in kept]
