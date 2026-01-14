"""
End-to-end LR_V0 predictor: segments -> embeddings -> retrieval -> clustering -> consolidation -> occurrences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .segments import Segment, propose_segments
from .embeddings import embed_segments, default_config as default_embed_cfg
from .retrieval import RetrievalConfig, build_graph, build_similarity_matrix
from .clustering import connected_components
from .consolidate import ConsolidateConfig, consolidate_cluster


@dataclass
class LRConfig:
    scale_lengths: Sequence[float] = (2.0, 4.0, 8.0, 16.0)
    hop_ratio: float = 0.25
    min_notes: int = 3
    retrieval: RetrievalConfig = RetrievalConfig()
    consolidate: ConsolidateConfig = ConsolidateConfig()
    use_pitch_class: bool = True  # legacy; unused in current embedding
    same_scale_only: bool = False  # alias to retrieval.same_scale_only
    embedding: str = "handcrafted"  # handcrafted | learned
    learned_ckpt: str | None = None
    learned_device: str = "cpu"
    learned_batch_size: int = 128
    learned_use_duration: bool = False
    pitch_bins: Sequence[float] | None = None
    log_ioi_bins: Sequence[float] | None = None
    ngram_orders: Sequence[int] | None = None
    max_skip: int | None = None
    ngram_bins: int | None = None


def lr_config_from_dict(raw: dict) -> LRConfig:
    cfg = LRConfig()
    if not isinstance(raw, dict):
        return cfg
    if "scale_lengths" in raw:
        cfg.scale_lengths = tuple(raw.get("scale_lengths", cfg.scale_lengths))
    if "hop_ratio" in raw:
        cfg.hop_ratio = float(raw.get("hop_ratio", cfg.hop_ratio))
    if "min_notes" in raw:
        cfg.min_notes = int(raw.get("min_notes", cfg.min_notes))
    if "use_pitch_class" in raw:
        cfg.use_pitch_class = bool(raw.get("use_pitch_class", cfg.use_pitch_class))
    if "embedding" in raw:
        cfg.embedding = str(raw.get("embedding", cfg.embedding))
    if "learned_ckpt" in raw:
        cfg.learned_ckpt = str(raw.get("learned_ckpt", cfg.learned_ckpt))
    if "learned_device" in raw:
        cfg.learned_device = str(raw.get("learned_device", cfg.learned_device))
    if "learned_batch_size" in raw:
        cfg.learned_batch_size = int(raw.get("learned_batch_size", cfg.learned_batch_size))
    if "learned_use_duration" in raw:
        cfg.learned_use_duration = bool(raw.get("learned_use_duration", cfg.learned_use_duration))
    if "pitch_bins" in raw:
        cfg.pitch_bins = tuple(raw.get("pitch_bins", []))
    if "log_ioi_bins" in raw:
        cfg.log_ioi_bins = tuple(raw.get("log_ioi_bins", []))
    if "ngram_orders" in raw:
        cfg.ngram_orders = tuple(raw.get("ngram_orders", []))
    if "max_skip" in raw:
        cfg.max_skip = int(raw.get("max_skip", 0))
    if "ngram_bins" in raw:
        cfg.ngram_bins = int(raw.get("ngram_bins", 0))
    if "retrieval" in raw and isinstance(raw["retrieval"], dict):
        rc = cfg.retrieval
        rraw = raw["retrieval"]
        if "top_k" in rraw:
            rc.top_k = int(rraw.get("top_k", rc.top_k))
        if "sim_threshold" in rraw:
            rc.sim_threshold = float(rraw.get("sim_threshold", rc.sim_threshold))
        if "max_time_iou" in rraw:
            rc.max_time_iou = float(rraw.get("max_time_iou", rc.max_time_iou))
        if "tempo_alpha" in rraw:
            rc.tempo_alpha = float(rraw.get("tempo_alpha", rc.tempo_alpha))
        if "same_scale_only" in rraw:
            rc.same_scale_only = bool(rraw.get("same_scale_only", rc.same_scale_only))
        cfg.retrieval = rc
    if "consolidate" in raw and isinstance(raw["consolidate"], dict):
        cc = cfg.consolidate
        craw = raw["consolidate"]
        if "iou_threshold" in craw:
            cc.iou_threshold = float(craw.get("iou_threshold", cc.iou_threshold))
        if "score_mode" in craw:
            cc.score_mode = str(craw.get("score_mode", cc.score_mode))
        cfg.consolidate = cc
    if "same_scale_only" in raw:
        cfg.same_scale_only = bool(raw.get("same_scale_only", cfg.same_scale_only))
    return cfg


def segments_to_occurrences(
    segments: Sequence[Segment], notes: np.ndarray
) -> List[List[Tuple[float, int]]]:
    occs: List[List[Tuple[float, int]]] = []
    for seg in segments:
        idxs = seg.note_indices
        occ = [(float(notes[i]["onset"]), int(notes[i]["pitch"])) for i in idxs]
        occs.append(occ)
    return occs


def predict_piece(notes: np.ndarray, piece_id: str, cfg: LRConfig):
    # Segments
    segments = propose_segments(
        notes=notes,
        piece_id=piece_id,
        scale_lengths=cfg.scale_lengths,
        hop_ratio=cfg.hop_ratio,
        min_notes=cfg.min_notes,
    )
    # Embeddings
    if cfg.embedding.lower() == "learned":
        if not cfg.learned_ckpt:
            raise ValueError("learned_ckpt is required when embedding=learned.")
        from .learned_embeddings import embed_segments_learned, load_learned_encoder

        model, _enc_cfg = load_learned_encoder(cfg.learned_ckpt, cfg.learned_device)
        embeddings, tempos, segments = embed_segments_learned(
            notes,
            segments,
            model,
            device=cfg.learned_device,
            batch_size=cfg.learned_batch_size,
            use_duration=cfg.learned_use_duration,
        )
    else:
        emb_cfg = default_embed_cfg()
        if cfg.pitch_bins:
            emb_cfg.pitch_bins = cfg.pitch_bins
        if cfg.log_ioi_bins:
            emb_cfg.log_ioi_bins = cfg.log_ioi_bins
        if cfg.ngram_orders:
            emb_cfg.ngram_orders = cfg.ngram_orders
        if cfg.max_skip is not None:
            emb_cfg.max_skip = cfg.max_skip
        if cfg.ngram_bins is not None and cfg.ngram_bins > 0:
            emb_cfg.ngram_bins = cfg.ngram_bins
        embeddings, tempos, segments = embed_segments(notes, segments, emb_cfg)

    # Retrieval graph
    cfg.retrieval.same_scale_only = cfg.same_scale_only or cfg.retrieval.same_scale_only
    graph = build_graph(segments, embeddings, tempos, cfg.retrieval)
    sims = build_similarity_matrix(embeddings, tempos, cfg.retrieval)

    # Clusters
    clusters = connected_components(graph)

    # Consolidation
    motifs: List[List[List[Tuple[float, int]]]] = []
    for cluster in clusters:
        kept_segments = consolidate_cluster(cluster, segments, sims, cfg.consolidate)
        if len(kept_segments) < 2:
            continue  # need at least 2 occurrences to form a motif
        occs = segments_to_occurrences(kept_segments, notes)
        motifs.append(occs)
    return motifs
