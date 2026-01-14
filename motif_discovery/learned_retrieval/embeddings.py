"""
Order-aware embeddings for segments (LR_V0.1).

Features:
- Step pairs (Δp, log Δt), tempo-normalized log IOIs with global tempo scalar g.
- Quantized tokens and n-gram/skip-gram bag-of-patterns (hashed) to retain local order.

Outputs L2-normalized vectors for cosine similarity plus tempo scalars.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from .segments import Segment
from motif_discovery.experiments import load_all_notes  # type: ignore


@dataclass
class EmbeddingConfig:
    pitch_bins: Sequence[float]
    log_ioi_bins: Sequence[float]
    ngram_orders: Sequence[int]
    max_skip: int = 0
    ngram_bins: int = 256  # hashed buckets per order
    tempo_eps: float = 1e-4


def _safe_l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm < eps:
        return np.zeros_like(x)
    return x / norm


def _quantize(val: float, bins: Sequence[float]) -> int:
    idx = np.searchsorted(bins, val, side="right") - 1
    return int(np.clip(idx, 0, len(bins) - 2))


def _hash_ngram(order: int, tokens: Sequence[Tuple[int, int]], bucket: int) -> int:
    return hash((order, *tokens)) & 0x7FFFFFFF % bucket


def _extract_steps(notes: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    pitches = notes["pitch"].astype(np.float32)
    onsets = notes["onset"].astype(np.float32)
    if len(pitches) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), 0.0
    dp = np.diff(pitches)
    dt = np.diff(onsets)
    log_dt = np.log(np.clip(dt, 1e-4, None))
    g = float(np.mean(log_dt))
    centered = log_dt - g
    return dp, centered, g


def embed_segment(notes: np.ndarray, segment: Segment, cfg: EmbeddingConfig) -> Tuple[np.ndarray, float]:
    idxs = segment.note_indices
    if len(idxs) < 2:
        return np.zeros(len(cfg.ngram_orders) * cfg.ngram_bins, dtype=np.float32), 0.0

    seg_notes = notes[idxs]
    dp, cu, g = _extract_steps(seg_notes)
    if dp.size == 0:
        return np.zeros(len(cfg.ngram_orders) * cfg.ngram_bins, dtype=np.float32), g

    tokens = [( _quantize(dp[i], cfg.pitch_bins), _quantize(cu[i], cfg.log_ioi_bins)) for i in range(len(dp))]

    vec = np.zeros(len(cfg.ngram_orders) * cfg.ngram_bins, dtype=np.float32)
    offset = 0
    for order in cfg.ngram_orders:
        buckets = cfg.ngram_bins
        ngram_counts = np.zeros(buckets, dtype=np.float32)
        if len(tokens) >= order:
            # contiguous n-grams
            for i in range(len(tokens) - order + 1):
                ngram = tuple(tokens[i : i + order])
                idx = _hash_ngram(order, ngram, buckets)
                ngram_counts[idx] += 1.0
            # simple skip-grams up to max_skip (optional)
            if cfg.max_skip > 0 and order == 2:
                for skip in range(1, cfg.max_skip + 1):
                    for i in range(len(tokens) - 1 - skip):
                        ngram = (tokens[i], tokens[i + 1 + skip])
                        idx = _hash_ngram(order + skip, ngram, buckets)
                        ngram_counts[idx] += 1.0
        vec[offset : offset + buckets] = ngram_counts
        offset += buckets

    vec = _safe_l2_normalize(vec)
    return vec, g


def _embedding_dim(cfg: EmbeddingConfig) -> int:
    return len(cfg.ngram_orders) * cfg.ngram_bins


def embed_segments(
    notes: np.ndarray, segments: Sequence[Segment], cfg: EmbeddingConfig
) -> Tuple[np.ndarray, np.ndarray, List[Segment]]:
    embs = []
    tempos = []
    for seg in segments:
        e, g = embed_segment(notes, seg, cfg)
        embs.append(e)
        tempos.append(g)
    if not embs:
        return np.zeros((0, _embedding_dim(cfg)), dtype=np.float32), np.zeros((0,), dtype=np.float32), list(segments)
    embeddings = np.vstack(embs)
    tempos_arr = np.array(tempos, dtype=np.float32)
    return embeddings, tempos_arr, list(segments)


def _load_segments(path: Path) -> List[Segment]:
    data = json.loads(path.read_text())
    raw_segments = data.get("segments", [])
    segments: List[Segment] = []
    for s in raw_segments:
        segments.append(
            Segment(
                piece_id=s["piece_id"],
                start=float(s["start"]),
                end=float(s["end"]),
                scale_id=int(s["scale_id"]),
                note_indices=list(s["note_indices"]),
                segment_id=s["segment_id"],
            )
        )
    return segments


def default_config() -> EmbeddingConfig:
    # Pitch bins: coarse semitone buckets
    pitch_bins = [-24, -12, -7, -5, -3, -2, -1, 0, 1, 2, 3, 5, 7, 12, 24]
    # Log IOI bins: center around 0 with small steps
    log_bins = list(np.linspace(-2.5, 2.5, 26))  # ~0.2 step
    return EmbeddingConfig(
        pitch_bins=pitch_bins,
        log_ioi_bins=log_bins,
        ngram_orders=(2, 3),
        max_skip=1,
        ngram_bins=256,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute embeddings for segments of a piece.")
    parser.add_argument("--piece", type=str, required=True, help="Piece id like 21-1.")
    parser.add_argument("--csv-note-dir", type=str, required=True, help="Directory with note CSVs.")
    parser.add_argument("--segments-json", type=str, required=True, help="Segments JSON (from run_segments).")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save embeddings as npz (embeddings + segment_ids + tempos).",
    )
    args = parser.parse_args()

    cfg = default_config()

    notes = load_all_notes(str(Path(args.csv_note_dir) / f"{args.piece}.csv"))
    segments = _load_segments(Path(args.segments_json))
    embeddings, tempos, segs = embed_segments(notes, segments, cfg)

    norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.array([])
    print(f"Piece: {args.piece}")
    print(f"Segments: {len(segs)}, Embedding shape: {embeddings.shape}")
    if norms.size:
        print(f"Norms: min {norms.min():.4f}, max {norms.max():.4f}, mean {norms.mean():.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, embeddings=embeddings, tempos=tempos, segment_ids=[s.segment_id for s in segs])
        print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
