#!/usr/bin/env python3
"""
Compute similarity distributions for LR_V0 embedding:
- same motif vs different motif cosine similarities
- optional time IoU stats for occurrences
Outputs aggregate stats and saves JSON if requested.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Make motif_discovery imports available
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery" / "repeated_pattern_discovery"))

from motif_discovery.experiments import load_all_motives  # type: ignore
from motif_discovery.learned_retrieval.embeddings import default_config, embed_segment  # type: ignore


def time_iou_from_spans(span_a: Tuple[float, float], span_b: Tuple[float, float]) -> float:
    a0, a1 = span_a
    b0, b1 = span_b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def percentile_stats(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute same-vs-different motif similarity distributions (LR_V0 embedding)."
    )
    parser.add_argument("--csv-label-dir", type=str, required=True, help="Label CSV dir (BPS).")
    parser.add_argument("--motif-midi-dir", type=str, required=True, help="Motif MIDI dir (BPS).")
    parser.add_argument("--pieces", type=str, default=None, help="Comma-separated piece ids (e.g., 21-1,29-1).")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save stats JSON.")
    args = parser.parse_args()

    label_dir = Path(args.csv_label_dir)
    midi_dir = Path(args.motif_midi_dir)
    if args.pieces:
        pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
    else:
        pieces = [p.stem for p in sorted(label_dir.glob("*.csv"))]

    cfg = default_config()

    same_sims: List[float] = []
    diff_sims: List[float] = []
    same_iou: List[float] = []
    diff_iou: List[float] = []

    for piece in pieces:
        lbl = label_dir / f"{piece}.csv"
        midi = midi_dir / f"{piece}.mid"
        motives = load_all_motives(str(lbl), str(midi))

        # Precompute embeddings for each occurrence
        occ_embeddings: List[Tuple[str, np.ndarray]] = []
        occ_spans: List[Tuple[str, float, float]] = []
        for m_id, occs in motives.items():
            for occ in occs:
                # Treat occurrence array as notes; build a minimal stub for segment
                seg = type("dummy", (), {})()
                seg.note_indices = list(range(occ.size))
                seg.piece_id = piece
                seg.scale_id = 0
                emb, _ = embed_segment(occ, seg, cfg)
                occ_embeddings.append((m_id, emb))
                occ_spans.append((m_id, float(np.min(occ["onset"])), float(np.max(occ["end"]))))

        # Pairwise sims
        for i in range(len(occ_embeddings)):
            mi, ei = occ_embeddings[i]
            for j in range(i + 1, len(occ_embeddings)):
                mj, ej = occ_embeddings[j]
                sim = float(np.dot(ei, ej))
                if mi == mj:
                    same_sims.append(sim)
                else:
                    diff_sims.append(sim)

        # IoU stats
        for i in range(len(occ_spans)):
            mi, s0, s1 = occ_spans[i]
            for j in range(i + 1, len(occ_spans)):
                mj, t0, t1 = occ_spans[j]
                iou = time_iou_from_spans((s0, s1), (t0, t1))
                if mi == mj:
                    same_iou.append(iou)
                else:
                    diff_iou.append(iou)

    def summarize(name: str, arr_list: List[float]):
        if not arr_list:
            print(f"{name}: no data")
            return {}
        arr = np.array(arr_list)
        stats = percentile_stats(arr)
        print(f"{name}: p25 {stats['p25']:.4f}, p50 {stats['p50']:.4f}, p75 {stats['p75']:.4f}, p90 {stats['p90']:.4f}, max {stats['max']:.4f}")
        return stats

    stats_out = {
        "pieces": pieces,
        "same_sims": summarize("same motif sims", same_sims),
        "diff_sims": summarize("diff motif sims", diff_sims),
        "same_iou": summarize("same motif IoU", same_iou),
        "diff_iou": summarize("diff motif IoU", diff_iou),
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats_out, indent=2))
        print(f"Saved stats to {out_path}")


if __name__ == "__main__":
    main()
