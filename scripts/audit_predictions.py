#!/usr/bin/env python3
"""
Audit motif discovery predictions for redundancy and explosion.

Inputs:
  --patterns_dir : directory containing per-piece motif JSON from extract_motif_predictions.py
Outputs (under --output_dir):
  - audit_table.csv : per-piece statistics
  - audit_summary.json : aggregate stats + notable cases
  - failure_cases.json : representative redundant/sparse/fragmentation examples
  - plots/* : histograms for spans, note counts, overlaps
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit motif predictions for redundancy/explosion."
    )
    parser.add_argument(
        "--patterns_dir",
        type=str,
        required=True,
        help="Directory with per-piece motif JSON (from extract_motif_predictions.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save audit outputs.",
    )
    parser.add_argument(
        "--pieces",
        type=str,
        default=None,
        help="Optional comma-separated piece ids to audit (e.g., 01-1,02-1). Defaults to all.",
    )
    parser.add_argument(
        "--near_dup_threshold",
        type=float,
        default=0.8,
        help="Jaccard/IoU threshold to flag near-duplicate motifs.",
    )
    parser.add_argument(
        "--subset_threshold",
        type=float,
        default=0.9,
        help="Coverage threshold to flag subset/superset motifs.",
    )
    parser.add_argument(
        "--sparse_ratio",
        type=float,
        default=4.0,
        help="Span/notes ratio to flag long-span sparse motifs.",
    )
    parser.add_argument(
        "--top_k_failures",
        type=int,
        default=10,
        help="Number of examples to keep in failure case pack.",
    )
    parser.add_argument(
        "--max_motifs_per_piece",
        type=int,
        default=None,
        help="Optional cap on motifs per piece to limit pairwise explosion. Keeps motifs with most occurrences first.",
    )
    parser.add_argument(
        "--pairwise_cap",
        type=int,
        default=300,
        help="Skip pairwise overlaps if a piece has more motifs than this cap.",
    )
    parser.add_argument(
        "--piece_pattern",
        type=str,
        default="*.json",
        help="Glob pattern to select pieces (e.g., '0?-1.json' to shard).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of processes to audit pieces in parallel.",
    )
    return parser.parse_args()


def load_patterns(patterns_dir: Path, keep_pieces=None, max_motifs=None, piece_pattern="*.json"):
    pieces = {}
    for path in sorted(patterns_dir.glob(piece_pattern)):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or "patterns" not in data:
            continue
        patterns = data.get("patterns", [])
        motifs = []
        for motif in patterns:
            occs = []
            for occ in motif:
                occs.append([(float(o[0]), int(o[1])) for o in occ])
            motifs.append(occs)
        piece_id = path.stem
        if keep_pieces and piece_id not in keep_pieces:
            continue
        if max_motifs and len(motifs) > max_motifs:
            motifs = sorted(motifs, key=lambda m: len(m), reverse=True)[:max_motifs]
        pieces[piece_id] = motifs
    return pieces


def occurrence_span(occ: List[Tuple[float, int]]) -> float:
    if not occ:
        return 0.0
    starts = [o[0] for o in occ]
    return max(starts) - min(starts) if len(starts) > 1 else 0.0


def note_jaccard(a: List[Tuple[float, int]], b: List[Tuple[float, int]]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union)


def time_iou(a: List[Tuple[float, int]], b: List[Tuple[float, int]]) -> float:
    if not a or not b:
        return 0.0
    a_start = min(x[0] for x in a)
    a_end = max(x[0] for x in a)
    b_start = min(x[0] for x in b)
    b_end = max(x[0] for x in b)
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def best_motif_overlap(motif_a, motif_b):
    best_note = 0.0
    best_time = 0.0
    best_pair = (None, None)
    for occ_a in motif_a:
        for occ_b in motif_b:
            n_j = note_jaccard(occ_a, occ_b)
            t_i = time_iou(occ_a, occ_b)
            if n_j > best_note or (math.isclose(n_j, best_note) and t_i > best_time):
                best_note = n_j
                best_time = t_i
                best_pair = (occ_a, occ_b)
    return best_note, best_time, best_pair


def compute_piece_stats(
    piece_id: str,
    motifs: List[List[List[Tuple[float, int]]]],
    thresholds: Dict[str, float],
    pairwise_cap: int,
):
    num_motifs = len(motifs)
    total_occurrences = sum(len(m) for m in motifs)

    motif_span_stats = []
    motif_note_stats = []

    for motif in motifs:
        spans = [occurrence_span(occ) for occ in motif]
        notes = [len(occ) for occ in motif]
        if spans:
            motif_span_stats.append(
                {
                    "min": min(spans),
                    "mean": statistics.mean(spans),
                    "max": max(spans),
                }
            )
        if notes:
            motif_note_stats.append(
                {
                    "min": min(notes),
                    "mean": statistics.mean(notes),
                    "max": max(notes),
                }
            )

    # Overlaps across motifs
    pair_overlaps = []
    sample = list(enumerate(motifs))
    if pairwise_cap and num_motifs > pairwise_cap:
        sample = sorted(sample, key=lambda m: len(m[1]), reverse=True)[:pairwise_cap]
    if pairwise_cap == 0:
        sample = []

    for idx, (i, motif_i) in enumerate(sample):
        for _, (j, motif_j) in enumerate(sample[idx + 1 :], start=idx + 1):
            best_note, best_time, _ = best_motif_overlap(motif_i, motif_j)
            pair_overlaps.append((i, j, best_note, best_time))

    near_dups = [
        p
        for p in pair_overlaps
        if p[2] >= thresholds["near_dup"] or p[3] >= thresholds["near_dup"]
    ]
    subsets = [
        p
        for p in pair_overlaps
        if (p[2] >= thresholds["subset"]) or (p[3] >= thresholds["subset"])
    ]

    # Long-span sparse motifs
    sparse = []
    for idx, motif in enumerate(motifs):
        spans = [occurrence_span(occ) for occ in motif if len(occ)]
        notes = [len(occ) for occ in motif if len(occ)]
        if not spans or not notes:
            continue
        ratio = max(spans) / max(statistics.mean(notes), 1e-6)
        if ratio >= thresholds["sparse"]:
            sparse.append({"motif": idx, "ratio": ratio})

    piece_summary = {
        "piece": piece_id,
        "num_motifs": num_motifs,
        "total_occurrences": total_occurrences,
        "motif_span_min": min((m["min"] for m in motif_span_stats), default=0.0),
        "motif_span_mean": (
            statistics.mean((m["mean"] for m in motif_span_stats))
            if motif_span_stats
            else 0.0
        ),
        "motif_span_max": max((m["max"] for m in motif_span_stats), default=0.0),
        "motif_notes_min": min((m["min"] for m in motif_note_stats), default=0.0),
        "motif_notes_mean": (
            statistics.mean((m["mean"] for m in motif_note_stats))
            if motif_note_stats
            else 0.0
        ),
        "motif_notes_max": max((m["max"] for m in motif_note_stats), default=0.0),
        "mean_note_jaccard": (
            statistics.mean((p[2] for p in pair_overlaps)) if pair_overlaps else 0.0
        ),
        "mean_time_iou": (
            statistics.mean((p[3] for p in pair_overlaps)) if pair_overlaps else 0.0
        ),
        "near_duplicate_pairs": len(near_dups),
        "subset_pairs": len(subsets),
        "sparse_motifs": len(sparse),
    }

    return piece_summary, pair_overlaps, sparse


def _compute_piece_stats_task(args):
    """Helper for process pool (must be top-level for pickling)."""
    piece_id, motifs, thresholds, pairwise_cap = args
    return piece_id, compute_piece_stats(piece_id, motifs, thresholds, pairwise_cap)


def write_csv(rows: List[Dict], path: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_hist(data, title, xlabel, out_path):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=30, color="#4c72b0", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    patterns_dir = Path(args.patterns_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    thresholds = {
        "near_dup": args.near_dup_threshold,
        "subset": args.subset_threshold,
        "sparse": args.sparse_ratio,
    }

    keep_pieces = args.pieces.split(",") if args.pieces else None
    pieces = load_patterns(
        patterns_dir,
        keep_pieces=keep_pieces,
        max_motifs=args.max_motifs_per_piece,
        piece_pattern=args.piece_pattern,
    )
    per_piece_rows = []
    all_pair_overlaps = []
    sparse_cases = []

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(
                    _compute_piece_stats_task, (piece_id, motifs, thresholds, args.pairwise_cap)
                ): piece_id
                for piece_id, motifs in pieces.items()
            }
            for fut in as_completed(futures):
                piece_id_ret, (summary, overlaps, sparse) = fut.result()
                per_piece_rows.append(summary)
                all_pair_overlaps.extend([(piece_id_ret, *o) for o in overlaps])
                sparse_cases.extend([(piece_id_ret, s["motif"], s["ratio"]) for s in sparse])
    else:
        for piece_id, motifs in pieces.items():
            summary, overlaps, sparse = compute_piece_stats(
                piece_id, motifs, thresholds, args.pairwise_cap
            )
            per_piece_rows.append(summary)
            all_pair_overlaps.extend([(piece_id, *o) for o in overlaps])
            sparse_cases.extend([(piece_id, s["motif"], s["ratio"]) for s in sparse])

    write_csv(per_piece_rows, output_dir / "audit_table.csv")

    # Aggregates
    agg = {
        "num_pieces": len(per_piece_rows),
        "avg_num_motifs": (
            statistics.mean((r["num_motifs"] for r in per_piece_rows))
            if per_piece_rows
            else 0.0
        ),
        "avg_total_occurrences": (
            statistics.mean((r["total_occurrences"] for r in per_piece_rows))
            if per_piece_rows
            else 0.0
        ),
        "avg_mean_note_jaccard": (
            statistics.mean((r["mean_note_jaccard"] for r in per_piece_rows))
            if per_piece_rows
            else 0.0
        ),
        "avg_mean_time_iou": (
            statistics.mean((r["mean_time_iou"] for r in per_piece_rows))
            if per_piece_rows
            else 0.0
        ),
        "total_near_duplicate_pairs": sum(
            r["near_duplicate_pairs"] for r in per_piece_rows
        ),
        "total_subset_pairs": sum(r["subset_pairs"] for r in per_piece_rows),
        "total_sparse_motifs": sum(r["sparse_motifs"] for r in per_piece_rows),
    }

    top_near_dups = sorted(all_pair_overlaps, key=lambda x: (x[2], x[3]), reverse=True)[
        : args.top_k_failures
    ]
    top_sparse = sorted(sparse_cases, key=lambda x: x[2], reverse=True)[
        : args.top_k_failures
    ]

    summary = {
        "aggregate": agg,
        "top_near_duplicates": [
            {
                "piece": p,
                "motif_a": i,
                "motif_b": j,
                "note_jaccard": nj,
                "time_iou": ti,
            }
            for p, i, j, nj, ti in top_near_dups
        ],
        "top_sparse": [
            {"piece": p, "motif": m, "span_per_note": ratio}
            for p, m, ratio in top_sparse
        ],
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))

    failure_pack = {
        "redundant_clusters": summary["top_near_duplicates"],
        "sparse_motifs": summary["top_sparse"],
    }
    (output_dir / "failure_cases.json").write_text(json.dumps(failure_pack, indent=2))

    # Plots
    note_counts = []
    spans = []
    note_j = []
    time_i = []
    for _, i, j, nj, ti in all_pair_overlaps:
        note_j.append(nj)
        time_i.append(ti)
    for motifs in pieces.values():
        for motif in motifs:
            for occ in motif:
                note_counts.append(len(occ))
                spans.append(occurrence_span(occ))

    if note_counts:
        plot_hist(
            note_counts,
            "Occurrence note counts",
            "notes per occurrence",
            plots_dir / "notes.png",
        )
    if spans:
        plot_hist(
            spans, "Occurrence spans", "time span (beats)", plots_dir / "spans.png"
        )
    if note_j:
        plot_hist(
            note_j,
            "Motif pair note Jaccard",
            "note-level Jaccard",
            plots_dir / "note_jaccard.png",
        )
    if time_i:
        plot_hist(time_i, "Motif pair time IoU", "time IoU", plots_dir / "time_iou.png")

    print(f"Audit complete. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
