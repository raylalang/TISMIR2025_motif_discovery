#!/usr/bin/env python3
"""
Analyze LR_V0 retrieval/clustering/consolidation to tune parameters.

Inputs per piece:
  - segments JSON (from run_segments.py)
  - embeddings NPZ (from run_embeddings.py)

Outputs to stdout:
  - embedding norm stats
  - similarity stats (overall + top-k)
  - edge stats (counts, pruned by time IoU, same-scale filtering)
  - component counts/sizes
  - consolidation stats (motifs kept, occurrences per motif)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Make package imports available
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery" / "repeated_pattern_discovery"))

from motif_discovery.learned_retrieval.embeddings import _load_segments  # type: ignore
from motif_discovery.learned_retrieval.retrieval import time_iou  # type: ignore
from motif_discovery.learned_retrieval.clustering import connected_components  # type: ignore
from motif_discovery.learned_retrieval.consolidate import ConsolidateConfig, consolidate_cluster  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LR_V0 retrieval graph.")
    parser.add_argument("--segments-dir", type=str, required=True, help="Dir with segment JSON files.")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="Dir with embeddings NPZ files.")
    parser.add_argument(
        "--pieces",
        type=str,
        default=None,
        help="Comma-separated piece ids (stems). Defaults to all JSONs in segments-dir.",
    )
    parser.add_argument("--top-k", type=str, default="5", help="Retrieval top-k (comma-separated for sweep).")
    parser.add_argument("--sim-threshold", type=str, default="0.6", help="Similarity threshold (comma-separated).")
    parser.add_argument("--max-time-iou", type=str, default="0.8", help="Max time IoU (comma-separated).")
    parser.add_argument(
        "--same-scale-only",
        action="store_true",
        help="If set, only connect segments within the same scale_id.",
    )
    parser.add_argument("--cons-iou", type=float, default=0.3, help="Consolidation IoU threshold.")
    parser.add_argument(
        "--cons-score",
        type=str,
        default="centrality",
        choices=["centrality", "note_count"],
        help="Consolidation scoring mode.",
    )
    return parser.parse_args()


def summarize_array(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def build_graph(
    segments: Sequence,
    embeddings: np.ndarray,
    top_k: int,
    sim_threshold: float,
    max_time_iou: float,
    same_scale_only: bool,
) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
    n = embeddings.shape[0]
    sims = embeddings @ embeddings.T
    np.fill_diagonal(sims, -1.0)
    graph: Dict[int, List[int]] = {i: [] for i in range(n)}
    stats = {"edges": 0, "pruned_time": 0, "by_threshold": 0, "by_topk": 0}
    if n == 0:
        return graph, stats
    top_k = max(1, top_k)
    for i in range(n):
        row = sims[i]
        top_idx = np.argpartition(-row, top_k - 1)[:top_k]
        for j in range(n):
            if i == j:
                continue
            if same_scale_only and segments[i].scale_id != segments[j].scale_id:
                continue
            if max_time_iou is not None and max_time_iou >= 0.0:
                if time_iou(segments[i], segments[j]) >= max_time_iou:
                    stats["pruned_time"] += 1
                    continue
            added = False
            if row[j] >= sim_threshold:
                stats["by_threshold"] += 1
                added = True
            elif j in top_idx:
                stats["by_topk"] += 1
                added = True
            if added:
                graph[i].append(j)
                stats["edges"] += 1
    return graph, stats


def main() -> None:
    args = parse_args()
    topk_list = [int(x) for x in args.top_k.split(",") if x.strip()]
    sim_list = [float(x) for x in args.sim_threshold.split(",") if x.strip()]
    iou_list = [float(x) for x in args.max_time_iou.split(",") if x.strip()]
    seg_dir = Path(args.segments_dir)
    emb_dir = Path(args.embeddings_dir)
    if args.pieces:
        pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
    else:
        pieces = [p.stem for p in sorted(seg_dir.glob("*.json"))]
    if not pieces:
        raise ValueError(f"No pieces found in {seg_dir}")

    agg_segments = agg_edges = agg_components = agg_motifs = 0
    agg_sim_stats = {"p25": [], "p50": [], "p75": [], "p90": [], "max": [], "mean": []}
    sweep_stats: Dict[Tuple[int, float, float], Dict[str, float]] = {}
    for piece in pieces:
        seg_path = seg_dir / f"{piece}.json"
        emb_path = emb_dir / f"{piece}.npz"
        if not seg_path.exists() or not emb_path.exists():
            print(f"[skip] missing inputs for {piece}")
            continue
        segments = _load_segments(seg_path)
        data = np.load(emb_path, allow_pickle=True)
        embeddings = data["embeddings"]
        tempos = data["tempos"] if "tempos" in data else np.zeros((embeddings.shape[0],), dtype=np.float32)
        seg_ids = data.get("segment_ids")
        if seg_ids is not None and len(seg_ids) != len(segments):
            print(f"[warn] segment_id count mismatch for {piece}")

        norms = np.linalg.norm(embeddings, axis=1) if embeddings.size else np.array([])
        sims = embeddings @ embeddings.T if embeddings.size else np.zeros((0, 0))
        if sims.size:
            np.fill_diagonal(sims, -1.0)

        print(f"Piece: {piece}")
        print(f"  segments: {len(segments)}")
        if norms.size:
            ns = summarize_array(norms)
            print(f"  embedding norms: min {ns['min']:.4f}, max {ns['max']:.4f}, mean {ns['mean']:.4f}")
        if sims.size:
            svals = sims[sims > -0.5]  # exclude diag
            sstat = summarize_array(svals)
            print(f"  similarity stats: p25 {sstat['p25']:.4f}, p50 {sstat['p50']:.4f}, p75 {sstat['p75']:.4f}, p90 {sstat['p90']:.4f}, max {sstat['max']:.4f}")
            for k in agg_sim_stats:
                agg_sim_stats[k].append(sstat[k])

        for topk in topk_list:
            for sim_th in sim_list:
                for iou_th in iou_list:
                    # Adjusted similarity with tempo penalty
                    sims = embeddings @ embeddings.T
                    np.fill_diagonal(sims, -1.0)
                    g_diff = np.abs(tempos[:, None] - tempos[None, :])
                    penalty = np.exp(-0.0 * g_diff)  # analysis: no tempo penalty here
                    sims = sims * penalty
                    graph, estats = build_graph(
                        segments,
                        embeddings,
                        top_k=topk,
                        sim_threshold=sim_th,
                        max_time_iou=iou_th,
                        same_scale_only=args.same_scale_only,
                    )
                    comps = connected_components(graph)

                    cons_cfg = ConsolidateConfig(iou_threshold=args.cons_iou, score_mode=args.cons_score)
                    motifs = []
                    for comp in comps:
                        kept = consolidate_cluster(comp, segments, sims, cons_cfg)
                        if len(kept) >= 2:
                            motifs.append(kept)

                    comp_sizes = [len(c) for c in comps]
                    key = (topk, sim_th, iou_th)
                    cur = sweep_stats.setdefault(
                        key, {"edges": 0, "components": 0, "motifs": 0, "pieces": 0}
                    )
                    cur["edges"] += estats["edges"]
                    cur["components"] += len(comps)
                    cur["motifs"] += len(motifs)
                    cur["pieces"] += 1

                    # Only print per-piece for first combo to limit verbosity
                    if key == (topk_list[0], sim_list[0], iou_list[0]):
                        if comp_sizes:
                            cstat = summarize_array(np.array(comp_sizes))
                            print(f"  components: {len(comps)} | size p50 {cstat['p50']:.1f}, p90 {cstat['p90']:.1f}, max {cstat['max']}")
                        print(f"  edges ({topk},{sim_th},{iou_th}): {estats['edges']} (thr {estats['by_threshold']}, topk {estats['by_topk']}, pruned_time {estats['pruned_time']})")
                        print(f"  motifs after consolidation: {len(motifs)} (occ median {np.median([len(m) for m in motifs]) if motifs else 0})")

        agg_segments += len(segments)
        # aggregates across first combo only for quick view
        if sims.size:
            first_key = (topk_list[0], sim_list[0], iou_list[0])
            # No direct edge stats kept per-piece for first combo; rely on sweep_stats summary
        # motifs aggregated via sweep_stats below

    print(f"Summary over {len(pieces)} pieces:")
    print(f"  total segments: {agg_segments}")
    if agg_sim_stats["p50"]:
        print("  similarity quantiles (mean over pieces):")
        for k in ("p25", "p50", "p75", "p90", "max", "mean"):
            vals = agg_sim_stats[k]
            print(f"    {k}: {np.mean(vals):.4f}")
    for key, stats in sweep_stats.items():
        topk, sim_th, iou_th = key
        print(f"  combo topk={topk} sim_th={sim_th} max_iou={iou_th}: edges {stats['edges']} comps {stats['components']} motifs {stats['motifs']} (pieces {stats['pieces']})")


if __name__ == "__main__":
    main()
