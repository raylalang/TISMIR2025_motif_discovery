#!/usr/bin/env python3
"""
Select the piece with highest three-layer F1, then visualize the highest-mean-similarity subgraph.
Prefers per-piece metrics if available; otherwise falls back to mir_eval.
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

from motif_discovery.experiments import load_all_motives, load_all_notes  # type: ignore
from mir_eval.pattern import three_layer_FPR  # type: ignore
from motif_discovery.learned_retrieval.segments import propose_segments  # type: ignore
from motif_discovery.learned_retrieval.embeddings import default_config as default_embed_cfg, embed_segments  # type: ignore
from motif_discovery.learned_retrieval.retrieval import build_graph, build_similarity_matrix  # type: ignore
from motif_discovery.learned_retrieval.clustering import connected_components  # type: ignore
from motif_discovery.learned_retrieval.predict import lr_config_from_dict, LRConfig  # type: ignore


def load_patterns(path: Path):
    data = json.loads(path.read_text())
    patterns = []
    for motif in data.get("patterns", []):
        occs = []
        for occ in motif:
            occs.append([(float(o[0]), int(o[1])) for o in occ])
        patterns.append(occs)
    return patterns


def load_config(path: Path) -> Dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception:
            raise ValueError(f"Config {path} is YAML but PyYAML is not installed.")
        loaded = yaml.safe_load(text)
        return loaded if loaded else {}


def pick_best_piece(pred_dir: Path, label_dir: Path, midi_dir: Path) -> Tuple[str, float]:
    best_piece = None
    best_f1 = -1.0
    for pred_path in sorted(pred_dir.glob("*.json")):
        piece = pred_path.stem
        label_csv = label_dir / f"{piece}.csv"
        midi_path = midi_dir / f"{piece}.mid"
        if not label_csv.exists() or not midi_path.exists():
            continue
        motives = load_all_motives(str(label_csv), str(midi_path))
        patterns_ref = [[list(occ[["onset", "pitch"]]) for occ in motif] for motif in motives.values()]
        patterns_est = load_patterns(pred_path)
        f1, _, _ = three_layer_FPR(patterns_ref, patterns_est)
        if f1 > best_f1:
            best_f1 = f1
            best_piece = piece
    if best_piece is None:
        raise ValueError("No valid pieces found in predictions directory.")
    return best_piece, best_f1


def pick_best_piece_from_metrics(metrics_path: Path, note_dir: Path, pred_dir: Path | None = None) -> Tuple[str, float]:
    payload = json.loads(metrics_path.read_text())
    pieces = payload.get("pieces", payload)
    if not isinstance(pieces, dict):
        raise ValueError(f"Unexpected metrics format in {metrics_path}")

    best_piece = None
    best_f1 = -1.0
    for piece, entry in pieces.items():
        metrics = entry.get("metrics", {})
        three_layer = metrics.get("three_layer", {})
        f1 = three_layer.get("f1", None)
        if f1 is None:
            continue
        if pred_dir is not None:
            pred_path = pred_dir / f"{piece}.json"
            if not pred_path.exists():
                continue
        note_path = note_dir / f"{piece}.csv"
        if not note_path.exists():
            continue
        if float(f1) > best_f1:
            best_f1 = float(f1)
            best_piece = piece
    if best_piece is None:
        raise ValueError("No valid pieces found in metrics_by_piece.json.")
    return best_piece, best_f1


def mean_similarity(sub_idxs: List[int], sims: np.ndarray) -> float:
    if len(sub_idxs) < 2:
        return 0.0
    sub = sims[np.ix_(sub_idxs, sub_idxs)]
    mask = ~np.eye(len(sub_idxs), dtype=bool)
    vals = sub[mask]
    return float(np.mean(vals)) if vals.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize LR_V0 retrieval subgraph.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory with config_resolved and motifs.")
    parser.add_argument("--output", type=str, required=True, help="Output image path (png).")
    parser.add_argument("--max-nodes", type=int, default=120, help="Max nodes to show in the subgraph.")
    parser.add_argument("--min-nodes", type=int, default=4, help="Minimum nodes to display across subgraphs.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config_resolved.yaml"
    cfg = load_config(cfg_path)

    motif_cfg = cfg.get("motif", {})
    label_dir = Path(motif_cfg.get("csv_label_dir", ""))
    midi_dir = Path(motif_cfg.get("motif_midi_dir", ""))
    note_dir = Path(motif_cfg.get("csv_note_dir", ""))
    pred_dir = Path(motif_cfg.get("save_predictions_dir", "")) if motif_cfg.get("save_predictions_dir") else None
    if pred_dir is None or not pred_dir.exists():
        # fallback: detect motifs_* folder in run_dir
        candidates = sorted(run_dir.glob("motifs*"))
        if not candidates:
            raise ValueError("No predictions directory found in run_dir.")
        pred_dir = candidates[0]

    metrics_path = pred_dir / "metrics_by_piece.json"
    if metrics_path.exists():
        best_piece, best_f1 = pick_best_piece_from_metrics(metrics_path, note_dir, pred_dir)
        best_source = "metrics_by_piece"
    else:
        best_piece, best_f1 = pick_best_piece(pred_dir, label_dir, midi_dir)
        best_source = "mir_eval"

    # Load notes
    notes = load_all_notes(str(note_dir / f"{best_piece}.csv"))

    # Build LR config from inline params
    lr_params = motif_cfg.get("lr_params", {})
    lr_cfg = lr_config_from_dict(lr_params) if lr_params else LRConfig()

    # Segments + embeddings
    segments = propose_segments(
        notes=notes,
        piece_id=best_piece,
        scale_lengths=lr_cfg.scale_lengths,
        hop_ratio=lr_cfg.hop_ratio,
        min_notes=lr_cfg.min_notes,
    )
    emb_cfg = default_embed_cfg()
    if lr_cfg.pitch_bins:
        emb_cfg.pitch_bins = lr_cfg.pitch_bins
    if lr_cfg.log_ioi_bins:
        emb_cfg.log_ioi_bins = lr_cfg.log_ioi_bins
    if lr_cfg.ngram_orders:
        emb_cfg.ngram_orders = lr_cfg.ngram_orders
    if lr_cfg.max_skip is not None:
        emb_cfg.max_skip = lr_cfg.max_skip
    if lr_cfg.ngram_bins is not None and lr_cfg.ngram_bins > 0:
        emb_cfg.ngram_bins = lr_cfg.ngram_bins
    embeddings, tempos, _ = embed_segments(notes, segments, emb_cfg)

    # Retrieval graph + similarities
    lr_cfg.retrieval.same_scale_only = lr_cfg.same_scale_only or lr_cfg.retrieval.same_scale_only
    graph = build_graph(segments, embeddings, tempos, lr_cfg.retrieval)
    sims = build_similarity_matrix(embeddings, tempos, lr_cfg.retrieval)

    # Choose component with highest mean similarity (must satisfy min size)
    comps = connected_components(graph)
    min_nodes = max(1, args.min_nodes)
    comp_scores = [(c, mean_similarity(c, sims)) for c in comps if len(c) >= min_nodes]
    if not comp_scores:
        raise ValueError("No components found with the minimum node count.")
    comp_scores.sort(key=lambda x: x[1], reverse=True)
    best_comp, best_comp_mean = comp_scores[0]

    # Include adjacent nodes (1-hop neighbors of the best component)
    cluster_set = set(best_comp)
    neighbor_set = set()
    for i in best_comp:
        for j in graph.get(i, []):
            if j not in cluster_set:
                neighbor_set.add(j)

    keep_neighbors = list(neighbor_set)
    max_nodes = args.max_nodes if args.max_nodes and args.max_nodes > 0 else None
    if max_nodes is not None:
        allow_neighbors = max_nodes - len(best_comp)
        if allow_neighbors < 0:
            allow_neighbors = 0
        if len(keep_neighbors) > allow_neighbors:
            # Rank adjacent nodes by max similarity to any cluster node, keep top-K.
            neighbor_scores = []
            cluster_list = list(cluster_set)
            for j in keep_neighbors:
                sims_to_cluster = sims[j, cluster_list]
                neighbor_scores.append((j, float(np.max(sims_to_cluster))))
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            keep_neighbors = [j for j, _score in neighbor_scores[:allow_neighbors]]

    sub_idxs = list(best_comp) + keep_neighbors
    sub_set = set(sub_idxs)

    # Build subgraph edges (weighted by similarity)
    edges = []
    seen = set()
    for i in sub_idxs:
        for j in graph.get(i, []):
            if j in sub_set:
                pair = (min(i, j), max(i, j))
                if pair in seen:
                    continue
                seen.add(pair)
                edges.append((pair[0], pair[1], float(sims[pair[0], pair[1]])))

    # Try networkx for visualization; fall back to adjacency heatmap
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import networkx as nx  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        G = nx.Graph()
        for i in sub_idxs:
            G.add_node(i)
        for i, j, w in edges:
            G.add_edge(i, j, weight=w)
        pos = nx.spring_layout(G, seed=7)
        widths = []
        edge_colors = []
        for i, j, w in edges:
            if i in cluster_set and j in cluster_set:
                scale = 1.0
            elif i in cluster_set or j in cluster_set:
                scale = 0.5
            else:
                scale = 0.3
            widths.append(0.5 + 3.0 * w * scale)
            alpha = min(1.0, 0.15 + 0.85 * w * scale)
            edge_colors.append((0.75, 0.75, 0.75, alpha))
        node_colors = ["#c7dbf7" if i in cluster_set else "#f7c9c4" for i in sub_idxs]
        nx.draw_networkx_nodes(G, pos, node_size=80, node_color=node_colors)
        nx.draw_networkx_edges(G, pos, width=widths, edge_color=edge_colors)
        plt.title(f"{best_piece} (F1={best_f1:.3f}) | comp mean sim={best_comp_mean:.3f}")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
    except Exception:
        import matplotlib.pyplot as plt  # type: ignore

        idx_map = {idx: i for i, idx in enumerate(sub_idxs)}
        adj = np.zeros((len(sub_idxs), len(sub_idxs)), dtype=np.float32)
        for i, j, w in edges:
            if i in cluster_set and j in cluster_set:
                scale = 1.0
            elif i in cluster_set or j in cluster_set:
                scale = 0.5
            else:
                scale = 0.3
            adj[idx_map[i], idx_map[j]] = w * scale
            adj[idx_map[j], idx_map[i]] = w * scale
        plt.imshow(adj, cmap="Greys", interpolation="nearest")
        plt.title(f"{best_piece} (F1={best_f1:.3f}) | adj weighted")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    meta = {
        "piece": best_piece,
        "three_layer_f1": best_f1,
        "three_layer_source": best_source,
        "component_size": len(best_comp),
        "component_mean_similarity": best_comp_mean,
        "subgraph_nodes": len(sub_idxs),
        "cluster_nodes": len(best_comp),
        "adjacent_nodes": len(keep_neighbors),
        "adjacent_nodes_total": len(neighbor_set),
        "min_nodes": min_nodes,
        "output": str(out_path),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
