#!/usr/bin/env python3
"""
Run a small slice of motif discovery for comparison (e.g., SIATEC vs SIATEC_CS).
Keeps outputs in a run directory with metrics, predictions, stdout, and resolved config.
"""
import argparse
import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
np.bool = np.bool_  # compat

# Make motif_discovery imports available
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery" / "repeated_pattern_discovery"))

from motif_discovery.experiments import (  # type: ignore
    load_all_notes,
    load_all_motives,
    get_all_occurrences,
)
from SIA import find_motives  # type: ignore
from dataset import Dataset  # type: ignore
import new_algorithms  # type: ignore
import orig_algorithms  # type: ignore
from mir_eval.pattern import establishment_FPR, occurrence_FPR, three_layer_FPR  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run motif discovery on a small piece list for comparison."
    )
    parser.add_argument(
        "--method", type=str, required=True, choices=["CSA", "SIATEC", "SIATEC_CS"]
    )
    parser.add_argument(
        "--csv_note_dir",
        type=str,
        required=True,
        help="Directory with note CSVs (e.g., MNID predictions or raw notes).",
    )
    parser.add_argument(
        "--csv_label_dir",
        type=str,
        default="datasets/Beethoven_motif-main/csv_label",
    )
    parser.add_argument(
        "--motif_midi_dir",
        type=str,
        default="datasets/Beethoven_motif-main/motif_midi",
    )
    parser.add_argument(
        "--pieces",
        type=str,
        default="21-1,29-1,06-1",
        help="Comma-separated piece ids (e.g., 21-1,29-1,06-1).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Run output dir. Defaults to runs/slice_<method>-<timestamp>.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save motif predictions per piece as JSON.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of pieces to process in parallel (multiprocessing).",
    )
    parser.add_argument(
        "--pairwise_cap",
        type=int,
        default=None,
        help="Optional cap passed to audit later; unused here (kept for config trace).",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_yaml_like(data: Any, indent: int = 0) -> str:
    spaces = "  " * indent
    if isinstance(data, dict):
        lines = []
        for key, val in data.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{spaces}{key}:")
                lines.append(dump_yaml_like(val, indent + 1))
            else:
                lines.append(f"{spaces}{key}: {val}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{spaces}-")
                lines.append(dump_yaml_like(item, indent + 1))
            else:
                lines.append(f"{spaces}- {item}")
        return "\n".join(lines)
    return f"{spaces}{data}"


def run_csa(notes):
    patterns_est = find_motives(
        notes,
        horizontalTolerance=0,
        verticalTolerance=1,
        adjacentTolerance=(2, 12),
        min_notes=4,
        min_cardinality=0.5,
    )
    return patterns_est


def run_siatec(csv_path: Path):
    dataset = Dataset(str(csv_path))
    tecs = new_algorithms.siatechf(dataset, min_cr=2)
    patterns_est = [get_all_occurrences(tec) for tec in tecs if len(tec.get_translators())]
    return patterns_est


def run_siatec_cs(csv_path: Path):
    dataset = Dataset(str(csv_path))
    tecs = orig_algorithms.siatech_compress(dataset)
    patterns_est = [get_all_occurrences(tec) for tec in tecs if len(tec.get_translators())]
    # Expand to note spans (mirrors experiments.py)
    for i in range(len(patterns_est)):
        for j in range(len(patterns_est[i])):
            start = min([patterns_est[i][j][k][0] for k in range(len(patterns_est[i][j]))])
            end = max([patterns_est[i][j][k][0] for k in range(len(patterns_est[i][j]))])
            new_occ = []
            for k in range(len(dataset._vectors)):
                if dataset._vectors[k][0] >= start and dataset._vectors[k][0] <= end:
                    new_occ.append(tuple((dataset._vectors[k][0], dataset._vectors[k][1])))
            patterns_est[i][j] = list(new_occ)
    return patterns_est


def to_json_serializable(patterns_est):
    serializable = []
    for motif in patterns_est:
        occs = []
        for occ in motif:
            occs.append([[float(x[0]), int(x[1])] for x in occ])
        serializable.append(occs)
    return serializable


def evaluate(patterns_ref, patterns_est):
    F_est, P_est, R_est = establishment_FPR(patterns_ref, patterns_est)
    F_occ, P_occ, R_occ = occurrence_FPR(patterns_ref, patterns_est)
    F_thr, P_thr, R_thr = three_layer_FPR(patterns_ref, patterns_est)
    return {
        "establishment": {"precision": P_est, "recall": R_est, "f1": F_est},
        "occurrence": {"precision": P_occ, "recall": R_occ, "f1": F_occ},
        "three_layer": {"precision": P_thr, "recall": R_thr, "f1": F_thr},
    }


def run_one_task(params):
    piece, method, csv_note_dir, csv_label_dir, motif_midi_dir = params
    csv_note_dir = Path(csv_note_dir)
    csv_label_dir = Path(csv_label_dir)
    motif_midi_dir = Path(motif_midi_dir)
    csv_path = csv_note_dir / f"{piece}.csv"
    label_csv = csv_label_dir / f"{piece}.csv"
    midi_path = motif_midi_dir / f"{piece}.mid"

    notes = load_all_notes(str(csv_path))
    motives = load_all_motives(str(label_csv), str(midi_path))
    patterns_ref = [[list(occur[["onset", "pitch"]]) for occur in motif] for motif in motives.values()]

    t0 = time.time()
    if method == "CSA":
        patterns_est = run_csa(notes)
    elif method == "SIATEC":
        patterns_est = run_siatec(csv_path)
    else:
        patterns_est = run_siatec_cs(csv_path)
    elapsed = time.time() - t0

    stats = evaluate(patterns_ref, patterns_est)
    return piece, patterns_est, stats, elapsed


def main():
    args = parse_args()
    pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir).expanduser().resolve()
    else:
        run_dir = REPO_ROOT / "runs" / f"slice_{args.method.lower()}-{ts}"
    ensure_dir(run_dir)

    log_path = run_dir / "stdout.log"
    metrics_path = run_dir / "metrics.json"
    preds_dir = run_dir / "motifs"
    if args.save_predictions:
        ensure_dir(preds_dir)

    csv_note_dir = Path(args.csv_note_dir).expanduser().resolve()
    csv_label_dir = Path(args.csv_label_dir).expanduser().resolve()
    motif_midi_dir = Path(args.motif_midi_dir).expanduser().resolve()

    metrics = {"method": args.method, "pieces": {}, "aggregate": {}}

    with log_path.open("w") as lf:
        lf.write(f"method: {args.method}\n")
        lf.write(f"pieces: {pieces}\n")
        lf.write(f"csv_note_dir: {csv_note_dir}\n")

    f1s = {"est": [], "occ": [], "thr": []}

    results = []
    if args.num_workers > 1:
        work_items = [
            (p, args.method, str(csv_note_dir), str(csv_label_dir), str(motif_midi_dir))
            for p in pieces
        ]
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(run_one_task, item): item[0] for item in work_items}
            for fut in as_completed(futures):
                results.append(fut.result())
    else:
        for p in pieces:
            results.append(
                run_one_task((p, args.method, str(csv_note_dir), str(csv_label_dir), str(motif_midi_dir)))
            )

    for piece, patterns_est, stats, elapsed in results:
        metrics["pieces"][piece] = {
            "metrics": stats,
            "num_motifs": len(patterns_est),
            "runtime_sec": elapsed,
        }
        f1s["est"].append(stats["establishment"]["f1"])
        f1s["occ"].append(stats["occurrence"]["f1"])
        f1s["thr"].append(stats["three_layer"]["f1"])

        if args.save_predictions:
            payload = {"piece": piece, "patterns": to_json_serializable(patterns_est)}
            (preds_dir / f"{piece}.json").write_text(json.dumps(payload))

        with log_path.open("a") as lf:
            lf.write(
                f"{piece}: est F1 {stats['establishment']['f1']:.4f}, "
                f"occ F1 {stats['occurrence']['f1']:.4f}, "
                f"thr F1 {stats['three_layer']['f1']:.4f}, "
                f"motifs {len(patterns_est)}, time {elapsed:.2f}s\n"
            )

    # Aggregate means
    if pieces:
        metrics["aggregate"] = {
            "mean_est_f1": float(np.mean(f1s["est"])),
            "mean_occ_f1": float(np.mean(f1s["occ"])),
            "mean_thr_f1": float(np.mean(f1s["thr"])),
        }

    metrics_path.write_text(json.dumps(metrics, indent=2))

    cfg = {
        "method": args.method,
        "csv_note_dir": str(csv_note_dir),
        "csv_label_dir": str(csv_label_dir),
        "motif_midi_dir": str(motif_midi_dir),
        "pieces": pieces,
        "output_dir": str(run_dir),
        "save_predictions": args.save_predictions,
        "pairwise_cap": args.pairwise_cap,
        "num_workers": args.num_workers,
    }
    (run_dir / "config_resolved.yaml").write_text(dump_yaml_like(cfg) + "\n")

    print(f"Slice run complete. metrics: {metrics_path}")
    if args.save_predictions:
        print(f"Motif predictions: {preds_dir}")


if __name__ == "__main__":
    main()
