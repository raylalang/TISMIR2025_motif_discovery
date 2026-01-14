#!/usr/bin/env python3
"""
Analyze motif occurrence spans and note counts to guide segment window defaults.

Reads motif labels (CSV + MIDI) using existing loaders and prints summary stats.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Make motif_discovery imports available when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery" / "repeated_pattern_discovery"))

from motif_discovery.experiments import load_all_motives  # type: ignore
# JKU loaders
sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))
from experiments_jkupdd import (  # type: ignore
    jkupdd_corpus,
    jkupdd_notes_csv,
    jkupdd_data_dir,
    load_jkupdd_patterns_csv,
    load_jkupdd_notes_csv,
)


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


def analyze_piece(label_csv: Path, midi_path: Path) -> Tuple[List[float], List[int]]:
    motives = load_all_motives(str(label_csv), str(midi_path))
    spans: List[float] = []
    note_counts: List[int] = []
    for motif_occurs in motives.values():
        for occ in motif_occurs:
            if occ.size == 0:
                continue
            onset = float(np.min(occ["onset"]))
            end = float(np.max(occ["end"]))
            spans.append(end - onset)
            note_counts.append(int(occ.size))
    return spans, note_counts


def occurrence_steps(occ) -> Tuple[List[float], List[float], List[float], float]:
    if occ.size < 2:
        return [], [], [], 0.0
    pitches = occ["pitch"].astype(np.float32)
    onsets = occ["onset"].astype(np.float32)
    order = np.argsort(onsets)
    pitches = pitches[order]
    onsets = onsets[order]
    dp = np.diff(pitches)
    dt = np.diff(onsets)
    log_dt = np.log(np.clip(dt, 1e-4, None))
    g = float(np.mean(log_dt)) if log_dt.size else 0.0
    centered = log_dt - g if log_dt.size else log_dt
    return dp.tolist(), log_dt.tolist(), centered.tolist(), g


def time_iou(span_a: Tuple[float, float], span_b: Tuple[float, float]) -> float:
    a0, a1 = span_a
    b0, b1 = span_b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Summarize motif spans and note counts for window default selection."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bps",
        choices=["bps", "jkupdd"],
        help="Dataset to analyze: bps (Beethoven) or jkupdd.",
    )
    parser.add_argument(
        "--csv-label-dir",
        type=str,
        default=str(REPO_ROOT / "datasets/Beethoven_motif-main/csv_label"),
        help="(bps) Directory with motif label CSVs.",
    )
    parser.add_argument(
        "--motif-midi-dir",
        type=str,
        default=str(REPO_ROOT / "datasets/Beethoven_motif-main/motif_midi"),
        help="(bps) Directory with motif MIDI files.",
    )
    parser.add_argument(
        "--pieces",
        type=str,
        default=None,
        help="Optional comma-separated piece ids. Defaults to all files in label dir (bps) or all jkupdd pieces.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save stats as JSON.",
    )
    parser.add_argument(
        "--segments-dir",
        type=str,
        default=None,
        help="Optional directory of segment JSONs (from scripts/run_segments.py) to analyze max_len.",
    )
    parser.add_argument(
        "--jkupdd-dir",
        type=str,
        default=str(REPO_ROOT / "motif_discovery/JKUPDD/JKUPDD-noAudio-Aug2013"),
        help="(jkupdd) Root of JKUPDD groundTruth directory.",
    )
    args = parser.parse_args()

    all_spans: List[float] = []
    all_counts: List[int] = []
    dp_all: List[float] = []
    log_dt_all: List[float] = []
    centered_log_dt_all: List[float] = []
    g_all: List[float] = []
    same_iou: List[float] = []
    diff_iou: List[float] = []
    seg_note_counts: List[int] = []
    seg_delta_counts: List[int] = []
    seg_scale_note_counts: Dict[str, List[int]] = {}
    seg_scale_delta_counts: Dict[str, List[int]] = {}

    if args.dataset == "bps":
        label_dir = Path(args.csv_label_dir)
        midi_dir = Path(args.motif_midi_dir)
        if args.pieces:
            pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
        else:
            pieces = [p.stem for p in sorted(label_dir.glob("*.csv"))]
        for piece in pieces:
            label_csv = label_dir / f"{piece}.csv"
            midi_path = midi_dir / f"{piece}.mid"
            spans, counts = analyze_piece(label_csv, midi_path)
            all_spans.extend(spans)
            all_counts.extend(counts)
            motives = load_all_motives(str(label_csv), str(midi_path))
            occ_spans = []
            for motif_id, motif_occurs in motives.items():
                for occ in motif_occurs:
                    dp, ldt, cldt, g = occurrence_steps(occ)
                    dp_all.extend(dp)
                    log_dt_all.extend(ldt)
                    centered_log_dt_all.extend(cldt)
                    g_all.append(g)
                    if occ.size:
                        occ_spans.append((motif_id, float(np.min(occ["onset"])), float(np.max(occ["end"]))))
            # time IoU same/diff motif
            for i in range(len(occ_spans)):
                mi, s0, s1 = occ_spans[i]
                for j in range(i + 1, len(occ_spans)):
                    mj, t0, t1 = occ_spans[j]
                    iou = time_iou((s0, s1), (t0, t1))
                    if mi == mj:
                        same_iou.append(iou)
                    else:
                        diff_iou.append(iou)
    else:  # jkupdd
        if args.pieces:
            pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
        else:
            pieces = jkupdd_corpus
        jk_root = Path(args.jkupdd_dir)
        for corpus_name, csv_name in zip(jkupdd_corpus, jkupdd_notes_csv):
            if corpus_name not in pieces:
                continue
            pattern_dir = jk_root / corpus_name / "monophonic" / "repeatedPatterns"
            note_csv = jk_root / corpus_name / "polyphonic" / "csv" / csv_name
            poly_notes = load_jkupdd_notes_csv(str(note_csv))
            max_onset = poly_notes[-1][0] if len(poly_notes) else 1e9
            patterns = load_jkupdd_patterns_csv(str(pattern_dir), max_note_onset=max_onset)
            for occs in patterns:
                for occ in occs:
                    onsets = [o[0] for o in occ]
                    if not onsets:
                        continue
                    all_spans.append(max(onsets) - min(onsets))
                    all_counts.append(len(occ))
                    occ_on = np.array([o[0] for o in occ], dtype=np.float32)
                    occ_pitch = np.array([o[1] for o in occ], dtype=np.float32)
                    order = np.argsort(occ_on)
                    dp = np.diff(occ_pitch[order])
                    dt = np.diff(occ_on[order])
                    ldt = np.log(np.clip(dt, 1e-4, None))
                    g = float(np.mean(ldt)) if ldt.size else 0.0
                    cldt = ldt - g if ldt.size else ldt
                    dp_all.extend(dp.tolist())
                    log_dt_all.extend(ldt.tolist())
                    centered_log_dt_all.extend(cldt.tolist())
                    g_all.append(g)

    if args.segments_dir:
        seg_dir = Path(args.segments_dir)
        if args.pieces:
            seg_pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
        else:
            seg_pieces = [p.stem for p in sorted(seg_dir.glob("*.json"))]
        for piece in seg_pieces:
            seg_path = seg_dir / f"{piece}.json"
            if not seg_path.exists():
                raise FileNotFoundError(f"Missing segment JSON: {seg_path}")
            payload = json.loads(seg_path.read_text())
            scales = payload.get("scales", [])
            for seg in payload.get("segments", []):
                n_notes = len(seg.get("note_indices", []))
                if n_notes <= 0:
                    continue
                seg_note_counts.append(n_notes)
                seg_delta_counts.append(max(0, n_notes - 1))
                scale_id = seg.get("scale_id")
                if scale_id is not None:
                    if scale_id < len(scales):
                        scale_key = f"{scales[scale_id]}"
                    else:
                        scale_key = f"scale_{scale_id}"
                    seg_scale_note_counts.setdefault(scale_key, []).append(n_notes)
                    seg_scale_delta_counts.setdefault(scale_key, []).append(max(0, n_notes - 1))

    span_stats = percentile_stats(np.array(all_spans)) if all_spans else {}
    count_stats = percentile_stats(np.array(all_counts)) if all_counts else {}
    dp_stats = percentile_stats(np.array(dp_all)) if dp_all else {}
    log_dt_stats = percentile_stats(np.array(log_dt_all)) if log_dt_all else {}
    cldt_stats = percentile_stats(np.array(centered_log_dt_all)) if centered_log_dt_all else {}
    g_stats = percentile_stats(np.array(g_all)) if g_all else {}
    same_iou_stats = percentile_stats(np.array(same_iou)) if same_iou else {}
    diff_iou_stats = percentile_stats(np.array(diff_iou)) if diff_iou else {}
    seg_note_stats = percentile_stats(np.array(seg_note_counts)) if seg_note_counts else {}
    seg_delta_stats = percentile_stats(np.array(seg_delta_counts)) if seg_delta_counts else {}
    seg_scale_note_stats = {
        k: percentile_stats(np.array(v)) for k, v in seg_scale_note_counts.items()
    }
    seg_scale_delta_stats = {
        k: percentile_stats(np.array(v)) for k, v in seg_scale_delta_counts.items()
    }

    print(f"Dataset: {args.dataset}")
    print(f"Pieces analyzed: {len(pieces)}")
    print(f"Total occurrences: {len(all_spans)}")
    if span_stats:
        print("Occurrence spans:")
        for k, v in span_stats.items():
            print(f"  {k:>3}: {v:.3f}")
    if count_stats:
        print("Note counts per occurrence:")
        for k, v in count_stats.items():
            print(f"  {k:>3}: {v:.2f}")
    if dp_stats:
        print("Pitch step (dp) stats:")
        for k, v in dp_stats.items():
            print(f"  {k:>3}: {v:.3f}")
    if log_dt_stats:
        print("Log IOI stats:")
        for k, v in log_dt_stats.items():
            print(f"  {k:>3}: {v:.3f}")
    if cldt_stats:
        print("Centered log IOI stats:")
        for k, v in cldt_stats.items():
            print(f"  {k:>3}: {v:.3f}")
    if g_stats:
        print("Tempo scalar g stats:")
        for k, v in g_stats.items():
            print(f"  {k:>3}: {v:.3f}")
    if same_iou_stats:
        print("Time IoU (same motif) stats:")
        for k, v in same_iou_stats.items():
            print(f"  {k:>3}: {v:.3f}")
    if diff_iou_stats:
        print("Time IoU (diff motif) stats:")
        for k, v in diff_iou_stats.items():
            print(f"  {k:>3}: {v:.3f}")
    if seg_note_stats:
        print("Segment note counts:")
        for k, v in seg_note_stats.items():
            print(f"  {k:>3}: {v:.2f}")
    if seg_delta_stats:
        print("Segment delta lengths (notes-1):")
        for k, v in seg_delta_stats.items():
            print(f"  {k:>3}: {v:.2f}")
    if seg_scale_delta_stats:
        print("Segment delta lengths by scale:")
        for scale_key in sorted(seg_scale_delta_stats.keys()):
            stats = seg_scale_delta_stats[scale_key]
            print(f"  scale {scale_key}: p50 {stats['p50']:.2f} p90 {stats['p90']:.2f} p95 {stats['p95']:.2f} p99 {stats['p99']:.2f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pieces": pieces,
            "num_occurrences": len(all_spans),
            "span_stats": span_stats,
            "note_count_stats": count_stats,
            "dp_stats": dp_stats,
            "log_dt_stats": log_dt_stats,
            "centered_log_dt_stats": cldt_stats,
            "g_stats": g_stats,
            "time_iou_same": same_iou_stats,
            "time_iou_diff": diff_iou_stats,
            "segment_note_count_stats": seg_note_stats,
            "segment_delta_count_stats": seg_delta_stats,
            "segment_scale_note_count_stats": seg_scale_note_stats,
            "segment_scale_delta_count_stats": seg_scale_delta_stats,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved stats to {out_path}")


if __name__ == "__main__":
    main()
