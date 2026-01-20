#!/usr/bin/env python3
"""
Analyze motif occurrence spans and note counts to guide segment window defaults.

Reads motif labels (CSV + MIDI) using existing loaders and prints summary stats.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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


def _parse_float_list(raw: Optional[str]) -> List[float]:
    if not raw:
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _round_onset(x: float) -> float:
    return float(int(round(float(x) * 100.0))) / 100.0


def _load_predictions_json(path: Path) -> List[List[List[Tuple[float, int]]]]:
    payload = json.loads(path.read_text())
    raw = payload.get("patterns", [])
    patterns: List[List[List[Tuple[float, int]]]] = []
    for motif in raw:
        occs: List[List[Tuple[float, int]]] = []
        for occ in motif:
            occs.append([(_round_onset(float(o[0])), int(o[1])) for o in occ])
        patterns.append(occs)
    return patterns


def _occ_span_onset(occ: Sequence[Tuple[float, int]]) -> float:
    if not occ:
        return 0.0
    times = [o[0] for o in occ]
    if len(times) < 2:
        return 0.0
    return float(max(times) - min(times))


def _occ_start_end_onset(occ: Sequence[Tuple[float, int]]) -> Tuple[float, float]:
    if not occ:
        return 0.0, 0.0
    times = [o[0] for o in occ]
    return float(min(times)), float(max(times))


def _note_jaccard(a: Sequence[Tuple[float, int]], b: Sequence[Tuple[float, int]]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _assign_span_to_scale(span: float, scales: Sequence[float], mode: str) -> str:
    if not scales:
        return "unbinned"
    scales_sorted = sorted(float(s) for s in scales)
    if mode == "closest":
        best = min(scales_sorted, key=lambda s: abs(s - span))
        return f"{best:g}"
    # default: "ceil" => smallest scale that covers the span
    for s in scales_sorted:
        if span <= s:
            return f"{s:g}"
    return f">{scales_sorted[-1]:g}"


def _safe_stats_int(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float32)
    return percentile_stats(arr)


def _load_note_events_bps(note_csv_dir: Path, piece: str) -> set[Tuple[float, int]]:
    # Minimal CSV reader to avoid extra deps.
    import csv

    path = note_csv_dir / f"{piece}.csv"
    if not path.exists():
        return set()
    out: set[Tuple[float, int]] = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            onset_raw = row.get("onset", "")
            pitch_raw = row.get("midi_number", "") or row.get("pitch", "")
            if onset_raw is None or pitch_raw is None:
                continue
            try:
                onset = _round_onset(float(onset_raw))
                pitch = int(float(pitch_raw))
            except Exception:
                continue
            out.add((onset, pitch))
    return out


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
        "--plots-dir",
        type=str,
        default=None,
        help="Optional directory to save distribution plots (PNG).",
    )
    parser.add_argument(
        "--plot-bins",
        type=int,
        default=50,
        help="Number of bins for histograms when --plots-dir is set.",
    )
    parser.add_argument(
        "--segments-dir",
        type=str,
        default=None,
        help="Optional directory of segment JSONs (from scripts/run_segments.py) to analyze max_len.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        default=None,
        help="Optional directory of per-piece predicted motif JSONs (from experiments.py --save_predictions_dir).",
    )
    parser.add_argument(
        "--scale-lengths",
        type=str,
        default="2.0,4.0,8.0,16.0",
        help="Comma-separated segment scale lengths used for binning predicted occurrence spans.",
    )
    parser.add_argument(
        "--span-assign",
        type=str,
        default="ceil",
        choices=["ceil", "closest"],
        help="How to assign a predicted occurrence span to a segment scale bin.",
    )
    parser.add_argument(
        "--csv-note-dir",
        type=str,
        default=None,
        help="Optional note CSV dir for coverage stats (expects <piece>.csv with onset+midi_number).",
    )
    parser.add_argument(
        "--within-motif-max-occ",
        type=int,
        default=50,
        help="When analyzing predicted motifs: cap occurrences per motif for within-motif overlap stats (avoids O(n^2) blowups).",
    )
    parser.add_argument(
        "--within-motif-max-pairs",
        type=int,
        default=500,
        help="When analyzing predicted motifs: cap occurrence pairs per motif for within-motif overlap stats.",
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

    # Predicted motif stats (optional)
    pred_scales = _parse_float_list(args.scale_lengths)
    pred_piece_summaries: Dict[str, Dict] = {}
    pred_motifs_per_piece: List[int] = []
    pred_occurrences_per_piece: List[int] = []
    pred_occurrences_per_motif: List[int] = []
    pred_occ_spans: List[float] = []
    pred_occ_note_counts: List[int] = []
    pred_bin_occ_counts: Dict[str, int] = {}
    pred_bin_motif_counts: Dict[str, int] = {}
    pred_bin_occ_per_piece: Dict[str, Dict[str, int]] = {}
    pred_bin_motif_per_piece: Dict[str, Dict[str, int]] = {}
    pred_note_coverage: List[float] = []
    pred_note_multiplicity: List[float] = []
    pred_within_motif_max_time_iou: List[float] = []
    pred_within_motif_mean_time_iou: List[float] = []
    pred_within_motif_max_note_j: List[float] = []
    pred_within_motif_mean_note_j: List[float] = []

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

    if args.predictions_dir:
        pred_dir = Path(args.predictions_dir)
        if args.pieces:
            pred_pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
        else:
            pred_pieces = [p.stem for p in sorted(pred_dir.glob("*.json"))]

        note_dir = Path(args.csv_note_dir) if args.csv_note_dir else None
        use_note_coverage = bool(note_dir) and args.dataset == "bps"

        for piece in pred_pieces:
            pred_path = pred_dir / f"{piece}.json"
            if not pred_path.exists():
                continue
            motifs = _load_predictions_json(pred_path)
            num_motifs = len(motifs)
            total_occ = sum(len(m) for m in motifs)
            pred_motifs_per_piece.append(num_motifs)
            pred_occurrences_per_piece.append(total_occ)

            # Per-scale binning (approximate by onset-span)
            piece_occ_bins: Dict[str, int] = {}
            piece_motif_bins: Dict[str, int] = {}

            # Coverage and multiplicity are optional (needs note CSV)
            all_notes = _load_note_events_bps(note_dir, piece) if use_note_coverage and note_dir else set()
            pred_notes_union: set[Tuple[float, int]] = set()
            pred_notes_total = 0

            for motif in motifs:
                pred_occurrences_per_motif.append(len(motif))

                # Assign motif to a scale bin by median occurrence span.
                motif_spans = [_occ_span_onset(occ) for occ in motif if occ]
                if motif_spans:
                    motif_rep = float(np.median(np.asarray(motif_spans, dtype=np.float32)))
                else:
                    motif_rep = 0.0
                motif_bin = _assign_span_to_scale(motif_rep, pred_scales, args.span_assign)
                piece_motif_bins[motif_bin] = piece_motif_bins.get(motif_bin, 0) + 1
                pred_bin_motif_counts[motif_bin] = pred_bin_motif_counts.get(motif_bin, 0) + 1

                # Within-motif overlap: do occurrences overlap heavily with each other?
                # This captures "duplicate occurrences within one motif" and "overly broad windows".
                within_time_sum = 0.0
                within_note_sum = 0.0
                within_pairs = 0
                within_time_max = 0.0
                within_note_max = 0.0
                if len(motif) >= 2 and args.within_motif_max_pairs != 0:
                    # Use a stable subset of occurrences (largest first) to keep runtime bounded.
                    occs_sorted = sorted(motif, key=lambda o: len(o), reverse=True)
                    occs_sorted = occs_sorted[: max(2, int(args.within_motif_max_occ))]
                    spans = [_occ_start_end_onset(occ) for occ in occs_sorted]
                    for i in range(len(occs_sorted)):
                        for j in range(i + 1, len(occs_sorted)):
                            ti = time_iou(spans[i], spans[j])
                            nj = _note_jaccard(occs_sorted[i], occs_sorted[j])
                            within_time_sum += float(ti)
                            within_note_sum += float(nj)
                            within_pairs += 1
                            within_time_max = max(within_time_max, float(ti))
                            within_note_max = max(within_note_max, float(nj))
                            if within_pairs >= int(args.within_motif_max_pairs):
                                break
                        if within_pairs >= int(args.within_motif_max_pairs):
                            break

                if within_pairs > 0:
                    pred_within_motif_max_time_iou.append(within_time_max)
                    pred_within_motif_mean_time_iou.append(within_time_sum / within_pairs)
                    pred_within_motif_max_note_j.append(within_note_max)
                    pred_within_motif_mean_note_j.append(within_note_sum / within_pairs)

                for occ in motif:
                    if not occ:
                        continue
                    span = _occ_span_onset(occ)
                    pred_occ_spans.append(span)
                    pred_occ_note_counts.append(len(occ))
                    occ_bin = _assign_span_to_scale(span, pred_scales, args.span_assign)
                    piece_occ_bins[occ_bin] = piece_occ_bins.get(occ_bin, 0) + 1
                    pred_bin_occ_counts[occ_bin] = pred_bin_occ_counts.get(occ_bin, 0) + 1

                    if use_note_coverage:
                        pred_notes_total += len(occ)
                        for onset, pitch in occ:
                            pred_notes_union.add((_round_onset(onset), int(pitch)))

            pred_bin_occ_per_piece[piece] = piece_occ_bins
            pred_bin_motif_per_piece[piece] = piece_motif_bins

            if use_note_coverage and all_notes:
                coverage = len(pred_notes_union) / max(1, len(all_notes))
                # average times each unique predicted note is re-used across occurrences
                multiplicity = pred_notes_total / max(1, len(pred_notes_union))
                pred_note_coverage.append(float(coverage))
                pred_note_multiplicity.append(float(multiplicity))

            pred_piece_summaries[piece] = {
                "num_motifs": int(num_motifs),
                "total_occurrences": int(total_occ),
                "occurrence_span_bins": piece_occ_bins,
                "motif_span_bins": piece_motif_bins,
            }

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

    pred_summary = {}
    if args.predictions_dir:
        def _bin_sort_key(key: str) -> Tuple[int, float, str]:
            # Stable ordering: numeric bins first, then ">" bins, then others.
            try:
                if key.startswith(">"):
                    return (1, float(key[1:]), key)
                return (0, float(key), key)
            except Exception:
                return (2, 0.0, key)

        pred_summary = {
            "num_pieces": len(pred_piece_summaries),
            "scale_lengths": [float(x) for x in pred_scales],
            "span_assign": str(args.span_assign),
            "motifs_per_piece_stats": _safe_stats_int(pred_motifs_per_piece),
            "occurrences_per_piece_stats": _safe_stats_int(pred_occurrences_per_piece),
            "occurrences_per_motif_stats": _safe_stats_int(pred_occurrences_per_motif),
            "pred_occurrence_span_stats": percentile_stats(np.asarray(pred_occ_spans, dtype=np.float32)) if pred_occ_spans else {},
            "pred_notes_per_occurrence_stats": _safe_stats_int(pred_occ_note_counts),
            "occurrence_bins_total": pred_bin_occ_counts,
            "motif_bins_total": pred_bin_motif_counts,
        }
        if pred_note_coverage:
            pred_summary["note_coverage_stats"] = percentile_stats(np.asarray(pred_note_coverage, dtype=np.float32))
        if pred_note_multiplicity:
            pred_summary["note_multiplicity_stats"] = percentile_stats(np.asarray(pred_note_multiplicity, dtype=np.float32))
        if pred_within_motif_max_time_iou:
            pred_summary["within_motif_max_time_iou_stats"] = percentile_stats(
                np.asarray(pred_within_motif_max_time_iou, dtype=np.float32)
            )
        if pred_within_motif_mean_time_iou:
            pred_summary["within_motif_mean_time_iou_stats"] = percentile_stats(
                np.asarray(pred_within_motif_mean_time_iou, dtype=np.float32)
            )
        if pred_within_motif_max_note_j:
            pred_summary["within_motif_max_note_jaccard_stats"] = percentile_stats(
                np.asarray(pred_within_motif_max_note_j, dtype=np.float32)
            )
        if pred_within_motif_mean_note_j:
            pred_summary["within_motif_mean_note_jaccard_stats"] = percentile_stats(
                np.asarray(pred_within_motif_mean_note_j, dtype=np.float32)
            )

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

    if pred_summary:
        print("Predicted motif summary (from --predictions-dir):")
        if pred_summary.get("motifs_per_piece_stats"):
            mps = pred_summary["motifs_per_piece_stats"]
            ops = pred_summary["occurrences_per_piece_stats"]
            print(f"  motifs per piece: p50 {mps.get('p50', 0):.1f} p90 {mps.get('p90', 0):.1f} max {mps.get('max', 0):.1f}")
            print(f"  occs per piece:  p50 {ops.get('p50', 0):.1f} p90 {ops.get('p90', 0):.1f} max {ops.get('max', 0):.1f}")
        if pred_summary.get("occurrences_per_motif_stats"):
            opm = pred_summary["occurrences_per_motif_stats"]
            print(f"  occs per motif:  p50 {opm.get('p50', 0):.1f} p90 {opm.get('p90', 0):.1f} max {opm.get('max', 0):.1f}")
        if pred_summary.get("occurrence_bins_total"):
            print("  occurrences by span bin:")
            for k in sorted(pred_summary["occurrence_bins_total"].keys(), key=_bin_sort_key):
                print(f"    {k}: {pred_summary['occurrence_bins_total'][k]}")
        if pred_summary.get("note_coverage_stats"):
            cov = pred_summary["note_coverage_stats"]
            mult = pred_summary.get("note_multiplicity_stats", {})
            print(f"  note coverage:   p50 {cov.get('p50', 0):.3f} p90 {cov.get('p90', 0):.3f}")
            if mult:
                print(f"  note reuse:      p50 {mult.get('p50', 0):.2f} p90 {mult.get('p90', 0):.2f}")
        if pred_summary.get("within_motif_max_time_iou_stats"):
            wi = pred_summary["within_motif_max_time_iou_stats"]
            wj = pred_summary.get("within_motif_max_note_jaccard_stats", {})
            print(f"  within-motif max time IoU: p50 {wi.get('p50', 0):.3f} p90 {wi.get('p90', 0):.3f}")
            if wj:
                print(f"  within-motif max note J:   p50 {wj.get('p50', 0):.3f} p90 {wj.get('p90', 0):.3f}")

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
            "predicted": pred_summary,
            "predicted_by_piece": pred_piece_summaries,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved stats to {out_path}")

    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib  # type: ignore

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "matplotlib is required to save plots; install it or omit --plots-dir."
            ) from exc

        def _plot_hist(data: List[float], title: str, xlabel: str, filename: str) -> None:
            if not data:
                return
            arr = np.asarray(data, dtype=np.float32)
            if arr.size < 2:
                return
            plt.figure(figsize=(6, 4))
            plt.hist(arr, bins=int(args.plot_bins), color="#4c72b0", edgecolor="white")
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(plots_dir / filename, dpi=200)
            plt.close()

        unit = "beats" if args.dataset == "bps" else "time"
        _plot_hist(
            all_spans,
            f"Occurrence spans ({args.dataset})",
            f"span ({unit})",
            "occurrence_spans.png",
        )
        _plot_hist(
            [float(x) for x in all_counts],
            f"Notes per occurrence ({args.dataset})",
            "notes per occurrence",
            "occurrence_note_counts.png",
        )
        _plot_hist(dp_all, f"Pitch steps Δp ({args.dataset})", "Δp (semitones)", "dp.png")
        _plot_hist(log_dt_all, f"log IOI ({args.dataset})", "log(Δt)", "log_dt.png")
        _plot_hist(
            centered_log_dt_all,
            f"Centered log IOI ({args.dataset})",
            "log(Δt) - g",
            "centered_log_dt.png",
        )
        _plot_hist(g_all, f"Tempo scalar g ({args.dataset})", "g", "g.png")

        if same_iou:
            _plot_hist(
                same_iou,
                f"Time IoU (same motif id) ({args.dataset})",
                "IoU",
                "time_iou_same.png",
            )
        if diff_iou:
            _plot_hist(
                diff_iou,
                f"Time IoU (different motif id) ({args.dataset})",
                "IoU",
                "time_iou_diff.png",
            )
        if seg_note_counts:
            _plot_hist(
                [float(x) for x in seg_note_counts],
                f"Segment note counts ({args.dataset})",
                "notes per segment",
                "segment_note_counts.png",
            )
        if seg_delta_counts:
            _plot_hist(
                [float(x) for x in seg_delta_counts],
                f"Segment delta lengths (notes-1) ({args.dataset})",
                "deltas per segment",
                "segment_delta_lengths.png",
            )

        if args.predictions_dir:
            _plot_hist(
                [float(x) for x in pred_motifs_per_piece],
                "Predicted motifs per piece",
                "motifs per piece",
                "pred_motifs_per_piece.png",
            )
            _plot_hist(
                [float(x) for x in pred_occurrences_per_piece],
                "Predicted occurrences per piece",
                "occurrences per piece",
                "pred_occurrences_per_piece.png",
            )
            _plot_hist(
                [float(x) for x in pred_occurrences_per_motif],
                "Predicted occurrences per motif",
                "occurrences per motif",
                "pred_occurrences_per_motif.png",
            )
            _plot_hist(
                [float(x) for x in pred_occ_spans],
                "Predicted occurrence onset-spans",
                "span (beats, onset-only)",
                "pred_occurrence_spans.png",
            )
            _plot_hist(
                [float(x) for x in pred_occ_note_counts],
                "Predicted notes per occurrence",
                "notes per occurrence",
                "pred_notes_per_occurrence.png",
            )
            if pred_note_coverage:
                _plot_hist(
                    [float(x) for x in pred_note_coverage],
                    "Predicted note coverage (unique notes / total notes)",
                    "coverage",
                    "pred_note_coverage.png",
                )
            if pred_note_multiplicity:
                _plot_hist(
                    [float(x) for x in pred_note_multiplicity],
                    "Predicted note reuse (total note assignments / unique notes)",
                    "reuse",
                    "pred_note_reuse.png",
                )
            if pred_within_motif_max_time_iou:
                _plot_hist(
                    [float(x) for x in pred_within_motif_max_time_iou],
                    "Within-motif max time IoU (predicted)",
                    "max IoU within motif",
                    "pred_within_motif_max_time_iou.png",
                )
            if pred_within_motif_max_note_j:
                _plot_hist(
                    [float(x) for x in pred_within_motif_max_note_j],
                    "Within-motif max note Jaccard (predicted)",
                    "max note J within motif",
                    "pred_within_motif_max_note_jaccard.png",
                )

        print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    main()
