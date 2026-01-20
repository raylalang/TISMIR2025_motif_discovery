#!/usr/bin/env python3
"""
Piano-roll visualization for motif predictions (and optional GT labels).

Primary use-case: visualize redundancy/duplicates flagged by audit_predictions.py by
highlighting specific motif IDs from a per-piece prediction JSON.

Inputs:
  - Note CSV (with onset, midi_number, duration columns) for the piece
  - Predicted motifs JSON from motif_discovery/experiments.py (--save_predictions_dir),
    format: {"piece": "21-1", "patterns": [ motif -> [occurrence -> [[onset,pitch],...]], ... ] }
Optional:
  - audit_summary.json or failure_cases.json from scripts/audit_predictions.py to pick a motif pair.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Note:
    onset: float
    pitch: int
    duration: float
    gt_type: str


def _round_onset(x: float) -> float:
    return float(int(round(float(x) * 100.0))) / 100.0


def load_note_csv(path: Path) -> List[Note]:
    notes: List[Note] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            onset = _round_onset(float(row.get("onset", "0") or 0.0))
            pitch = int(row.get("midi_number") or row.get("pitch") or 0)
            duration = float(row.get("duration") or 0.0)
            gt_type = str(row.get("type") or "").strip()
            if duration <= 0:
                continue
            notes.append(Note(onset=onset, pitch=pitch, duration=duration, gt_type=gt_type))
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def load_patterns_json(path: Path) -> Tuple[str, List[List[List[Tuple[float, int]]]]]:
    payload = json.loads(path.read_text())
    piece = str(payload.get("piece") or path.stem)
    raw_patterns = payload.get("patterns", [])
    patterns: List[List[List[Tuple[float, int]]]] = []
    for motif in raw_patterns:
        occs: List[List[Tuple[float, int]]] = []
        for occ in motif:
            occs.append([(_round_onset(float(o[0])), int(o[1])) for o in occ])
        patterns.append(occs)
    return piece, patterns


def _parse_id_list(raw: Optional[str]) -> List[int]:
    if not raw:
        return []
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _iter_audit_candidates(audit_payload: dict) -> Iterable[dict]:
    if "top_near_duplicates" in audit_payload:
        yield from audit_payload.get("top_near_duplicates", [])
    elif "redundant_clusters" in audit_payload:
        yield from audit_payload.get("redundant_clusters", [])

def _iter_sparse_candidates(audit_payload: dict) -> Iterable[dict]:
    if "top_sparse" in audit_payload:
        yield from audit_payload.get("top_sparse", [])
    elif "sparse_motifs" in audit_payload:
        yield from audit_payload.get("sparse_motifs", [])


def pick_motif_ids_from_audit(
    audit_path: Path, index: int = 0, piece: Optional[str] = None
) -> Tuple[str, List[int], str]:
    payload = json.loads(audit_path.read_text())

    candidates = list(_iter_audit_candidates(payload))
    if piece:
        candidates = [c for c in candidates if str(c.get("piece")) == piece]
        if not candidates:
            raise ValueError(f"No near-duplicate entries for piece '{piece}' in {audit_path}")
    if not candidates:
        raise ValueError(f"No near-duplicate entries found in {audit_path}")
    if index < 0 or index >= len(candidates):
        raise ValueError(f"audit index {index} out of range [0, {len(candidates)-1}]")

    entry = candidates[index]
    piece_id = str(entry.get("piece"))
    motif_a = int(entry.get("motif_a"))
    motif_b = int(entry.get("motif_b"))
    note_j = entry.get("note_jaccard")
    time_i = entry.get("time_iou")
    meta = f"near-dup idx={index} note_jaccard={round(note_j, 2)} time_iou={round(time_i, 2)}"
    return piece_id, [motif_a, motif_b], meta


def pick_sparse_motif_from_audit(
    audit_path: Path, index: int = 0, piece: Optional[str] = None
) -> Tuple[str, List[int], str]:
    payload = json.loads(audit_path.read_text())
    candidates = list(_iter_sparse_candidates(payload))
    if piece:
        candidates = [c for c in candidates if str(c.get("piece")) == piece]
        if not candidates:
            raise ValueError(f"No sparse-motif entries for piece '{piece}' in {audit_path}")
    if not candidates:
        raise ValueError(f"No sparse-motif entries found in {audit_path}")
    if index < 0 or index >= len(candidates):
        raise ValueError(f"audit index {index} out of range [0, {len(candidates)-1}]")

    entry = candidates[index]
    piece_id = str(entry.get("piece"))
    motif_id = int(entry.get("motif"))
    ratio = entry.get("span_per_note", entry.get("ratio"))
    meta = f"sparse idx={index} span_per_note={ratio}"
    return piece_id, [motif_id], meta


def build_note_index(notes: Sequence[Note]) -> Dict[Tuple[float, int], List[int]]:
    idx: Dict[Tuple[float, int], List[int]] = {}
    for i, n in enumerate(notes):
        idx.setdefault((_round_onset(n.onset), int(n.pitch)), []).append(i)
    return idx


def motif_occurrence_spans(occs: Sequence[Sequence[Tuple[float, int]]]) -> List[Tuple[float, float]]:
    spans: List[Tuple[float, float]] = []
    for occ in occs:
        if not occ:
            continue
        times = [o[0] for o in occ]
        spans.append((float(min(times)), float(max(times))))
    return spans


def occurrence_span(occ: Sequence[Tuple[float, int]]) -> Tuple[float, float]:
    if not occ:
        return 0.0, 0.0
    times = [o[0] for o in occ]
    return float(min(times)), float(max(times))


def note_jaccard(a: Sequence[Tuple[float, int]], b: Sequence[Tuple[float, int]]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = set_a & set_b
    union = set_a | set_b
    return len(inter) / len(union)


def time_iou(a: Sequence[Tuple[float, int]], b: Sequence[Tuple[float, int]]) -> float:
    if not a or not b:
        return 0.0
    a0, a1 = occurrence_span(a)
    b0, b1 = occurrence_span(b)
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def best_overlap_pair(
    motif_a: Sequence[Sequence[Tuple[float, int]]],
    motif_b: Sequence[Sequence[Tuple[float, int]]],
) -> Tuple[float, float, Optional[Sequence[Tuple[float, int]]], Optional[Sequence[Tuple[float, int]]]]:
    best_note = -1.0
    best_time = -1.0
    best_a = None
    best_b = None
    for occ_a in motif_a:
        for occ_b in motif_b:
            n_j = note_jaccard(occ_a, occ_b)
            t_i = time_iou(occ_a, occ_b)
            if (n_j > best_note) or (abs(n_j - best_note) < 1e-12 and t_i > best_time):
                best_note = float(n_j)
                best_time = float(t_i)
                best_a = occ_a
                best_b = occ_b
    return best_note, best_time, best_a, best_b


def _pick_representative_occurrence(
    motif: Sequence[Sequence[Tuple[float, int]]],
) -> Optional[Sequence[Tuple[float, int]]]:
    # Pick the occurrence with the most notes (ties: shorter span).
    best_occ = None
    best_notes = -1
    best_span = None
    for occ in motif:
        span = occurrence_span(occ)
        n_notes = len(occ)
        if n_notes > best_notes:
            best_notes = n_notes
            best_occ = occ
            best_span = span
        elif n_notes == best_notes and best_span is not None:
            if (span[1] - span[0]) < (best_span[1] - best_span[0]):
                best_occ = occ
                best_span = span
    return best_occ


def _compute_note_multiplicity(
    notes: Sequence[Note],
    patterns: Sequence[Sequence[Sequence[Tuple[float, int]]]],
    motif_ids: Optional[Sequence[int]] = None,
    top_motifs: int = 0,
) -> Tuple[np.ndarray, int]:
    note_index = build_note_index(notes)
    counts = np.zeros(len(notes), dtype=np.int32)

    if motif_ids is not None and len(motif_ids):
        chosen = list(motif_ids)
    else:
        # Default to all motifs, optionally truncated to top-K by occurrence count.
        order = sorted(range(len(patterns)), key=lambda i: len(patterns[i]), reverse=True)
        if top_motifs and top_motifs > 0:
            order = order[: int(top_motifs)]
        chosen = order

    seen = 0
    for mid in chosen:
        if mid < 0 or mid >= len(patterns):
            continue
        for occ in patterns[mid]:
            for onset, pitch in occ:
                key = (_round_onset(onset), int(pitch))
                cand = note_index.get(key)
                if not cand:
                    continue
                counts[cand[0]] += 1
                seen += 1
    return counts, seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot piano-roll with motif highlights.")
    parser.add_argument("--piece", type=str, default=None, help="Piece id like 21-1 (optional if provided by audit).")
    parser.add_argument(
        "--csv-note-dir",
        type=str,
        default=None,
        help="Directory containing <piece>.csv note files.",
    )
    parser.add_argument(
        "--note-csv",
        type=str,
        default=None,
        help="Direct path to the note CSV file (overrides --csv-note-dir/--piece).",
    )
    parser.add_argument(
        "--patterns-json",
        type=str,
        default=None,
        help="Path to predicted motif JSON for a piece (from --save_predictions_dir).",
    )
    parser.add_argument(
        "--patterns-dir",
        type=str,
        default=None,
        help="Directory containing <piece>.json predictions (alternative to --patterns-json).",
    )
    parser.add_argument(
        "--motif-ids",
        type=str,
        default=None,
        help="Comma-separated motif ids to highlight (e.g., 12,34). If omitted, can use --from-audit.",
    )
    parser.add_argument(
        "--from-audit",
        type=str,
        default=None,
        help="Path to audit_summary.json or failure_cases.json to select motifs to highlight.",
    )
    parser.add_argument(
        "--audit-mode",
        type=str,
        default="near_dup",
        choices=["near_dup", "sparse"],
        help="When using --from-audit: select from near-duplicates or sparse motifs.",
    )
    parser.add_argument(
        "--audit-index",
        type=int,
        default=0,
        help="Index into the selected audit list (optionally filtered by --piece).",
    )
    parser.add_argument(
        "--show-gt",
        action="store_true",
        help="Highlight GT motif-labeled notes (non-empty 'type' column) in addition to predictions.",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional plot title override.")
    parser.add_argument(
        "--time-range",
        type=str,
        default=None,
        help="Optional x-range as 'start:end' (in beats).",
    )
    parser.add_argument(
        "--zoom",
        action="store_true",
        help="Auto-zoom x-range to the most relevant highlighted occurrence(s) (overrides --time-range).",
    )
    parser.add_argument(
        "--zoom-pad",
        type=float,
        default=2.0,
        help="Padding (in beats) for --zoom.",
    )
    parser.add_argument(
        "--zoom-context",
        type=float,
        default=0.0,
        help="When --zoom: expand bounds to include other occurrence spans within this distance (beats).",
    )
    parser.add_argument(
        "--only-highlighted",
        action="store_true",
        help="If set, do not draw all notes; only draw GT and highlighted motif notes/spans.",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="all",
        choices=["all", "best_pair", "none"],
        help="How to draw occurrence span bands for highlighted motifs.",
    )
    parser.add_argument(
        "--context-bands",
        type=str,
        default="none",
        choices=["none", "all"],
        help="When --bands best_pair: optionally also draw all other occurrence spans as faint context.",
    )
    parser.add_argument(
        "--highlight-notes",
        type=str,
        default="all",
        choices=["all", "bands"],
        help="Which notes to color for the highlighted motifs: all occurrences, or only the occurrence(s) used for span bands.",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="all_notes",
        choices=["all_notes", "multiplicity", "none"],
        help="Background rendering mode: full piano roll, note multiplicity heatmap, or none.",
    )
    parser.add_argument(
        "--multiplicity-top-motifs",
        type=int,
        default=0,
        help="When --background multiplicity and no --motif-ids: use only the top-K motifs by occurrence count (0=all).",
    )
    parser.add_argument(
        "--multiplicity-log",
        action="store_true",
        help="Use log1p scaling for multiplicity heatmap (recommended).",
    )
    parser.add_argument("--output", type=str, required=True, help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=200, help="PNG DPI.")
    parser.add_argument("--base-alpha", type=float, default=0.25, help="Alpha for background notes.")
    parser.add_argument("--span-alpha", type=float, default=0.08, help="Alpha for occurrence span bands.")
    parser.add_argument(
        "--overlap-color",
        type=str,
        default="#e377c2",
        help="Color for notes that are highlighted by 2+ motifs (default: magenta).",
    )
    args = parser.parse_args()

    if args.from_audit:
        audit_path = Path(args.from_audit)
        if args.audit_mode == "sparse":
            audit_piece, audit_ids, audit_meta = pick_sparse_motif_from_audit(
                audit_path, index=int(args.audit_index), piece=args.piece
            )
        else:
            audit_piece, audit_ids, audit_meta = pick_motif_ids_from_audit(
                audit_path, index=int(args.audit_index), piece=args.piece
            )
        piece = args.piece or audit_piece
        motif_ids = audit_ids
        meta = audit_meta
    else:
        piece = args.piece
        motif_ids = _parse_id_list(args.motif_ids)
        meta = ""

    if not piece:
        raise ValueError("--piece is required unless --from-audit provides it.")

    # Resolve note CSV path
    if args.note_csv:
        note_csv_path = Path(args.note_csv).expanduser()
    else:
        if not args.csv_note_dir:
            raise ValueError("Provide --note-csv or --csv-note-dir.")
        note_csv_path = Path(args.csv_note_dir).expanduser() / f"{piece}.csv"
    if not note_csv_path.exists():
        raise FileNotFoundError(f"Missing note CSV: {note_csv_path}")

    # Resolve patterns JSON path
    patterns_path = None
    if args.patterns_json:
        patterns_path = Path(args.patterns_json).expanduser()
    elif args.patterns_dir:
        patterns_path = Path(args.patterns_dir).expanduser() / f"{piece}.json"
    if patterns_path is not None and not patterns_path.exists():
        raise FileNotFoundError(f"Missing patterns JSON: {patterns_path}")

    notes = load_note_csv(note_csv_path)
    if not notes:
        raise ValueError(f"No notes loaded from {note_csv_path}")

    patterns: List[List[List[Tuple[float, int]]]] = []
    if patterns_path is not None:
        _piece_from_json, patterns = load_patterns_json(patterns_path)

    # Build time/pitch extents
    min_t = min(n.onset for n in notes)
    max_t = max(n.onset + n.duration for n in notes)
    min_p = min(n.pitch for n in notes)
    max_p = max(n.pitch for n in notes)

    # If we are rendering "best_pair" style views, compute the focus occurrence(s) once.
    focus_occ_by_motif: Dict[int, Sequence[Tuple[float, int]]] = {}
    if patterns and motif_ids:
        if len(motif_ids) >= 2:
            a_id, b_id = motif_ids[0], motif_ids[1]
            _bn, _bt, occ_a, occ_b = best_overlap_pair(patterns[a_id], patterns[b_id])
            if occ_a is not None:
                focus_occ_by_motif[a_id] = occ_a
            if occ_b is not None:
                focus_occ_by_motif[b_id] = occ_b
        else:
            mid = motif_ids[0]
            rep = _pick_representative_occurrence(patterns[mid])
            if rep is not None:
                focus_occ_by_motif[mid] = rep

    # Prepare highlight sets
    note_index = build_note_index(notes)
    highlight_by_motif: Dict[int, List[int]] = {}
    highlight_sets: Dict[int, set[int]] = {}
    spans_by_motif: Dict[int, List[Tuple[float, float]]] = {}
    rep_span_by_motif: Dict[int, Tuple[float, float]] = {}
    for motif_id in motif_ids:
        if motif_id < 0 or motif_id >= len(patterns):
            raise ValueError(f"motif id {motif_id} out of range for {patterns_path}")
        occs = patterns[motif_id]
        spans_by_motif[motif_id] = motif_occurrence_spans(occs)
        rep = _pick_representative_occurrence(occs)
        if rep is not None:
            rep_span_by_motif[motif_id] = occurrence_span(rep)
        idxs: List[int] = []
        if args.highlight_notes == "bands" and args.bands == "best_pair":
            focus = focus_occ_by_motif.get(motif_id)
            if focus is not None:
                for onset, pitch in focus:
                    key = (_round_onset(onset), int(pitch))
                    cand = note_index.get(key)
                    if not cand:
                        continue
                    idxs.append(cand[0])
        else:
            for occ in occs:
                for onset, pitch in occ:
                    key = (_round_onset(onset), int(pitch))
                    cand = note_index.get(key)
                    if not cand:
                        continue
                    idxs.append(cand[0])
        uniq = sorted(set(idxs))
        highlight_by_motif[motif_id] = uniq
        highlight_sets[motif_id] = set(uniq)

    gt_idxs = [i for i, n in enumerate(notes) if n.gt_type] if args.show_gt else []
    overlap_idxs: set[int] = set()
    if len(motif_ids) >= 2:
        counts: Dict[int, int] = {}
        for mid in motif_ids:
            for idx in highlight_sets.get(mid, set()):
                counts[idx] = counts.get(idx, 0) + 1
        overlap_idxs = {idx for idx, c in counts.items() if c >= 2}

    # Apply time-range crop if provided (or auto-zoom)
    if args.zoom and motif_ids and patterns:
        # Motifs often have occurrences across the whole piece. Zoom should focus on the
        # *specific* redundant pair of occurrences (audit-style), not min/max over all.
        focus_spans: List[Tuple[float, float]] = []
        if len(motif_ids) >= 2:
            a_id, b_id = motif_ids[0], motif_ids[1]
            occ_a = focus_occ_by_motif.get(a_id)
            occ_b = focus_occ_by_motif.get(b_id)
            if occ_a is not None and occ_b is not None:
                a0, a1 = occurrence_span(occ_a)
                b0, b1 = occurrence_span(occ_b)
                focus_spans.append((min(a0, b0), max(a1, b1)))
                if not meta:
                    best_note = note_jaccard(occ_a, occ_b)
                    best_time = time_iou(occ_a, occ_b)
                    meta = f"best_pair note_jaccard={best_note:.3f} time_iou={best_time:.3f}"
        else:
            mid = motif_ids[0]
            span = rep_span_by_motif.get(mid)
            if span is not None:
                focus_spans.append(span)

        if focus_spans:
            min_t = min(s[0] for s in focus_spans) - float(args.zoom_pad)
            max_t = max(s[1] for s in focus_spans) + float(args.zoom_pad)
            # Optionally expand to include nearby occurrence spans (useful to see adjacent repeats).
            ctx = float(args.zoom_context)
            if ctx > 0:
                for _ in range(3):
                    prev = (min_t, max_t)
                    for mid in motif_ids:
                        for s0, s1 in spans_by_motif.get(mid, []):
                            if s1 >= (min_t - ctx) and s0 <= (max_t + ctx):
                                min_t = min(min_t, s0 - float(args.zoom_pad))
                                max_t = max(max_t, s1 + float(args.zoom_pad))
                    if (min_t, max_t) == prev:
                        break
        else:
            # Fallback: zoom to all highlighted spans.
            span_points: List[float] = []
            for mid in motif_ids:
                for s0, s1 in spans_by_motif.get(mid, []):
                    span_points.extend([s0, s1])
            if span_points:
                min_t = min(span_points) - float(args.zoom_pad)
                max_t = max(span_points) + float(args.zoom_pad)
    elif args.time_range:
        parts = args.time_range.split(":")
        if len(parts) != 2:
            raise ValueError("--time-range must be 'start:end'")
        t0 = float(parts[0]) if parts[0].strip() else min_t
        t1 = float(parts[1]) if parts[1].strip() else max_t
        min_t, max_t = min(t0, t1), max(t0, t1)

    # Plot
    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Rectangle  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting.") from exc

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor("white")

    # Background notes (light)
    if not args.only_highlighted and args.background != "none":
        if args.background == "multiplicity":
            if not patterns:
                raise ValueError("--background multiplicity requires predictions (provide --patterns-dir/--patterns-json).")
            counts, _seen = _compute_note_multiplicity(
                notes,
                patterns,
                motif_ids=None,  # all/topK motifs, not just highlighted
                top_motifs=int(args.multiplicity_top_motifs),
            )
            if args.multiplicity_log:
                values = np.log1p(counts.astype(np.float32))
                label = "log1p(assignments)"
            else:
                values = counts.astype(np.float32)
                label = "assignments"
            vmax = float(values.max()) if values.size else 1.0
            norm = plt.Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
            cmap = plt.get_cmap("magma")
            for i, n in enumerate(notes):
                if n.onset + n.duration < min_t or n.onset > max_t:
                    continue
                v = float(values[i])
                if v <= 0.0:
                    face = "#e0e0e0"
                    alpha = float(args.base_alpha)
                else:
                    face = cmap(norm(v))
                    alpha = 0.75
                ax.add_patch(
                    Rectangle(
                        (n.onset, n.pitch - 0.4),
                        n.duration,
                        0.8,
                        facecolor=face,
                        edgecolor="none",
                        alpha=alpha,
                    )
                )
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label(label)
        else:
            for n in notes:
                if n.onset + n.duration < min_t or n.onset > max_t:
                    continue
                ax.add_patch(
                    Rectangle(
                        (n.onset, n.pitch - 0.4),
                        n.duration,
                        0.8,
                        facecolor="#7f7f7f",
                        edgecolor="none",
                        alpha=float(args.base_alpha),
                    )
                )

    # GT overlay
    if gt_idxs:
        for i in gt_idxs:
            n = notes[i]
            if n.onset + n.duration < min_t or n.onset > max_t:
                continue
            ax.add_patch(
                Rectangle(
                    (n.onset, n.pitch - 0.45),
                    n.duration,
                    0.9,
                    facecolor="#2ca02c",
                    edgecolor="none",
                    alpha=0.30,
                )
            )

    colors = [
        "#d62728",  # red
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
    ]

    # Predicted motif spans + notes
    for k, motif_id in enumerate(motif_ids):
        color = colors[k % len(colors)]
        if args.bands != "none":
            if args.bands == "best_pair":
                if args.context_bands == "all":
                    for (s0, s1) in spans_by_motif.get(motif_id, []):
                        if s1 < min_t or s0 > max_t:
                            continue
                        ax.add_patch(
                            Rectangle(
                                (s0, min_p - 1),
                                max(1e-6, s1 - s0),
                                (max_p - min_p) + 2,
                                facecolor=color,
                                edgecolor="none",
                                alpha=float(args.span_alpha) * 0.20,
                            )
                        )
                focus_span = rep_span_by_motif.get(motif_id)
                if len(motif_ids) >= 2 and motif_id in focus_occ_by_motif:
                    focus_span = occurrence_span(focus_occ_by_motif[motif_id])
                if focus_span:
                    s0, s1 = focus_span
                    if not (s1 < min_t or s0 > max_t):
                        ax.add_patch(
                            Rectangle(
                                (s0, min_p - 1),
                                max(1e-6, s1 - s0),
                                (max_p - min_p) + 2,
                                facecolor=color,
                                edgecolor="none",
                                alpha=float(args.span_alpha),
                            )
                        )
            else:
                for (s0, s1) in spans_by_motif.get(motif_id, []):
                    if s1 < min_t or s0 > max_t:
                        continue
                    ax.add_patch(
                        Rectangle(
                            (s0, min_p - 1),
                            max(1e-6, s1 - s0),
                            (max_p - min_p) + 2,
                            facecolor=color,
                            edgecolor="none",
                            alpha=float(args.span_alpha),
                        )
                    )
        for idx in highlight_by_motif.get(motif_id, []):
            if idx in overlap_idxs:
                continue
            n = notes[idx]
            if n.onset + n.duration < min_t or n.onset > max_t:
                continue
            ax.add_patch(
                Rectangle(
                    (n.onset, n.pitch - 0.48),
                    n.duration,
                    0.96,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.3,
                    alpha=0.85,
                )
            )

    # Overlap notes (draw last, on top)
    if overlap_idxs:
        for idx in sorted(overlap_idxs):
            n = notes[idx]
            if n.onset + n.duration < min_t or n.onset > max_t:
                continue
            ax.add_patch(
                Rectangle(
                    (n.onset, n.pitch - 0.5),
                    n.duration,
                    1.0,
                    facecolor=str(args.overlap_color),
                    edgecolor="black",
                    linewidth=0.4,
                    alpha=0.95,
                )
            )

    ax.set_xlim(min_t, max_t)
    ax.set_ylim(min_p - 2, max_p + 2)
    ax.set_xlabel("time (beats)")
    ax.set_ylabel("MIDI pitch")

    # Title/legend
    title = args.title
    if not title:
        title = f"{piece}"
        if motif_ids:
            title += f" | motifs {','.join(str(x) for x in motif_ids)}"
        if meta:
            title += f" | {meta}"
    ax.set_title(title)

    legend_items = []
    if gt_idxs:
        legend_items.append(("GT motif notes", "#2ca02c"))
    if overlap_idxs:
        legend_items.append(("Overlap (2+ motifs)", str(args.overlap_color)))
    for k, motif_id in enumerate(motif_ids):
        color = colors[k % len(colors)]
        occ_count = len(patterns[motif_id]) if patterns else 0
        suffix = ""
        if args.highlight_notes == "bands" and args.bands == "best_pair":
            suffix = ", focus"
        legend_items.append((f"Pred motif {motif_id} (occ {occ_count}{suffix})", color))
    if legend_items:
        handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none", alpha=0.8) for _t, c in legend_items]
        labels = [t for t, _c in legend_items]
        ax.legend(handles, labels, loc="upper right", framealpha=0.9)

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)


if __name__ == "__main__":
    main()
