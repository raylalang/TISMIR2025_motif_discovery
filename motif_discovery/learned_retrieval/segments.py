"""
Multi-scale segment proposals for learned motif discovery (v0).

Rules:
- Fixed list of window lengths (seconds or beats, depending on input timeline).
- Hop per scale = length * hop_ratio (default 0.25 -> L/4).
- Include notes whose onset is in [start, end).
- Drop segments with fewer than min_notes.
- Each segment carries piece_id, start, end, scale_id, note_indices, segment_id.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

# Structured arrays from motif_discovery.experiments.load_all_notes have an "onset" field.
NOTE_ONSET_KEY = "onset"


@dataclass
class Segment:
    piece_id: str
    start: float
    end: float
    scale_id: int
    note_indices: List[int]
    segment_id: str

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "piece_id": self.piece_id,
            "scale_id": self.scale_id,
            "start": float(self.start),
            "end": float(self.end),
            "note_indices": list(self.note_indices),
        }


def _iter_windows(
    start_time: float, end_time: float, length: float, hop_ratio: float
) -> Iterable[float]:
    hop = length * hop_ratio
    if hop <= 0:
        raise ValueError(f"Invalid hop_ratio {hop_ratio}; must yield positive hop.")
    cur = start_time
    # Include last window that starts before or at the last note onset.
    while cur <= end_time:
        yield cur
        cur += hop


def propose_segments(
    notes: np.ndarray,
    piece_id: str,
    scale_lengths: Sequence[float],
    hop_ratio: float = 0.25,
    min_notes: int = 1,
    time_values: np.ndarray | None = None,
) -> List[Segment]:
    if notes.size == 0:
        return []
    if time_values is not None:
        if len(time_values) != len(notes):
            raise ValueError("time_values must match notes length.")
        onsets = np.asarray(time_values, dtype=np.float32)
    else:
        onsets = notes[NOTE_ONSET_KEY]
    min_onset = float(onsets.min())
    max_onset = float(onsets.max())

    segments: List[Segment] = []
    seg_counter = 0

    for scale_idx, length in enumerate(scale_lengths):
        if length <= 0:
            raise ValueError(f"Scale length must be positive; got {length}")
        for start in _iter_windows(min_onset, max_onset, length, hop_ratio):
            end = start + length
            idx_mask = (onsets >= start) & (onsets < end)
            note_idxs = np.nonzero(idx_mask)[0]
            if len(note_idxs) < min_notes:
                continue
            seg_id = f"{piece_id}_s{scale_idx}_{seg_counter}"
            seg_counter += 1
            segments.append(
                Segment(
                    piece_id=piece_id,
                    start=start,
                    end=end,
                    scale_id=scale_idx,
                    note_indices=note_idxs.tolist(),
                    segment_id=seg_id,
                )
            )
    return segments


def note_onsets_to_beats(onsets: np.ndarray, midi_path: str) -> np.ndarray:
    import pretty_midi

    if onsets.size == 0:
        return np.array([], dtype=np.float32)
    midi = pretty_midi.PrettyMIDI(midi_path)
    beat_times = np.asarray(midi.get_beats(), dtype=np.float32)
    if beat_times.size < 2:
        raise ValueError(f"Need at least two beats to map time to beats: {midi_path}")
    intervals = np.diff(beat_times)
    intervals = np.clip(intervals, 1e-6, None)
    idx = np.searchsorted(beat_times, onsets, side="right") - 1
    beat_pos = np.empty_like(onsets, dtype=np.float32)
    before = idx < 0
    after = idx >= intervals.size
    mid = ~(before | after)
    if np.any(before):
        beat_pos[before] = (onsets[before] - beat_times[0]) / intervals[0]
    if np.any(after):
        beat_pos[after] = (intervals.size) + (onsets[after] - beat_times[-1]) / intervals[-1]
    if np.any(mid):
        mid_idx = idx[mid]
        beat_pos[mid] = mid_idx + (onsets[mid] - beat_times[mid_idx]) / intervals[mid_idx]
    return beat_pos


def _parse_scale_list(raw: str) -> List[float]:
    if not raw:
        return []
    return [float(x) for x in raw.split(",")]


def _load_notes(path: Path) -> np.ndarray:
    # Lazy import to avoid adding dependencies for callers that already have notes.
    from motif_discovery.experiments import load_all_notes  # type: ignore

    return load_all_notes(str(path))


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for multi-scale segment proposals (LR_V0)."
    )
    parser.add_argument("--piece", type=str, required=True, help="Piece id like 21-1")
    parser.add_argument(
        "--csv-note-dir",
        type=str,
        required=True,
        help="Directory containing note CSVs (e.g., MNID outputs).",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="2.0,4.0,8.0,16.0",
        help="Comma-separated window lengths (same units as note onsets). Default tuned from BPS spans: 2,4,8,16",
    )
    parser.add_argument(
        "--hop-ratio",
        type=float,
        default=0.25,
        help="Hop as a fraction of window length (default 0.25 => L/4).",
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=3,
        help="Drop segments with fewer than this many notes.",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="If set, print segments as JSON instead of counts only.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save segments as JSON. Creates parent dirs if needed.",
    )
    args = parser.parse_args()

    scales = _parse_scale_list(args.scales)
    if not scales:
        raise ValueError("At least one scale length is required.")

    note_path = Path(args.csv_note_dir) / f"{args.piece}.csv"
    notes = _load_notes(note_path)
    segments = propose_segments(
        notes=notes,
        piece_id=args.piece,
        scale_lengths=scales,
        hop_ratio=args.hop_ratio,
        min_notes=args.min_notes,
    )

    per_scale = {}
    for s in segments:
        per_scale.setdefault(s.scale_id, 0)
        per_scale[s.scale_id] += 1

    print(f"Piece: {args.piece}")
    print(f"Total segments: {len(segments)}")
    for scale_id in sorted(per_scale):
        print(f"  scale {scale_id} ({scales[scale_id]}): {per_scale[scale_id]}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "piece": args.piece,
            "scales": scales,
            "hop_ratio": args.hop_ratio,
            "min_notes": args.min_notes,
            "segments": [s.to_dict() for s in segments],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved segments to {out_path}")

    if args.dump_json:
        print(json.dumps([s.to_dict() for s in segments], indent=2))


if __name__ == "__main__":
    main()
