#!/usr/bin/env python3
"""
Build oracle note CSVs by extracting GT motif notes from label+MIDI.

Outputs CSVs with the same columns as csv_notes_clean so they are drop-in
for motif discovery input.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Make motif_discovery imports available when running from repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery" / "repeated_pattern_discovery"))

from motif_discovery.experiments import load_all_motives  # type: ignore


def iter_motif_notes(motives: Dict) -> Iterable[Tuple[float, int, float]]:
    for occs in motives.values():
        for occ in occs:
            if occ.size == 0:
                continue
            for note in occ:
                onset = float(note["onset"])
                end = float(note["end"])
                pitch = int(note["pitch"])
                duration = float(end - onset)
                yield onset, pitch, duration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract oracle note CSVs from GT motif labels + motif MIDI."
    )
    parser.add_argument(
        "--csv-label-dir",
        type=str,
        default=str(REPO_ROOT / "datasets/Beethoven_motif-main/csv_label"),
        help="Directory with motif label CSVs.",
    )
    parser.add_argument(
        "--motif-midi-dir",
        type=str,
        default=str(REPO_ROOT / "datasets/Beethoven_motif-main/motif_midi"),
        help="Directory with motif MIDI files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "datasets/bps_oracle_notes"),
        help="Output directory for oracle note CSVs.",
    )
    parser.add_argument(
        "--pieces",
        type=str,
        default=None,
        help="Optional comma-separated list of piece ids (e.g., 01-1,02-1).",
    )
    args = parser.parse_args()

    label_dir = Path(args.csv_label_dir)
    midi_dir = Path(args.motif_midi_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pieces:
        pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
    else:
        pieces = [p.stem for p in sorted(label_dir.glob("*.csv"))]

    if not pieces:
        raise ValueError(f"No pieces found in {label_dir}")

    header = [
        "onset",
        "midi_number",
        "morphetic_number",
        "duration",
        "staff_number",
        "measure",
        "type",
    ]

    for piece in pieces:
        label_csv = label_dir / f"{piece}.csv"
        midi_path = midi_dir / f"{piece}.mid"
        if not label_csv.exists():
            raise FileNotFoundError(f"Missing label CSV: {label_csv}")
        if not midi_path.exists():
            raise FileNotFoundError(f"Missing motif MIDI: {midi_path}")

        motives = load_all_motives(str(label_csv), str(midi_path))
        note_rows = list(iter_motif_notes(motives))
        if not note_rows:
            print(f"{piece}: no motif notes found, skipping.")
            continue

        # De-duplicate by onset+pitch+duration.
        unique = {}
        for onset, pitch, duration in note_rows:
            key = (onset, pitch, duration)
            unique[key] = (onset, pitch, duration)

        rows = list(unique.values())
        rows.sort(key=lambda r: (r[0], r[1]))

        out_path = out_dir / f"{piece}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for onset, pitch, duration in rows:
                writer.writerow(
                    [
                        f"{onset:.6f}",
                        pitch,
                        "",
                        f"{duration:.6f}",
                        0,
                        0,
                        "oracle",
                    ]
                )
        print(f"{piece}: wrote {len(rows)} notes -> {out_path}")


if __name__ == "__main__":
    main()
