#!/usr/bin/env python3
"""
Batch embedding computation for LR_V0 handcrafted features.
Requires per-piece segments JSON (from run_segments.py) and note CSVs.
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np

# repo imports
import sys

# Make motif_discovery imports available
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery" / "repeated_pattern_discovery"))

from motif_discovery.learned_retrieval.embeddings import embed_segments, default_config, _load_segments  # type: ignore
from motif_discovery.experiments import load_all_notes  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Batch compute segment embeddings.")
    parser.add_argument("--segments-dir", type=str, required=True, help="Directory with segment JSON files.")
    parser.add_argument("--csv-note-dir", type=str, required=True, help="Directory with note CSVs.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save embeddings (.npz per piece).")
    parser.add_argument(
        "--pieces",
        type=str,
        default=None,
        help="Optional comma-separated list of piece ids. Defaults to all JSONs in segments-dir.",
    )
    parser.add_argument(
        "--no-pitch-class",
        action="store_true",
        help="Disable pitch-class histogram.",
    )
    args = parser.parse_args()

    seg_dir = Path(args.segments_dir)
    note_dir = Path(args.csv_note_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pieces:
        pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
    else:
        pieces = [p.stem for p in sorted(seg_dir.glob("*.json"))]

    if not pieces:
        raise ValueError(f"No pieces found in {seg_dir}")

    cfg = default_config()
    if args.no_pitch_class:
        cfg.use_pitch_class = False

    total = 0
    for piece in pieces:
        seg_path = seg_dir / f"{piece}.json"
        note_path = note_dir / f"{piece}.csv"
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segments JSON: {seg_path}")
        if not note_path.exists():
            raise FileNotFoundError(f"Missing note CSV: {note_path}")

        segments = _load_segments(seg_path)
        notes = load_all_notes(str(note_path))
    embeddings, tempos, _ = embed_segments(notes, segments, cfg)

    out_path = out_dir / f"{piece}.npz"
    np.savez_compressed(out_path, embeddings=embeddings, tempos=tempos, segment_ids=[s.segment_id for s in segments])
        total += len(segments)
        print(f"{piece}: embeddings {embeddings.shape} -> {out_path}")

    print(f"Done. Pieces: {len(pieces)}, total segments: {total}, output: {out_dir}")


if __name__ == "__main__":
    main()
