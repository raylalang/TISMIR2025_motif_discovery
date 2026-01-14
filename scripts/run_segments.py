#!/usr/bin/env python3
"""
Batch segment proposal over a note CSV directory.

Outputs one JSON per piece with segment metadata.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List
from tqdm import tqdm

# Make motif_discovery imports available
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))
sys.path.insert(0, str(REPO_ROOT / "motif_discovery" / "repeated_pattern_discovery"))

from motif_discovery.learned_retrieval.segments import propose_segments  # type: ignore
from motif_discovery.experiments import load_all_notes  # type: ignore


def parse_scales(raw: str) -> List[float]:
    vals = [float(x) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one scale length is required.")
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segment proposals over a dataset.")
    parser.add_argument("--csv-note-dir", type=str, required=True, help="Directory with note CSV files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save per-piece segment JSON.",
    )
    parser.add_argument(
        "--pieces",
        type=str,
        default=None,
        help="Optional comma-separated list of piece ids (stems without .csv). Defaults to all in csv-note-dir.",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="2.0,4.0,8.0,16.0",
        help="Comma-separated window lengths (same units as note onsets).",
    )
    parser.add_argument(
        "--hop-ratio",
        type=float,
        default=0.25,
        help="Hop as a fraction of window length.",
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=3,
        help="Drop segments with fewer than this many notes.",
    )
    args = parser.parse_args()

    scales = parse_scales(args.scales)
    csv_dir = Path(args.csv_note_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.pieces:
        pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
    else:
        pieces = [p.stem for p in sorted(csv_dir.glob("*.csv"))]

    if not pieces:
        raise ValueError(f"No pieces found in {csv_dir}")

    total_segments = 0
    for piece in tqdm(pieces, desc="Processing pieces"):
        csv_path = csv_dir / f"{piece}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing note CSV: {csv_path}")

        notes = load_all_notes(str(csv_path))
        segments = propose_segments(
            notes=notes,
            piece_id=piece,
            scale_lengths=scales,
            hop_ratio=args.hop_ratio,
            min_notes=args.min_notes,
        )

        payload = {
            "piece": piece,
            "scales": scales,
            "hop_ratio": args.hop_ratio,
            "min_notes": args.min_notes,
            "num_segments": len(segments),
            "segments": [s.to_dict() for s in segments],
        }
        out_path = out_dir / f"{piece}.json"
        out_path.write_text(json.dumps(payload, indent=2))
        total_segments += len(segments)
        print(f"{piece}: segments={len(segments)} -> {out_path}")

    print(f"Done. Pieces: {len(pieces)}, total segments: {total_segments}, output: {out_dir}")


if __name__ == "__main__":
    main()
