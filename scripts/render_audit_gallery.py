#!/usr/bin/env python3
"""
Render a small "gallery" of audit-flagged cases (near-duplicates and sparse motifs).

Inputs:
  - predictions directory (per-piece JSON from motif_discovery/experiments.py --save_predictions_dir)
  - audit_summary.json or failure_cases.json from scripts/audit_predictions.py
  - note CSV directory (e.g., datasets/Beethoven_motif-main/csv_notes_clean)

Outputs:
  - PNGs per case (piano-roll with motif highlights)
  - index.html with thumbnails and metadata
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _iter_near_dup(payload: dict) -> List[dict]:
    if "top_near_duplicates" in payload:
        return list(payload.get("top_near_duplicates", []))
    if "redundant_clusters" in payload:
        return list(payload.get("redundant_clusters", []))
    return []


def _iter_sparse(payload: dict) -> List[dict]:
    if "top_sparse" in payload:
        return list(payload.get("top_sparse", []))
    if "sparse_motifs" in payload:
        return list(payload.get("sparse_motifs", []))
    return []


def _relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)

def _load_num_motifs(pred_path: Path) -> int:
    try:
        payload = json.loads(pred_path.read_text())
        return int(len(payload.get("patterns", []) or []))
    except Exception:
        return 0


def _write_html(out_dir: Path, rows: Sequence[Dict[str, str]]) -> None:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'/>",
        "<title>Motif audit gallery</title>",
        "<style>",
        "body{font-family:system-ui, -apple-system, sans-serif; margin:24px;}",
        "table{border-collapse:collapse; width:100%;}",
        "th,td{border-bottom:1px solid #ddd; padding:10px; vertical-align:top;}",
        "th{position:sticky; top:0; background:#fff; text-align:left;}",
        ".imgs{display:flex; flex-wrap:wrap; gap:10px;}",
        "img{max-width:420px; height:auto; border:1px solid #ddd;}",
        "code{background:#f6f6f6; padding:2px 4px; border-radius:4px;}",
        "</style></head><body>",
        "<h1>Motif audit gallery</h1>",
        "<p>Each row is a case rendered in multiple views (motif highlights, best-pair zoom, multiplicity heatmap).</p>",
        "<table>",
        "<thead><tr><th>Case</th><th>Info</th><th>Preview</th></tr></thead><tbody>",
    ]
    for r in rows:
        parts.append("<tr>")
        parts.append(f"<td>{html.escape(r['case'])}</td>")
        parts.append(f"<td><pre>{html.escape(r['info'])}</pre></td>")
        imgs = r["img"].split("|")
        parts.append("<td><div class='imgs'>")
        for img in imgs:
            img = img.strip()
            if not img:
                continue
            parts.append(f"<a href='{html.escape(img)}'><img src='{html.escape(img)}'/></a>")
        parts.append("</div></td>")
        parts.append("</tr>")
    parts.extend(["</tbody></table>", "</body></html>"])
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")

def _run_plot(cmd: List[str]) -> None:
    from subprocess import run as _run
    _run(cmd, check=True)

def _plot_script() -> str:
    return str(Path(__file__).parent / "plot_motif_pianoroll.py")

def _render_near_dup_views(
    *,
    piece: str,
    motif_ids: Sequence[int],
    meta: str,
    pred_json: Path,
    note_csv: Path,
    csv_note_dir: Path,
    patterns_dir: Path,
    out_dir: Path,
    idx: int,
    show_gt: bool,
    zoom_pad: float,
    multiplicity_top_motifs: int,
) -> Tuple[str, str]:
    imgs: List[Path] = []

    base_args = [
        sys.executable,
        _plot_script(),
        "--piece",
        piece,
        "--csv-note-dir",
        str(csv_note_dir),
        "--patterns-dir",
        str(patterns_dir),
        "--motif-ids",
        ",".join(str(x) for x in motif_ids),
        "--zoom",
        "--zoom-pad",
        str(float(zoom_pad)),
        "--zoom-context",
        "0.5",
    ]
    if show_gt:
        base_args.append("--show-gt")

    # View 1: best-pair span bands (clean redundancy view)
    out_bestpair = out_dir / f"near_dup_{idx:02d}_{piece}_m{motif_ids[0]}_{motif_ids[1]}_bestpair.png"
    _run_plot(base_args + ["--bands", "best_pair", "--context-bands", "all", "--output", str(out_bestpair)])
    imgs.append(out_bestpair)

    # View 2: notes-only (no span bands)
    out_notesonly = out_dir / f"near_dup_{idx:02d}_{piece}_m{motif_ids[0]}_{motif_ids[1]}_notesonly.png"
    _run_plot(base_args + ["--bands", "none", "--only-highlighted", "--output", str(out_notesonly)])
    imgs.append(out_notesonly)

    # View 3: multiplicity heatmap in the same zoom window
    out_mult = out_dir / f"near_dup_{idx:02d}_{piece}_m{motif_ids[0]}_{motif_ids[1]}_multiplicity.png"
    _run_plot(
        base_args
        + [
            "--bands",
            "none",
            "--background",
            "multiplicity",
            "--multiplicity-log",
            "--multiplicity-top-motifs",
            str(int(multiplicity_top_motifs)),
            "--output",
            str(out_mult),
        ]
    )
    imgs.append(out_mult)

    info = (
        meta
        + f"\npatterns: {pred_json}"
        + f"\nnotes: {note_csv}"
        + f"\nviews: bestpair | notesonly | multiplicity"
    )
    img_field = "|".join(_relpath(p, out_dir) for p in imgs)
    return info, img_field


def _render_sparse_views(
    *,
    piece: str,
    motif_id: int,
    meta: str,
    pred_json: Path,
    note_csv: Path,
    csv_note_dir: Path,
    patterns_dir: Path,
    out_dir: Path,
    idx: int,
    show_gt: bool,
    zoom_pad: float,
    multiplicity_top_motifs: int,
) -> Tuple[str, str]:
    imgs: List[Path] = []
    base_args = [
        sys.executable,
        _plot_script(),
        "--piece",
        piece,
        "--csv-note-dir",
        str(csv_note_dir),
        "--patterns-dir",
        str(patterns_dir),
        "--motif-ids",
        str(int(motif_id)),
        "--zoom",
        "--zoom-pad",
        str(float(zoom_pad)),
        "--zoom-context",
        "0.5",
    ]
    if show_gt:
        base_args.append("--show-gt")

    out_zoom = out_dir / f"sparse_{idx:02d}_{piece}_m{motif_id}_zoom.png"
    _run_plot(base_args + ["--bands", "best_pair", "--context-bands", "all", "--output", str(out_zoom)])
    imgs.append(out_zoom)

    out_mult = out_dir / f"sparse_{idx:02d}_{piece}_m{motif_id}_multiplicity.png"
    _run_plot(
        base_args
        + [
            "--bands",
            "none",
            "--background",
            "multiplicity",
            "--multiplicity-log",
            "--multiplicity-top-motifs",
            str(int(multiplicity_top_motifs)),
            "--output",
            str(out_mult),
        ]
    )
    imgs.append(out_mult)

    info = meta + f"\npatterns: {pred_json}\nnotes: {note_csv}\nviews: zoom | multiplicity"
    img_field = "|".join(_relpath(p, out_dir) for p in imgs)
    return info, img_field


def _render_explosion_piece(
    *,
    piece: str,
    num_motifs: int,
    pred_json: Path,
    note_csv: Path,
    csv_note_dir: Path,
    patterns_dir: Path,
    out_dir: Path,
    rank: int,
    show_gt: bool,
    multiplicity_top_motifs: int,
) -> Tuple[str, str]:
    imgs: List[Path] = []
    base_args = [
        sys.executable,
        _plot_script(),
        "--piece",
        piece,
        "--csv-note-dir",
        str(csv_note_dir),
        "--patterns-dir",
        str(patterns_dir),
        "--bands",
        "none",
        "--background",
        "multiplicity",
        "--multiplicity-log",
        "--multiplicity-top-motifs",
        str(int(multiplicity_top_motifs)),
    ]
    if show_gt:
        base_args.append("--show-gt")

    out_mult_full = out_dir / f"explosion_{rank:02d}_{piece}_motifs{num_motifs}_multiplicity.png"
    _run_plot(base_args + ["--output", str(out_mult_full)])
    imgs.append(out_mult_full)

    info = f"motifs={num_motifs}\npatterns: {pred_json}\nnotes: {note_csv}\nview: multiplicity(full)"
    img_field = "|".join(_relpath(p, out_dir) for p in imgs)
    return info, img_field


def main() -> None:
    parser = argparse.ArgumentParser(description="Render audit-flagged cases to a gallery.")
    parser.add_argument("--audit", type=str, required=True, help="Path to audit_summary.json or failure_cases.json.")
    parser.add_argument("--patterns-dir", type=str, required=True, help="Predictions dir containing <piece>.json.")
    parser.add_argument("--csv-note-dir", type=str, required=True, help="Directory containing <piece>.csv note files.")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for PNGs and index.html.")
    parser.add_argument("--top-k", type=int, default=10, help="How many near-dup and sparse cases to render.")
    parser.add_argument("--top-k-explosion", type=int, default=8, help="How many most-exploded pieces to render.")
    parser.add_argument("--zoom-pad", type=float, default=2.0, help="Padding (beats) around motif span when zooming.")
    parser.add_argument("--show-gt", action="store_true", help="Overlay GT motif-labeled notes from note CSV.")
    parser.add_argument(
        "--multiplicity-top-motifs",
        type=int,
        default=200,
        help="For multiplicity heatmap: use only top-K motifs by occurrence count (0=all).",
    )
    args = parser.parse_args()

    audit_path = Path(args.audit).expanduser()
    patterns_dir = Path(args.patterns_dir).expanduser()
    csv_note_dir = Path(args.csv_note_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_json(audit_path)
    near_dups = _iter_near_dup(payload)[: int(args.top_k)]
    sparse = _iter_sparse(payload)[: int(args.top_k)]

    rows: List[Dict[str, str]] = []

    # Explosion: top pieces by motif count.
    explosion_items: List[Tuple[int, str]] = []
    for pred_path in sorted(patterns_dir.glob("*.json")):
        piece = pred_path.stem
        explosion_items.append((_load_num_motifs(pred_path), piece))
    explosion_items.sort(reverse=True)
    explosion_items = explosion_items[: int(args.top_k_explosion)]

    for rank, (num_motifs, piece) in enumerate(explosion_items):
        note_csv = csv_note_dir / f"{piece}.csv"
        pred_json = patterns_dir / f"{piece}.json"
        if not note_csv.exists() or not pred_json.exists():
            continue
        info, img_field = _render_explosion_piece(
            piece=piece,
            num_motifs=int(num_motifs),
            pred_json=pred_json,
            note_csv=note_csv,
            csv_note_dir=csv_note_dir,
            patterns_dir=patterns_dir,
            out_dir=out_dir,
            rank=rank,
            show_gt=bool(args.show_gt),
            multiplicity_top_motifs=int(args.multiplicity_top_motifs),
        )
        rows.append({"case": f"explosion[{rank}] {piece}", "info": info, "img": img_field})

    # Near-duplicates
    for idx in range(len(near_dups)):
        entry = near_dups[idx]
        piece = str(entry.get("piece"))
        motif_ids = [int(entry.get("motif_a")), int(entry.get("motif_b"))]
        meta = f"note_jaccard={entry.get('note_jaccard')} time_iou={entry.get('time_iou')}"
        note_csv = csv_note_dir / f"{piece}.csv"
        pred_json = patterns_dir / f"{piece}.json"
        if not note_csv.exists() or not pred_json.exists():
            continue
        info, img_field = _render_near_dup_views(
            piece=piece,
            motif_ids=motif_ids,
            meta=meta,
            pred_json=pred_json,
            note_csv=note_csv,
            csv_note_dir=csv_note_dir,
            patterns_dir=patterns_dir,
            out_dir=out_dir,
            idx=idx,
            show_gt=bool(args.show_gt),
            zoom_pad=float(args.zoom_pad),
            multiplicity_top_motifs=int(args.multiplicity_top_motifs),
        )
        rows.append(
            {
                "case": f"near_dup[{idx}] {piece}",
                "info": info,
                "img": img_field,
            }
        )

    # Sparse motifs
    for idx in range(len(sparse)):
        entry = sparse[idx]
        piece = str(entry.get("piece"))
        motif_id = int(entry.get("motif"))
        meta = f"span_per_note={entry.get('span_per_note', entry.get('ratio'))}"
        note_csv = csv_note_dir / f"{piece}.csv"
        pred_json = patterns_dir / f"{piece}.json"
        if not note_csv.exists() or not pred_json.exists():
            continue
        info, img_field = _render_sparse_views(
            piece=piece,
            motif_id=motif_id,
            meta=meta,
            pred_json=pred_json,
            note_csv=note_csv,
            csv_note_dir=csv_note_dir,
            patterns_dir=patterns_dir,
            out_dir=out_dir,
            idx=idx,
            show_gt=bool(args.show_gt),
            zoom_pad=float(args.zoom_pad),
            multiplicity_top_motifs=int(args.multiplicity_top_motifs),
        )
        rows.append(
            {
                "case": f"sparse[{idx}] {piece}",
                "info": info,
                "img": img_field,
            }
        )

    _write_html(out_dir, rows)
    print(f"Wrote {len(rows)} images + {out_dir/'index.html'}")


if __name__ == "__main__":
    main()
