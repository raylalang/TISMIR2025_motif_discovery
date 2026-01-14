#!/usr/bin/env python3
"""
Unified runner:
- Baseline mode: MNID inference -> motif discovery (CSA/SIATEC/SIATEC_CS/LR_V0).
- Motif-only mode: skip MNID, run motif discovery directly (e.g., LR_V0).
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_BASE_CONFIG = "configs/csa_mnid_bps.yaml"
DEFAULT_LR_CONFIG = "configs/lr_v0_nomnid_bps.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNID + motif discovery, or motif-only.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=f"Path to YAML/JSON config (default: {DEFAULT_BASE_CONFIG} for baseline, {DEFAULT_LR_CONFIG} for motif-only).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional override for output run directory.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Optional override for motif.method (CSA, SIATEC, SIATEC_CS, LR_V0).",
    )
    parser.add_argument(
        "--motif-only",
        action="store_true",
        help="Skip MNID step; use provided csv_note_dir.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    text = config_path.read_text()
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    else:
        loaded = yaml.safe_load(text)
        return loaded if loaded is not None else {}


def deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def pathify(path_str: Optional[str], root: Path) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else (root / path).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


def run_subprocess(
    cmd: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    step_name: str,
) -> str:
    started = datetime.now().isoformat()
    header = f"\n===== {step_name} | {started} =====\ncommand: {' '.join(cmd)}\n"
    with log_path.open("a") as lf:
        lf.write(header)
    print(header.strip())

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = proc.stdout or ""
    with log_path.open("a") as lf:
        lf.write(output)
        lf.write("\n")

    if proc.returncode != 0:
        raise RuntimeError(
            f"{step_name} failed with code {proc.returncode}. See log at {log_path}."
        )
    return output


def parse_metrics(log_text: str) -> Dict[str, Dict[str, float]]:
    metrics = {}
    patterns = {
        "establishment": re.compile(
            r"Mean_est P ([0-9.]+), R ([0-9.]+), F ([0-9.]+)"
        ),
        "occurrence": re.compile(
            r"Mean_occ P ([0-9.]+), R ([0-9.]+), F ([0-9.]+)"
        ),
        "three_layer": re.compile(
            r"Mean_thr P ([0-9.]+), R ([0-9.]+), F ([0-9.]+)"
        ),
    }
    for name, regex in patterns.items():
        match = regex.search(log_text)
        if match:
            p, r, f1 = match.groups()
            metrics[name] = {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
            }
    return metrics


def build_config(
    raw: Dict[str, Any], repo_root: Path, config_path: Path, run_dir_override: Optional[str]
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "run_name": config_path.stem,
        "seed": 17,
        "output_root": "runs",
    }
    cfg = deep_update(defaults, deepcopy(raw))

    output_root = pathify(cfg.get("output_root"), repo_root) or (repo_root / "runs")
    ensure_dir(output_root)

    run_name = cfg.get("run_name") or config_path.stem
    if run_dir_override:
        run_dir = pathify(run_dir_override, repo_root)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_root / f"{run_name}-{timestamp}"
    ensure_dir(run_dir)

    cfg["output_root"] = output_root
    cfg["run_dir"] = run_dir
    return cfg


def resolve_paths(cfg: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    # Motif config
    motif_cfg = cfg.get("motif", {})
    # Allow legacy top-level fields for motif-only configs
    for k in ("csv_note_dir", "csv_label_dir", "motif_midi_dir", "save_predictions"):
        if k not in motif_cfg and k in cfg:
            motif_cfg[k] = cfg[k]
    cfg["motif"] = motif_cfg
    return cfg


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    # Choose default config if not provided
    cfg_path = Path(
        args.config
        or (DEFAULT_LR_CONFIG if args.motif_only else DEFAULT_BASE_CONFIG)
    ).expanduser().resolve()

    raw_config = load_config(cfg_path)
    if args.method:
        raw_config.setdefault("motif", {})["method"] = args.method

    cfg = build_config(raw_config, repo_root, cfg_path, args.run_dir)
    cfg = resolve_paths(cfg, repo_root)

    run_dir: Path = cfg["run_dir"]
    log_path = run_dir / "stdout.log"

    set_global_seeds(int(cfg.get("seed", 0)))

    motif_cfg = cfg["motif"]
    method = motif_cfg.get("method", "LR_V0")

    # Determine if MNID is available/desired
    mnid_cfg = cfg.get("mnid")
    mnid_skip = isinstance(mnid_cfg, dict) and mnid_cfg.get("skip", False)
    motif_only = args.motif_only or (not isinstance(mnid_cfg, dict)) or mnid_skip
    run_mnid = (not motif_only) and isinstance(mnid_cfg, dict)

    # MNID inference (optional)
    note_dir = pathify(motif_cfg.get("csv_note_dir"), repo_root)
    if run_mnid:
        mnid_cfg = deepcopy(mnid_cfg)
        mnid_cfg["input_dir"] = pathify(mnid_cfg.get("input_dir"), repo_root)
        mnid_cfg["bps_fh_dir"] = pathify(mnid_cfg.get("bps_fh_dir"), repo_root)
        mnid_cfg["dict_file"] = pathify(mnid_cfg.get("dict_file"), repo_root)
        mnid_cfg["ckpt_dir"] = pathify(mnid_cfg.get("ckpt_dir"), repo_root)
        if mnid_cfg.get("output_csv_dir"):
            mnid_cfg["output_csv_dir"] = pathify(mnid_cfg["output_csv_dir"], repo_root)
        else:
            mnid_cfg["output_csv_dir"] = run_dir / "predictions"
        ensure_dir(mnid_cfg["output_csv_dir"])
        note_dir = mnid_cfg["output_csv_dir"]

        mnid_env = os.environ.copy()
        mnid_env["PYTHONHASHSEED"] = str(cfg.get("seed", 0))
        mnid_env["PYTHONPATH"] = "." + (
            os.pathsep + mnid_env["PYTHONPATH"] if mnid_env.get("PYTHONPATH") else ""
        )

        mnid_cmd = [
            sys.executable,
            str(repo_root / "MNID" / "extract_csv_ft.py"),
            "--input_dir",
            str(mnid_cfg["input_dir"]),
            "--bps_fh_dir",
            str(mnid_cfg["bps_fh_dir"]),
            "--output_csv_dir",
            str(mnid_cfg["output_csv_dir"]),
            "--dict_file",
            str(mnid_cfg["dict_file"]),
            "--ckpt_dir",
            str(mnid_cfg["ckpt_dir"]),
        ]
        if mnid_cfg.get("cpu", False):
            mnid_cmd.append("--cpu")

        run_subprocess(
            mnid_cmd,
            cwd=repo_root / "MNID",
            env=mnid_env,
            log_path=log_path,
            step_name="MNID inference",
        )
    else:
        if isinstance(mnid_cfg, dict) and mnid_cfg.get("output_csv_dir"):
            note_dir = pathify(mnid_cfg.get("output_csv_dir"), repo_root)
        if note_dir is None:
            raise ValueError("csv_note_dir must be provided for motif-only mode or when mnid is null/skip.")

    # Motif discovery + evaluation
    motif_env = os.environ.copy()
    motif_env["PYTHONHASHSEED"] = str(cfg.get("seed", 0))

    csv_label_dir = pathify(motif_cfg.get("csv_label_dir"), repo_root)
    motif_midi_dir = pathify(motif_cfg.get("motif_midi_dir"), repo_root)
    save_predictions = bool(motif_cfg.get("save_predictions", False))
    save_predictions_dir = (
        run_dir / "motifs"
        if save_predictions
        else None
    )
    if save_predictions_dir:
        ensure_dir(save_predictions_dir)

    motif_cmd = [
        sys.executable,
        str(repo_root / "motif_discovery" / "experiments.py"),
        "--csv_note_dir",
        str(note_dir),
        "--method",
        str(method),
    ]
    if csv_label_dir:
        motif_cmd.extend(["--csv_label_dir", str(csv_label_dir)])
    if motif_midi_dir:
        motif_cmd.extend(["--motif_midi_dir", str(motif_midi_dir)])
    if save_predictions_dir:
        motif_cmd.extend(["--save_predictions_dir", str(save_predictions_dir)])
    lr_tmp_path = None
    if method in ("LR_V0", "LR_V1"):
        if motif_cfg.get("lr_config"):
            lr_cfg_path = pathify(motif_cfg.get("lr_config"), repo_root)
            if lr_cfg_path:
                motif_cmd.extend(["--lr_config", str(lr_cfg_path)])
        elif motif_cfg.get("lr_params"):
            lr_tmp_path = run_dir / "lr_params_resolved.json"
            ensure_dir(lr_tmp_path.parent)
            with lr_tmp_path.open("w") as f:
                json.dump(motif_cfg.get("lr_params"), f, indent=2)
            motif_cmd.extend(["--lr_config", str(lr_tmp_path)])
        if motif_cfg.get("num_workers"):
            motif_cmd.extend(["--num_workers", str(int(motif_cfg.get("num_workers")))])

    motif_log = run_subprocess(
        motif_cmd,
        cwd=repo_root / "motif_discovery",
        env=motif_env,
        log_path=log_path,
        step_name="Motif discovery + evaluation",
    )

    metrics = parse_metrics(motif_log)
    if not metrics:
        raise RuntimeError(
            f"Could not parse metrics from motif discovery output. Check {log_path}."
        )

    metrics_path = run_dir / "metrics.json"
    metrics_payload = {
        "method": method,
        "csv_note_dir": str(note_dir),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    # Save resolved config for reproducibility
    resolved_path = run_dir / "config_resolved.yaml"
    cfg_serializable = deepcopy(cfg)
    cfg_serializable["run_dir"] = str(cfg_serializable["run_dir"])
    cfg_serializable["output_root"] = str(cfg_serializable["output_root"])
    if "mnid" in cfg_serializable and isinstance(cfg_serializable["mnid"], dict):
        for key in ("input_dir", "bps_fh_dir", "dict_file", "ckpt_dir", "output_csv_dir"):
            if key in cfg_serializable["mnid"]:
                cfg_serializable["mnid"][key] = str(cfg_serializable["mnid"][key])
    if "motif" in cfg_serializable and isinstance(cfg_serializable["motif"], dict):
        for key in ("csv_label_dir", "motif_midi_dir", "csv_note_dir"):
            if key in cfg_serializable["motif"]:
                cfg_serializable["motif"][key] = str(cfg_serializable["motif"][key])
        if save_predictions_dir:
            cfg_serializable["motif"]["save_predictions_dir"] = str(save_predictions_dir)
    resolved_path.write_text(json.dumps(cfg_serializable, indent=2) + "\n")

    summary_lines = [
        "Pipeline run complete.",
        f"metrics.json: {metrics_path}",
        f"predictions:  {note_dir}",
    ]
    if save_predictions_dir:
        summary_lines.append(f"motifs:       {save_predictions_dir}")
    summary_lines.extend(
        [
            f"stdout log:   {log_path}",
            f"run dir:      {run_dir}",
        ]
    )
    summary = "\n".join(summary_lines)
    print(summary)
    with log_path.open("a") as lf:
        lf.write("\n" + summary + "\n")

    # Cleanup temp lr config if created
    if lr_tmp_path and lr_tmp_path.exists():
        try:
            lr_tmp_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
