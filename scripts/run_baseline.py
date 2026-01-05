#!/usr/bin/env python3
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

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # pytype: disable=annotation-type-mismatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline pipeline: MNID inference -> motif discovery evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML/JSON config describing data/ckpt paths.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Optional override for output run directory.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    text = config_path.read_text()
    # JSON is a valid subset of YAML, so try JSON first for zero-deps.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise ValueError(
                f"Could not parse config at {config_path}. Install PyYAML or use JSON syntax."
            )
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
        # Deterministic ops can be slower; enabled here for reproducibility.
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


def build_config(
    raw: Dict[str, Any], repo_root: Path, config_path: Path, run_dir_override: Optional[str]
) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "run_name": config_path.stem,
        "seed": 17,
        "output_root": "runs",
        "mnid": {
            "input_dir": "datasets/Beethoven_motif-main/csv_notes_clean",
            "bps_fh_dir": "datasets/BPS_FH_Dataset",
            "dict_file": "MNID/data_creation/prepare_data/dict/CP.pkl",
            "ckpt_dir": "MNID/BPS_MNID_MI",
            "cpu": False,
        },
        "motif": {
            "method": "CSA",
            "csv_label_dir": "datasets/Beethoven_motif-main/csv_label",
            "motif_midi_dir": "datasets/Beethoven_motif-main/motif_midi",
            "save_predictions": True,
        },
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
    if run_dir is None:
        run_dir = output_root / run_name
    ensure_dir(run_dir)

    mnid_cfg = cfg["mnid"]
    mnid_cfg["input_dir"] = pathify(mnid_cfg.get("input_dir"), repo_root)
    mnid_cfg["bps_fh_dir"] = pathify(mnid_cfg.get("bps_fh_dir"), repo_root)
    mnid_cfg["dict_file"] = pathify(mnid_cfg.get("dict_file"), repo_root)
    mnid_cfg["ckpt_dir"] = pathify(mnid_cfg.get("ckpt_dir"), repo_root)
    if mnid_cfg.get("output_csv_dir"):
        mnid_cfg["output_csv_dir"] = pathify(mnid_cfg["output_csv_dir"], repo_root)
    else:
        mnid_cfg["output_csv_dir"] = run_dir / "predictions"
    ensure_dir(mnid_cfg["output_csv_dir"])

    motif_cfg = cfg["motif"]
    motif_cfg["csv_label_dir"] = pathify(motif_cfg.get("csv_label_dir"), repo_root)
    motif_cfg["motif_midi_dir"] = pathify(motif_cfg.get("motif_midi_dir"), repo_root)
    motif_cfg["save_predictions"] = bool(motif_cfg.get("save_predictions", False))
    if motif_cfg["save_predictions"]:
        pred_dir = run_dir / f"motifs_{str(motif_cfg['method']).lower()}"
        ensure_dir(pred_dir)
        motif_cfg["save_predictions_dir"] = pred_dir
    else:
        motif_cfg["save_predictions_dir"] = None

    cfg["run_dir"] = run_dir
    cfg["output_root"] = output_root
    return cfg


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config).expanduser().resolve()
    raw_config = load_config(config_path)
    cfg = build_config(raw_config, repo_root, config_path, args.run_dir)

    run_dir: Path = cfg["run_dir"]
    log_path = run_dir / "stdout.log"

    set_global_seeds(int(cfg.get("seed", 0)))

    # MNID inference (note prediction)
    mnid_cfg = cfg["mnid"]
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

    # Motif discovery + evaluation
    motif_cfg = cfg["motif"]
    motif_env = os.environ.copy()
    motif_env["PYTHONHASHSEED"] = str(cfg.get("seed", 0))

    motif_cmd = [
        sys.executable,
        str(repo_root / "motif_discovery" / "experiments.py"),
        "--csv_note_dir",
        str(mnid_cfg["output_csv_dir"]),
        "--method",
        str(motif_cfg["method"]),
    ]
    if motif_cfg.get("csv_label_dir"):
        motif_cmd.extend(["--csv_label_dir", str(motif_cfg["csv_label_dir"])])
    if motif_cfg.get("motif_midi_dir"):
        motif_cmd.extend(["--motif_midi_dir", str(motif_cfg["motif_midi_dir"])])
    if motif_cfg.get("save_predictions_dir"):
        motif_cmd.extend(["--save_predictions_dir", str(motif_cfg["save_predictions_dir"])])

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
        "method": motif_cfg.get("method"),
        "csv_note_dir": str(mnid_cfg["output_csv_dir"]),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    # Save resolved config for reproducibility
    resolved_path = run_dir / "config_resolved.yaml"
    cfg_serializable = deepcopy(cfg)
    cfg_serializable["run_dir"] = str(cfg_serializable["run_dir"])
    cfg_serializable["output_root"] = str(cfg_serializable["output_root"])
    for key in ("input_dir", "bps_fh_dir", "dict_file", "ckpt_dir", "output_csv_dir"):
        if key in cfg_serializable["mnid"]:
            cfg_serializable["mnid"][key] = str(cfg_serializable["mnid"][key])
    for key in ("csv_label_dir", "motif_midi_dir"):
        if key in cfg_serializable["motif"]:
            cfg_serializable["motif"][key] = str(cfg_serializable["motif"][key])
    if cfg_serializable["motif"].get("save_predictions_dir"):
        cfg_serializable["motif"]["save_predictions_dir"] = str(cfg_serializable["motif"]["save_predictions_dir"])

    resolved_path.write_text(dump_yaml_like(cfg_serializable) + "\n")

    repro_cmd = f"{sys.executable} {Path(__file__).relative_to(repo_root)} --config {config_path}"
    if args.run_dir:
        repro_cmd += f" --run-dir {args.run_dir}"

    summary_lines = [
        "Baseline run complete.",
        f"metrics.json: {metrics_path}",
        f"predictions:  {mnid_cfg['output_csv_dir']}",
    ]
    if motif_cfg.get("save_predictions_dir"):
        summary_lines.append(f"motifs:       {motif_cfg['save_predictions_dir']}")
    summary_lines.extend(
        [
            f"stdout log:   {log_path}",
            f"repro cmd:    {repro_cmd}",
        ]
    )
    summary = "\n".join(summary_lines)
    print(summary)
    with log_path.open("a") as lf:
        lf.write("\n" + summary + "\n")


if __name__ == "__main__":
    main()
