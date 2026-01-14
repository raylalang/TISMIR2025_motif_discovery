#!/usr/bin/env python3
"""
Train learned sequence encoder for LR_V1 using supervised contrastive loss.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import sys

# Make motif_discovery imports available
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "motif_discovery") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "motif_discovery"))

from motif_discovery.experiments import load_all_motives  # type: ignore
from motif_discovery.learned_retrieval.learned_embeddings import notes_to_sequence
from motif_discovery.learned_retrieval.learned_encoder import LearnedEncoderConfig, LRSequenceEncoder


def load_config(path: Path) -> Dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception:
            raise ValueError(f"Config {path} is YAML but PyYAML is not installed.")
        loaded = yaml.safe_load(text)
        return loaded if loaded else {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_run_dir(cfg: Dict, config_path: Path) -> Path:
    run_name = cfg.get("run_name") or config_path.stem
    output_root = Path(cfg.get("output_root", "runs")).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / f"{run_name}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def augment_sequence(seq: np.ndarray, cfg: Dict) -> np.ndarray:
    out = seq.copy()
    tempo_range = cfg.get("tempo_scale_range")
    if tempo_range and len(tempo_range) == 2:
        scale = random.uniform(float(tempo_range[0]), float(tempo_range[1]))
        out[:, 1] = out[:, 1] + np.log(max(scale, 1e-4))

    jitter_std = float(cfg.get("jitter_std", 0.0))
    if jitter_std > 0:
        out[:, 1] = out[:, 1] + np.random.normal(0.0, jitter_std, size=out[:, 1].shape)

    drop_prob = float(cfg.get("drop_prob", 0.0))
    if drop_prob > 0 and len(out) > 1:
        keep = np.random.rand(len(out)) > drop_prob
        if keep.sum() < 1:
            keep[random.randrange(len(out))] = True
        out = out[keep]
    return out


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[np.ndarray], labels: List[int], max_len: int, augment: Dict):
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        if self.augment:
            seq = augment_sequence(seq, self.augment)
        if self.max_len and len(seq) > self.max_len:
            seq = seq[: self.max_len]
        return seq, int(self.labels[idx])


def collate_batch(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    feat_dim = sequences[0].shape[1]
    x = np.zeros((len(sequences), max_len, feat_dim), dtype=np.float32)
    mask = np.zeros((len(sequences), max_len), dtype=bool)
    for i, seq in enumerate(sequences):
        L = len(seq)
        if L == 0:
            continue
        x[i, :L] = seq
        mask[i, :L] = True
    return torch.from_numpy(x), torch.from_numpy(mask), torch.tensor(labels, dtype=torch.long)


def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor | None:
    z = F.normalize(z, p=2, dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    eye = torch.eye(sim.size(0), device=sim.device)
    mask = mask - eye
    exp_sim = torch.exp(sim) * (1 - eye)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
    pos_counts = mask.sum(dim=1)
    valid = pos_counts > 0
    if not torch.any(valid):
        return None
    loss = -(mask * log_prob).sum(dim=1) / pos_counts.clamp(min=1)
    return loss[valid].mean()


def load_sequences(cfg: Dict) -> Tuple[List[np.ndarray], List[int]]:
    data_cfg = cfg.get("data", {})
    label_dir = Path(data_cfg.get("csv_label_dir", "")).expanduser()
    midi_dir = Path(data_cfg.get("motif_midi_dir", "")).expanduser()
    min_notes = int(data_cfg.get("min_notes", 3))
    max_occ = data_cfg.get("max_occurrences_per_motif")
    max_occ = int(max_occ) if max_occ is not None else None
    label_scope = str(data_cfg.get("label_scope", "piece"))
    use_duration = bool(data_cfg.get("use_duration", False))

    pieces = data_cfg.get("pieces")
    if not pieces:
        pieces = [f"{i:02d}-1" for i in range(1, 33)]

    sequences: List[np.ndarray] = []
    labels: List[int] = []
    label_map: Dict[str, int] = {}

    for piece in pieces:
        label_csv = label_dir / f"{piece}.csv"
        midi_path = midi_dir / f"{piece}.mid"
        motives = load_all_motives(str(label_csv), str(midi_path))
        for motif_type, occs in motives.items():
            occ_list = list(occs)
            random.shuffle(occ_list)
            if max_occ is not None:
                occ_list = occ_list[:max_occ]
            for occ in occ_list:
                if len(occ) < min_notes:
                    continue
                seq, _g = notes_to_sequence(occ, use_duration=use_duration)
                if seq is None or len(seq) == 0:
                    continue
                key = f"{piece}:{motif_type}" if label_scope == "piece" else str(motif_type)
                if key not in label_map:
                    label_map[key] = len(label_map)
                sequences.append(seq)
                labels.append(label_map[key])
    return sequences, labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LR_V1 learned embedding.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML/JSON config.")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    seed = int(cfg.get("seed", 17))
    set_seed(seed)

    run_dir = resolve_run_dir(cfg, Path(args.config))
    (run_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2))

    model_cfg_raw = cfg.get("model", {})
    model_cfg = LearnedEncoderConfig(
        input_dim=int(model_cfg_raw.get("input_dim", 2)),
        d_model=int(model_cfg_raw.get("d_model", 128)),
        n_layers=int(model_cfg_raw.get("n_layers", 2)),
        n_heads=int(model_cfg_raw.get("n_heads", 4)),
        dropout=float(model_cfg_raw.get("dropout", 0.1)),
        max_len=int(model_cfg_raw.get("max_len", 256)),
    )

    sequences, labels = load_sequences(cfg)
    if not sequences:
        raise ValueError("No training sequences found. Check data paths and config.")

    augment_cfg = cfg.get("augment", {})
    dataset = SequenceDataset(sequences, labels, model_cfg.max_len, augment_cfg)
    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 128))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    device = str(train_cfg.get("device", "cpu"))
    model = LRSequenceEncoder(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    temperature = float(train_cfg.get("temperature", 0.2))
    epochs = int(train_cfg.get("epochs", 10))
    log_every = int(train_cfg.get("log_every", 50))

    best_loss = None
    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)
            z = model(x, mask)
            loss = supcon_loss(z, y, temperature)
            if loss is None:
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running.append(float(loss.item()))
            step += 1
            if log_every and step % log_every == 0:
                avg = sum(running[-log_every:]) / min(len(running), log_every)
                print(f"epoch {epoch} step {step} loss {avg:.4f}")

        epoch_loss = float(np.mean(running)) if running else float("inf")
        print(f"epoch {epoch} mean loss {epoch_loss:.4f}")
        ckpt = {
            "config": asdict(model_cfg),
            "state_dict": model.state_dict(),
        }
        torch.save(ckpt, run_dir / "encoder_last.pt")
        if best_loss is None or epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, run_dir / "encoder.pt")

    stats = {
        "best_loss": best_loss,
        "epochs": epochs,
        "num_sequences": len(sequences),
        "num_labels": len(set(labels)),
        "model": asdict(model_cfg),
    }
    (run_dir / "train_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
