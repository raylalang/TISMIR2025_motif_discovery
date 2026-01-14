from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - torch optional at import time
    raise RuntimeError("PyTorch is required for learned embeddings.") from exc

from .learned_encoder import LearnedEncoderConfig, LRSequenceEncoder
from .segments import Segment


_MODEL_CACHE: dict[Tuple[str, str], Tuple[LRSequenceEncoder, LearnedEncoderConfig]] = {}


def _safe_log(dt: np.ndarray) -> np.ndarray:
    return np.log(np.clip(dt, 1e-4, None))


def notes_to_sequence(notes: np.ndarray, use_duration: bool = False) -> Tuple[np.ndarray | None, float]:
    if notes is None or len(notes) < 2:
        return None, 0.0
    if isinstance(notes, np.ndarray) and notes.dtype.names:
        notes = np.sort(notes, order=["onset", "pitch"])
        onsets = notes["onset"].astype(np.float32)
        pitches = notes["pitch"].astype(np.float32)
        durations = notes["duration"].astype(np.float32) if use_duration and "duration" in notes.dtype.names else None
    else:
        notes = np.asarray(notes)
        onsets = notes[:, 0].astype(np.float32)
        pitches = notes[:, 1].astype(np.float32)
        durations = notes[:, 2].astype(np.float32) if use_duration and notes.shape[1] > 2 else None

    dp = np.diff(pitches)
    log_dt = _safe_log(np.diff(onsets))
    g = float(np.mean(log_dt)) if log_dt.size else 0.0

    feats = [dp, log_dt]
    if durations is not None:
        feats.append(durations[1:])
    seq = np.stack(feats, axis=1).astype(np.float32)
    return seq, g


def segment_to_sequence(
    notes: np.ndarray, segment: Segment, use_duration: bool = False
) -> Tuple[np.ndarray | None, float]:
    idxs = segment.note_indices
    if idxs is None or len(idxs) < 2:
        return None, 0.0
    seg_notes = notes[idxs]
    return notes_to_sequence(seg_notes, use_duration=use_duration)


def _pad_batch(seqs: Sequence[np.ndarray], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = [min(len(s), max_len) for s in seqs]
    batch = np.zeros((len(seqs), max_len, seqs[0].shape[1]), dtype=np.float32)
    mask = np.zeros((len(seqs), max_len), dtype=bool)
    for i, seq in enumerate(seqs):
        L = lengths[i]
        if L == 0:
            continue
        batch[i, :L] = seq[:L]
        mask[i, :L] = True
    return torch.from_numpy(batch), torch.from_numpy(mask)


def load_learned_encoder(
    ckpt_path: str, device: str = "cpu"
) -> Tuple[LRSequenceEncoder, LearnedEncoderConfig]:
    key = (ckpt_path, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    payload = torch.load(ckpt_path, map_location=device)
    cfg_dict = payload.get("config", {})
    cfg = LearnedEncoderConfig(**cfg_dict)
    model = LRSequenceEncoder(cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    _MODEL_CACHE[key] = (model, cfg)
    return model, cfg


def embed_sequences(
    sequences: Sequence[np.ndarray],
    model: LRSequenceEncoder,
    device: str = "cpu",
    batch_size: int = 128,
) -> np.ndarray:
    if not sequences:
        return np.zeros((0, model.out_dim), dtype=np.float32)
    max_len = model.cfg.max_len
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            x, mask = _pad_batch(batch_seqs, max_len)
            x = x.to(device)
            mask = mask.to(device)
            emb = model(x, mask).cpu().numpy()
            outputs.append(emb)
    return np.concatenate(outputs, axis=0) if outputs else np.zeros((0, model.out_dim), dtype=np.float32)


def embed_segments_learned(
    notes: np.ndarray,
    segments: Sequence[Segment],
    model: LRSequenceEncoder,
    device: str = "cpu",
    batch_size: int = 128,
    use_duration: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Sequence[Segment]]:
    tempos = np.zeros(len(segments), dtype=np.float32)
    sequences: List[np.ndarray] = []
    valid_indices: List[int] = []
    for i, seg in enumerate(segments):
        seq, g = segment_to_sequence(notes, seg, use_duration=use_duration)
        tempos[i] = float(g)
        if seq is None or len(seq) == 0:
            continue
        sequences.append(seq)
        valid_indices.append(i)

    embeddings = np.zeros((len(segments), model.out_dim), dtype=np.float32)
    if sequences:
        emb_valid = embed_sequences(sequences, model, device=device, batch_size=batch_size)
        for idx, emb in zip(valid_indices, emb_valid):
            embeddings[idx] = emb
    return embeddings, tempos, segments
