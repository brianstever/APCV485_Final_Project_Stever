"""LFCC + GMM baseline for ASVspoof-style spoofing detection.

This script implements a traditional signal-processing baseline with LFCC
frame features and one diagonal-covariance Gaussian mixture model per class.
Scores are log p(spoof) - log p(bonafide).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import torch
import torchaudio
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import ASVspoofDataset
from src.data.preprocess import SAMPLE_RATE, load_waveform, pad_or_trim

_WORKER_TRANSFORM = None
_WORKER_BONA_GMM = None
_WORKER_SPOOF_GMM = None
_WORKER_AUDIO_DIR = None
_WORKER_FILE_EXT = ".flac"


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1_score": f1_score(labels, preds, zero_division=0),
    }


def make_lfcc() -> torchaudio.transforms.LFCC:
    return torchaudio.transforms.LFCC(
        sample_rate=SAMPLE_RATE,
        n_lfcc=60,
        speckwargs={
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "window_fn": torch.hann_window,
        },
    )


def lfcc_frames(filepath: str, transform: torchaudio.transforms.LFCC) -> np.ndarray:
    waveform = pad_or_trim(load_waveform(filepath))
    with torch.no_grad():
        feat = transform(waveform.unsqueeze(0)).squeeze(0).transpose(0, 1)
    feat = feat.numpy().astype(np.float32)
    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-6
    return (feat - mean) / std


def sample_frames(frames: np.ndarray, max_frames: int, rng: np.random.Generator) -> np.ndarray:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames
    idx = rng.choice(len(frames), size=max_frames, replace=False)
    return frames[idx]


def build_dataset(protocol_path: str, audio_dir: str, protocol_format: str, seed: int) -> ASVspoofDataset:
    return ASVspoofDataset(
        protocol_path=protocol_path,
        audio_dir=audio_dir,
        file_id_index=1,
        system_index=4 if protocol_format == "asvspoof2021" else 3,
        label_index=5 if protocol_format == "asvspoof2021" else 4,
        seed=seed,
    )


def collect_training_frames(
    dataset: ASVspoofDataset,
    transform: torchaudio.transforms.LFCC,
    max_frames_per_file: int,
    max_frames_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[np.ndarray]] = {0: [], 1: []}
    counts = {0: 0, 1: 0}

    samples = list(dataset.samples)
    random.Random(seed).shuffle(samples)
    for file_id, label, _ in tqdm(samples, desc="collect LFCC"):
        if counts[label] >= max_frames_per_class:
            continue
        filepath = os.path.join(dataset.audio_dir, file_id + dataset.file_ext)
        frames = lfcc_frames(filepath, transform)
        frames = sample_frames(frames, max_frames_per_file, rng)
        remaining = max_frames_per_class - counts[label]
        frames = frames[:remaining]
        by_class[label].append(frames)
        counts[label] += len(frames)
        if counts[0] >= max_frames_per_class and counts[1] >= max_frames_per_class:
            break

    return np.vstack(by_class[0]), np.vstack(by_class[1])


def train_gmms(
    bonafide_frames: np.ndarray,
    spoof_frames: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[GaussianMixture, GaussianMixture]:
    bona = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=150,
        random_state=seed,
        verbose=1,
    ).fit(bonafide_frames)
    spoof = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        max_iter=150,
        random_state=seed,
        verbose=1,
    ).fit(spoof_frames)
    return bona, spoof


def _init_score_worker(model_path: str, audio_dir: str, file_ext: str) -> None:
    """Initialize each scoring process with its own transform and loaded GMMs."""
    global _WORKER_TRANSFORM, _WORKER_BONA_GMM, _WORKER_SPOOF_GMM, _WORKER_AUDIO_DIR, _WORKER_FILE_EXT
    bundle = joblib.load(model_path)
    _WORKER_TRANSFORM = make_lfcc()
    _WORKER_BONA_GMM = bundle["bonafide"]
    _WORKER_SPOOF_GMM = bundle["spoof"]
    _WORKER_AUDIO_DIR = audio_dir
    _WORKER_FILE_EXT = file_ext


def _score_one(sample: tuple[str, int, str]) -> tuple[int, float]:
    file_id, label, _ = sample
    filepath = os.path.join(_WORKER_AUDIO_DIR, file_id + _WORKER_FILE_EXT)
    frames = lfcc_frames(filepath, _WORKER_TRANSFORM)
    bona_ll = _WORKER_BONA_GMM.score(frames)
    spoof_ll = _WORKER_SPOOF_GMM.score(frames)
    return label, float(spoof_ll - bona_ll)


def score_dataset(
    dataset: ASVspoofDataset,
    transform: torchaudio.transforms.LFCC,
    bona_gmm: GaussianMixture,
    spoof_gmm: GaussianMixture,
    model_path: str | None = None,
    num_workers: int = 1,
    no_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    scores = []

    if num_workers > 1:
        if model_path is None:
            raise ValueError("Parallel scoring requires a saved --model_path")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_score_worker,
            initargs=(model_path, dataset.audio_dir, dataset.file_ext),
        ) as executor:
            iterator = executor.map(_score_one, dataset.samples, chunksize=64)
            iterator = tqdm(
                iterator,
                total=len(dataset.samples),
                desc="score LFCC-GMM",
                disable=no_progress,
            )
            t0 = time.time()
            for idx, (label, score) in enumerate(iterator, start=1):
                labels.append(label)
                scores.append(score)
                if no_progress and (idx % 10_000 == 0 or idx == len(dataset.samples)):
                    elapsed = max(time.time() - t0, 1e-6)
                    print(
                        f"score progress: {idx}/{len(dataset.samples)} clips ({idx / elapsed:.1f} clips/s)",
                        flush=True,
                    )
        return np.array(labels), np.array(scores)

    t0 = time.time()
    iterator = tqdm(dataset.samples, desc="score LFCC-GMM", disable=no_progress)
    for idx, (file_id, label, _) in enumerate(iterator, start=1):
        filepath = os.path.join(dataset.audio_dir, file_id + dataset.file_ext)
        frames = lfcc_frames(filepath, transform)
        bona_ll = bona_gmm.score(frames)
        spoof_ll = spoof_gmm.score(frames)
        scores.append(spoof_ll - bona_ll)
        labels.append(label)
        if no_progress and (idx % 10_000 == 0 or idx == len(dataset.samples)):
            elapsed = max(time.time() - t0, 1e-6)
            print(f"score progress: {idx}/{len(dataset.samples)} clips ({idx / elapsed:.1f} clips/s)", flush=True)
    return np.array(labels), np.array(scores)


def save_summary(
    labels: np.ndarray,
    scores: np.ndarray,
    save_dir: str,
    model_path: str,
    train_protocol: str,
    eval_protocol: str,
) -> dict[str, object]:
    preds = (scores >= 0).astype(int)
    eer = compute_eer(labels, scores)
    roc_auc = compute_auc(labels, scores)
    metrics = compute_metrics(labels, preds)
    n_bonafide = int((labels == 0).sum())
    n_spoof = int((labels == 1).sum())
    summary = {
        "model": "lfcc_gmm",
        "model_path": model_path,
        "train_protocol": train_protocol,
        "eval_protocol": eval_protocol,
        "n_samples": int(len(labels)),
        "n_bonafide": n_bonafide,
        "n_spoof": n_spoof,
        "decision_threshold": 0.0,
        "eer": eer,
        "auc_roc": roc_auc,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metrics_summary_lfcc_gmm.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate an LFCC + GMM baseline.")
    parser.add_argument("--train_protocol", required=True)
    parser.add_argument("--train_audio_dir", required=True)
    parser.add_argument("--eval_protocol", required=True)
    parser.add_argument("--eval_audio_dir", required=True)
    parser.add_argument("--train_protocol_format", choices=["asvspoof2019", "asvspoof2021"], default="asvspoof2019")
    parser.add_argument("--eval_protocol_format", choices=["asvspoof2019", "asvspoof2021"], default="asvspoof2019")
    parser.add_argument("--save_dir", default="results/lfcc_gmm")
    parser.add_argument("--model_path", default=None,
                        help="Optional existing LFCC-GMM joblib path for eval-only scoring")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training and score with --model_path or save_dir/lfcc_gmm.joblib")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of processes to use for LFCC-GMM scoring")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable tqdm progress bars and print sparse progress updates")
    parser.add_argument("--n_components", type=int, default=16)
    parser.add_argument("--max_frames_per_file", type=int, default=40)
    parser.add_argument("--max_frames_per_class", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    transform = make_lfcc()
    train_ds = build_dataset(args.train_protocol, args.train_audio_dir, args.train_protocol_format, args.seed)
    eval_ds = build_dataset(args.eval_protocol, args.eval_audio_dir, args.eval_protocol_format, args.seed + 1)

    t0 = time.time()
    model_path = args.model_path or os.path.join(args.save_dir, "lfcc_gmm.joblib")
    if args.eval_only:
        bundle = joblib.load(model_path)
        bona_gmm = bundle["bonafide"]
        spoof_gmm = bundle["spoof"]
    else:
        bona_frames, spoof_frames = collect_training_frames(
            train_ds,
            transform,
            args.max_frames_per_file,
            args.max_frames_per_class,
            args.seed,
        )
        bona_gmm, spoof_gmm = train_gmms(bona_frames, spoof_frames, args.n_components, args.seed)
        joblib.dump({"bonafide": bona_gmm, "spoof": spoof_gmm, "args": vars(args)}, model_path)

    print(f"Scoring {len(eval_ds.samples)} clips with {max(1, args.num_workers)} worker(s)", flush=True)
    labels, scores = score_dataset(
        eval_ds,
        transform,
        bona_gmm,
        spoof_gmm,
        model_path=model_path,
        num_workers=args.num_workers,
        no_progress=args.no_progress,
    )
    summary = save_summary(labels, scores, args.save_dir, model_path, args.train_protocol, args.eval_protocol)
    summary["elapsed_seconds"] = time.time() - t0
    with open(os.path.join(args.save_dir, "metrics_summary_lfcc_gmm.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
