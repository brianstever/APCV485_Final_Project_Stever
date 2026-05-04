"""
Evaluation utilities for the deepfake audio detector.

Computes EER (primary metric), AUC-ROC, accuracy, precision, recall,
F1-score, and single-sample inference latency.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_eval_dataloader
from src.models.cnn_se import DeepfakeCNNSE
from src.models.baseline import ResNet18Baseline
from src.augment import podcast_augment


def parse_systems(value: str | None) -> list[str] | None:
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def get_device() -> torch.device:
    """Prefer CUDA, then Apple Silicon MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Metrics ─────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Equal Error Rate: the threshold where FAR == FRR.

    labels: 0 = bonafide, 1 = spoof
    scores: higher = more likely spoof
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1_score": f1_score(labels, preds, zero_division=0),
    }


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


# ── Plotting helpers ────────────────────────────────────────────────

def plot_roc(labels, scores, save_path=None):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.25)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion(labels, preds, save_path=None):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Bonafide", "Spoof"],
        yticklabels=["Bonafide", "Spoof"],
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distribution(labels, scores, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.hist(scores[labels == 0], bins=50, alpha=0.55, label="Bonafide", density=True)
    plt.hist(scores[labels == 1], bins=50, alpha=0.55, label="Spoof", density=True)
    plt.xlabel("Detector Score")
    plt.ylabel("Density")
    plt.title("Score Distribution")
    plt.legend()
    plt.grid(True, alpha=0.25)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history_path, save_path=None):
    data = np.load(history_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(data["train_loss"], label="Train Loss")
    ax1.plot(data["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    ax2.plot(data["val_eer"], label="Val EER", color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("EER")
    ax2.set_title("Validation EER")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Latency measurement ────────────────────────────────────────────

def measure_latency(model, device, n_runs: int = 100) -> float:
    """Average inference time in ms for a single 4-second spectrogram."""
    model.eval()
    dummy = torch.randn(1, 1, 128, 401).to(device)

    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    times = []
    for _ in range(n_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times))


# ── Full evaluation run ────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, loader, device, no_progress=False):
    model.eval()
    all_scores = []
    all_labels = []
    total_samples = len(loader.dataset)
    processed = 0
    t0 = time.time()

    for batch_idx, (mels, labels) in enumerate(tqdm(loader, desc="eval", disable=no_progress), start=1):
        mels = mels.to(device)
        logits = model(mels).squeeze(1)
        scores = torch.sigmoid(logits).cpu()
        all_scores.append(scores)
        all_labels.append(labels)
        processed += labels.numel()

        if no_progress and (batch_idx % 250 == 0 or processed == total_samples):
            elapsed = max(time.time() - t0, 1e-6)
            rate = processed / elapsed
            print(f"eval progress: {processed}/{total_samples} clips ({rate:.1f} clips/s)", flush=True)

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()
    return all_labels, all_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake audio detector")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", choices=["cnn_se", "resnet18"], default="cnn_se")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--split", choices=["eval", "dev"], default="eval",
                        help="Evaluate on the official eval split or on dev when eval audio is unavailable")
    parser.add_argument("--protocol_path", type=str, default=None,
                        help="Optional explicit protocol/metadata path")
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Optional explicit directory containing evaluation audio")
    parser.add_argument("--protocol_format", choices=["asvspoof2019", "asvspoof2021"], default="asvspoof2019",
                        help="Column layout for labels in the protocol file")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Optional deterministic stratified cap for faster real-data evaluation")
    parser.add_argument("--augment_eval", action="store_true",
                        help="Apply podcast-style waveform augmentation during evaluation")
    parser.add_argument("--augment_prob", type=float, default=1.0,
                        help="Probability for eval-time waveform augmentation when --augment_eval is set")
    parser.add_argument("--include_systems", type=str, default=None,
                        help="Comma-separated spoof system IDs to keep; bonafide is always kept")
    parser.add_argument("--exclude_systems", type=str, default=None,
                        help="Comma-separated spoof system IDs to remove")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable tqdm progress bars for cleaner long-run logs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    if args.model == "cnn_se":
        model = DeepfakeCNNSE().to(device)
    else:
        model = ResNet18Baseline(pretrained=False).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val EER {ckpt['val_eer']:.4f})")

    protocol_filename = "ASVspoof2019.LA.cm.eval.trl.txt"
    audio_subdir = "ASVspoof2019_LA_eval/flac"
    if args.split == "dev":
        protocol_filename = "ASVspoof2019.LA.cm.dev.trl.txt"
        audio_subdir = "ASVspoof2019_LA_dev/flac"

    eval_loader = get_eval_dataloader(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_fn=podcast_augment if args.augment_eval else None,
        augment_prob=args.augment_prob if args.augment_eval else 0.0,
        protocol_filename=protocol_filename,
        audio_subdir=audio_subdir,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        protocol_path=args.protocol_path,
        audio_dir=args.audio_dir,
        include_systems=parse_systems(args.include_systems),
        exclude_systems=parse_systems(args.exclude_systems),
        file_id_index=1,
        system_index=4 if args.protocol_format == "asvspoof2021" else 3,
        label_index=5 if args.protocol_format == "asvspoof2021" else 4,
    )
    print(f"Evaluating {len(eval_loader.dataset)} clips on {device.type}", flush=True)
    labels, scores = evaluate_model(model, eval_loader, device, args.no_progress)
    preds = (scores >= 0.5).astype(int)

    eer = compute_eer(labels, scores)
    roc_auc = compute_auc(labels, scores)
    metrics = compute_metrics(labels, preds)
    latency = measure_latency(model, device)
    n_bonafide = int((labels == 0).sum())
    n_spoof = int((labels == 1).sum())

    print(f"\n{'='*40}")
    print(f"  Samples:    {len(labels)} ({n_bonafide} bonafide, {n_spoof} spoof)")
    print(f"  EER:        {eer:.4f}")
    print(f"  AUC-ROC:    {roc_auc:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1-Score:   {metrics['f1_score']:.4f}")
    print(f"  Latency:    {latency:.2f} ms / clip")
    print(f"{'='*40}\n")

    plot_roc(labels, scores,
             save_path=os.path.join(args.save_dir, "roc_curve.png"))
    plot_confusion(labels, preds,
                   save_path=os.path.join(args.save_dir, "confusion_matrix.png"))
    plot_score_distribution(labels, scores,
                            save_path=os.path.join(args.save_dir, "score_distribution.png"))

    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "data_root": args.data_root,
        "split": args.split,
        "protocol_path": args.protocol_path,
        "audio_dir": args.audio_dir,
        "protocol_format": args.protocol_format,
        "max_eval_samples": args.max_eval_samples,
        "augment_eval": args.augment_eval,
        "augment_prob": args.augment_prob if args.augment_eval else 0.0,
        "device": device.type,
        "n_samples": int(len(labels)),
        "n_bonafide": n_bonafide,
        "n_spoof": n_spoof,
        "decision_threshold": 0.5,
        "eer": eer,
        "auc_roc": roc_auc,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "latency_ms_per_clip": latency,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }
    summary_path = os.path.join(args.save_dir, f"metrics_summary_{args.model}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to {summary_path}")


if __name__ == "__main__":
    main()
