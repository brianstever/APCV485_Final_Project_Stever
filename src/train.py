"""
Training loop for the deepfake audio detector.

Supports the CNN+SE model and the ResNet-18 baseline.  Uses binary
cross-entropy with inverse-frequency class weights, Adam with cosine
annealing, mixed-precision training, and model checkpointing.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import get_dataloaders
from src.models.cnn_se import DeepfakeCNNSE
from src.models.baseline import ResNet18Baseline
from src.augment import podcast_augment
from src.eval import compute_eer, compute_metrics


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


def compute_class_weights(loader) -> torch.Tensor:
    """Inverse-frequency weights from the dataset's sample list (no audio I/O)."""
    counts = np.zeros(2)
    for _, label, _ in loader.dataset.samples:
        counts[label] += 1
    w_pos = counts[0] / counts[1] if counts[1] > 0 else 1.0
    return torch.tensor([w_pos], dtype=torch.float32)


def sample_counts(loader) -> dict:
    """Summarize dataset labels without touching audio files."""
    counts = {0: 0, 1: 0}
    for _, label, _ in loader.dataset.samples:
        counts[label] += 1
    return {
        "n_samples": len(loader.dataset.samples),
        "n_bonafide": counts[0],
        "n_spoof": counts[1],
    }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, no_progress=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for mels, labels in tqdm(loader, desc="  train", leave=False, disable=no_progress):
        mels = mels.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(mels).squeeze(1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * mels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += mels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, no_progress=False):
    model.eval()
    running_loss = 0.0
    all_scores = []
    all_labels = []

    for mels, labels in tqdm(loader, desc="  val  ", leave=False, disable=no_progress):
        mels = mels.to(device)
        labels = labels.float().to(device)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(mels).squeeze(1)
            loss = criterion(logits, labels)

        running_loss += loss.item() * mels.size(0)
        scores = torch.sigmoid(logits).cpu()
        all_scores.append(scores)
        all_labels.append(labels.cpu().long())

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = running_loss / len(all_labels)
    eer = compute_eer(all_labels, all_scores)
    metrics = compute_metrics(all_labels, (all_scores >= 0.5).astype(int))
    return avg_loss, eer, metrics


def save_training_history_plot(history, save_path: str) -> None:
    """Save training/validation loss and EER curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    ax2.plot(history["val_eer"], label="Val EER", color="tab:orange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("EER")
    ax2.set_title("Validation EER")
    ax2.legend()
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train deepfake audio detector")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model", choices=["cnn_se", "resnet18"], default="cnn_se")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true",
                        help="Enable podcast-style augmentation on training data")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Optional deterministic stratified cap for faster real-data runs")
    parser.add_argument("--max_dev_samples", type=int, default=None,
                        help="Optional deterministic stratified cap for validation")
    parser.add_argument("--train_include_systems", type=str, default=None,
                        help="Comma-separated spoof system IDs to keep in training; bonafide is always kept")
    parser.add_argument("--train_exclude_systems", type=str, default=None,
                        help="Comma-separated spoof system IDs to remove from training")
    parser.add_argument("--dev_include_systems", type=str, default=None,
                        help="Comma-separated spoof system IDs to keep in dev; bonafide is always kept")
    parser.add_argument("--dev_exclude_systems", type=str, default=None,
                        help="Comma-separated spoof system IDs to remove from dev")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable tqdm progress bars for cleaner long-run logs")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    aug_fn = podcast_augment if args.augment else None
    train_loader, dev_loader = get_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_fn=aug_fn,
        augment_prob=0.5,
        max_train_samples=args.max_train_samples,
        max_dev_samples=args.max_dev_samples,
        train_include_systems=parse_systems(args.train_include_systems),
        train_exclude_systems=parse_systems(args.train_exclude_systems),
        dev_include_systems=parse_systems(args.dev_include_systems),
        dev_exclude_systems=parse_systems(args.dev_exclude_systems),
        seed=args.seed,
    )

    if args.model == "cnn_se":
        model = DeepfakeCNNSE().to(device)
    else:
        model = ResNet18Baseline(pretrained=False).to(device)

    pos_weight = compute_class_weights(train_loader).to(device)
    train_counts = sample_counts(train_loader)
    dev_counts = sample_counts(dev_loader)
    print(
        "Training setup: "
        f"{train_counts['n_samples']} train clips "
        f"({train_counts['n_bonafide']} bonafide, {train_counts['n_spoof']} spoof), "
        f"{dev_counts['n_samples']} dev clips "
        f"({dev_counts['n_bonafide']} bonafide, {dev_counts['n_spoof']} spoof), "
        f"device={device.type}",
        flush=True,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_eer = 1.0
    history = {"train_loss": [], "val_loss": [], "val_eer": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, args.no_progress,
        )
        val_loss, val_eer, val_metrics = validate(model, dev_loader, criterion, device, args.no_progress)
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_eer"].append(val_eer)

        print(
            f"Epoch {epoch:>3d}/{args.epochs} | "
            f"train_loss {train_loss:.4f}  acc {train_acc:.3f} | "
            f"val_loss {val_loss:.4f}  EER {val_eer:.4f} | "
            f"{elapsed:.1f}s"
        )

        if val_eer < best_eer:
            best_eer = val_eer
            ckpt_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_eer": val_eer,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> saved checkpoint (EER {val_eer:.4f})")

    history_path = os.path.join(args.save_dir, "training_history.npz")
    np.savez(
        history_path,
        **{k: np.array(v) for k, v in history.items()},
    )
    save_training_history_plot(history, os.path.join(args.save_dir, "training_history.png"))
    summary_path = os.path.join(args.save_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "data_root": args.data_root,
            "device": device.type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "augment": args.augment,
            "augment_prob": 0.5 if args.augment else 0.0,
            "max_train_samples": args.max_train_samples,
            "max_dev_samples": args.max_dev_samples,
            "train_include_systems": parse_systems(args.train_include_systems),
            "train_exclude_systems": parse_systems(args.train_exclude_systems),
            "dev_include_systems": parse_systems(args.dev_include_systems),
            "dev_exclude_systems": parse_systems(args.dev_exclude_systems),
            "train_counts": train_counts,
            "dev_counts": dev_counts,
            "best_val_eer": best_eer,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "final_val_eer": history["val_eer"][-1],
        }, f, indent=2)
    print(f"\nTraining complete. Best validation EER: {best_eer:.4f}")


if __name__ == "__main__":
    main()
