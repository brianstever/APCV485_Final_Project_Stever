"""Generate dataset summary and report figures that do not require training."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment import add_noise, apply_rir, mp3_compress
from src.data.preprocess import (
    SAMPLE_RATE,
    load_waveform,
    normalize_spectrogram,
    pad_or_trim,
    waveform_to_mel,
)


def count_protocol(path: str) -> dict:
    bonafide = 0
    spoof = 0
    systems = set()
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            if parts[4] == "bonafide":
                bonafide += 1
            else:
                spoof += 1
                systems.add(parts[3])
    return {
        "bonafide": bonafide,
        "spoof": spoof,
        "total": bonafide + spoof,
        "systems": sorted(systems),
    }


def save_dataset_stats(data_root: str, save_dir: str) -> dict:
    proto_dir = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols")
    protocols = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    stats = {}
    for split, filename in protocols.items():
        path = os.path.join(proto_dir, filename)
        if os.path.exists(path):
            stats[split] = count_protocol(path)
        else:
            stats[split] = {"missing": True, "path": path}
    path = os.path.join(save_dir, "dataset_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def first_flac(audio_dir: str) -> str:
    files = sorted(f for f in os.listdir(audio_dir) if f.endswith(".flac"))
    if not files:
        raise FileNotFoundError(f"No .flac files found in {audio_dir}")
    return os.path.join(audio_dir, files[0])


def save_sample_spectrogram(sample_path: str, save_dir: str) -> None:
    waveform = load_waveform(sample_path)
    waveform = pad_or_trim(waveform)
    mel = normalize_spectrogram(waveform_to_mel(waveform))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(waveform.numpy(), linewidth=0.3)
    axes[0].set_title("Waveform (4s, 16 kHz)")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True, alpha=0.25)

    im = axes[1].imshow(mel.numpy(), aspect="auto", origin="lower", cmap="magma")
    axes[1].set_title("128-bin Log-Mel Spectrogram (normalized)")
    axes[1].set_xlabel("Time Frame")
    axes[1].set_ylabel("Mel Bin")
    plt.colorbar(im, ax=axes[1])

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "sample_spectrogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_augmentation_comparison(sample_path: str, save_dir: str) -> None:
    clean = pad_or_trim(load_waveform(sample_path))
    augmentations = {
        "Clean": clean,
        "+ Noise (15 dB)": add_noise(clean, snr_db=15.0),
        "+ MP3 128k": mp3_compress(clean, bitrate="128k"),
        "+ Room Reverb": apply_rir(clean),
    }

    fig, axes = plt.subplots(len(augmentations), 2, figsize=(14, 3 * len(augmentations)))
    for i, (title, wav) in enumerate(augmentations.items()):
        mel_aug = normalize_spectrogram(waveform_to_mel(wav))

        axes[i, 0].plot(wav.numpy(), linewidth=0.3)
        axes[i, 0].set_title(f"{title} - Waveform")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].grid(True, alpha=0.25)

        axes[i, 1].imshow(mel_aug.numpy(), aspect="auto", origin="lower", cmap="magma")
        axes[i, 1].set_title(f"{title} - Spectrogram")
        axes[i, 1].set_ylabel("Mel Bin")

    axes[-1, 0].set_xlabel("Sample")
    axes[-1, 1].set_xlabel("Time Frame")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "augmentation_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate dataset stats and static report figures.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    stats = save_dataset_stats(args.data_root, args.save_dir)
    train_audio = os.path.join(args.data_root, "ASVspoof2019_LA_train", "flac")
    sample_path = first_flac(train_audio)
    save_sample_spectrogram(sample_path, args.save_dir)
    save_augmentation_comparison(sample_path, args.save_dir)

    print(json.dumps(stats, indent=2))
    print(f"Saved static artifacts to {args.save_dir}")


if __name__ == "__main__":
    main()
