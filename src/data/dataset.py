"""
ASVspoof PyTorch Dataset and DataLoader helpers.

Parses the official ASVspoof 2019 LA protocol files to build labeled
sample lists, then loads and preprocesses audio on the fly.  Augmentation
is applied stochastically during training if an augment_fn is provided.
"""

import os
import random
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader

from src.data.preprocess import process_audio, load_waveform, pad_or_trim, waveform_to_mel, normalize_spectrogram
from src.augment import spec_augment


def _pin_memory_enabled() -> bool:
    """Pinned memory is useful for CUDA transfers but unsupported by MPS."""
    return torch.cuda.is_available()


def _limit_samples(
    samples: List[Tuple[str, int, str]],
    max_samples: Optional[int],
    seed: int,
) -> List[Tuple[str, int, str]]:
    """Return a deterministic stratified subset when a sample cap is requested."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(samples):
        return samples

    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for sample in samples:
        by_label[sample[1]].append(sample)

    limited: List[Tuple[str, int, str]] = []
    remaining = max_samples
    labels = [0, 1]
    for label in labels[:-1]:
        group = by_label[label]
        target = round(max_samples * (len(group) / len(samples))) if samples else 0
        target = min(len(group), max(1 if group else 0, target))
        rng.shuffle(group)
        limited.extend(group[:target])
        remaining -= target

    last_group = by_label[labels[-1]]
    rng.shuffle(last_group)
    limited.extend(last_group[: max(0, remaining)])
    rng.shuffle(limited)
    return limited[:max_samples]


class ASVspoofDataset(Dataset):
    """
    Loads ASVspoof 2019/2021 LA samples from a protocol file.

    Each line in the protocol has the form:
        SPEAKER_ID  AUDIO_FILE_ID  -  SYSTEM_ID  LABEL
    where LABEL is 'bonafide' or 'spoof'.
    """

    def __init__(
        self,
        protocol_path: str,
        audio_dir: str,
        augment_fn: Optional[Callable] = None,
        augment_prob: float = 0.0,
        apply_spec_augment: bool = False,
        file_ext: str = ".flac",
        include_systems: Optional[List[str]] = None,
        exclude_systems: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
        file_id_index: int = 1,
        system_index: int = 3,
        label_index: int = 4,
    ):
        self.audio_dir = audio_dir
        self.augment_fn = augment_fn
        self.augment_prob = augment_prob
        self.apply_spec_augment = apply_spec_augment
        self.file_ext = file_ext
        self.samples: List[Tuple[str, int, str]] = []

        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                needed_index = max(file_id_index, system_index, label_index)
                if len(parts) <= needed_index:
                    continue
                file_id = parts[file_id_index]
                system_id = parts[system_index]
                label_str = parts[label_index]
                if include_systems and system_id not in include_systems and label_str != "bonafide":
                    continue
                if exclude_systems and system_id in exclude_systems:
                    continue
                label = 0 if label_str == "bonafide" else 1
                self.samples.append((file_id, label, system_id))
        self.samples = _limit_samples(self.samples, max_samples, seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_id, label, _ = self.samples[idx]
        filepath = os.path.join(self.audio_dir, file_id + self.file_ext)

        waveform = load_waveform(filepath)

        if self.augment_fn is not None and torch.rand(1).item() < self.augment_prob:
            waveform = self.augment_fn(waveform)

        waveform = pad_or_trim(waveform)
        mel = waveform_to_mel(waveform)
        mel = normalize_spectrogram(mel)

        if self.apply_spec_augment:
            mel = spec_augment(mel)

        mel = mel.unsqueeze(0)  # (1, n_mels, time) — single-channel image

        return mel, label


def get_dataloaders(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    augment_fn: Optional[Callable] = None,
    augment_prob: float = 0.5,
    max_train_samples: Optional[int] = None,
    max_dev_samples: Optional[int] = None,
    train_include_systems: Optional[List[str]] = None,
    train_exclude_systems: Optional[List[str]] = None,
    dev_include_systems: Optional[List[str]] = None,
    dev_exclude_systems: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and dev DataLoaders from the standard ASVspoof 2019 LA layout.

    data_root should point at the LA/ subdirectory, e.g. data/LA/.
    Expected structure under data_root:
        ASVspoof2019_LA_cm_protocols/
            ASVspoof2019.LA.cm.train.trn.txt
            ASVspoof2019.LA.cm.dev.trl.txt
        ASVspoof2019_LA_train/flac/
        ASVspoof2019_LA_dev/flac/
    """
    proto_dir = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols")
    train_proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.train.trn.txt")
    dev_proto = os.path.join(proto_dir, "ASVspoof2019.LA.cm.dev.trl.txt")
    train_audio = os.path.join(data_root, "ASVspoof2019_LA_train", "flac")
    dev_audio = os.path.join(data_root, "ASVspoof2019_LA_dev", "flac")

    train_ds = ASVspoofDataset(
        protocol_path=train_proto,
        audio_dir=train_audio,
        augment_fn=augment_fn,
        augment_prob=augment_prob,
        apply_spec_augment=True,
        include_systems=train_include_systems,
        exclude_systems=train_exclude_systems,
        max_samples=max_train_samples,
        seed=seed,
    )
    dev_ds = ASVspoofDataset(
        protocol_path=dev_proto,
        audio_dir=dev_audio,
        augment_fn=None,
        augment_prob=0.0,
        include_systems=dev_include_systems,
        exclude_systems=dev_exclude_systems,
        max_samples=max_dev_samples,
        seed=seed + 1,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=_pin_memory_enabled(), drop_last=True,
        persistent_workers=num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin_memory_enabled(),
        persistent_workers=num_workers > 0,
    )
    return train_loader, dev_loader


def get_eval_dataloader(
    data_root: str,
    batch_size: int = 64,
    num_workers: int = 4,
    augment_fn: Optional[Callable] = None,
    augment_prob: float = 0.0,
    protocol_filename: str = "ASVspoof2019.LA.cm.eval.trl.txt",
    audio_subdir: str = "ASVspoof2019_LA_eval/flac",
    max_eval_samples: Optional[int] = None,
    seed: int = 42,
    protocol_path: Optional[str] = None,
    audio_dir: Optional[str] = None,
    include_systems: Optional[List[str]] = None,
    exclude_systems: Optional[List[str]] = None,
    file_id_index: int = 1,
    system_index: int = 3,
    label_index: int = 4,
) -> DataLoader:
    """Build an evaluation DataLoader (2019 eval or 2021 LA eval)."""
    proto_dir = os.path.join(data_root, "ASVspoof2019_LA_cm_protocols")
    eval_proto = protocol_path or os.path.join(proto_dir, protocol_filename)
    eval_audio = audio_dir or os.path.join(data_root, audio_subdir)

    eval_ds = ASVspoofDataset(
        protocol_path=eval_proto,
        audio_dir=eval_audio,
        augment_fn=augment_fn,
        augment_prob=augment_prob,
        include_systems=include_systems,
        exclude_systems=exclude_systems,
        max_samples=max_eval_samples,
        seed=seed,
        file_id_index=file_id_index,
        system_index=system_index,
        label_index=label_index,
    )
    return DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin_memory_enabled(),
        persistent_workers=num_workers > 0,
    )
