"""
Mel-spectrogram preprocessing pipeline for ASVspoof audio.

Converts raw waveforms to 128-bin log-Mel spectrograms using the project
setup: 16 kHz mono, 25 ms Hann window, 10 ms hop, 4-second clips, and
zero-mean / unit-variance normalization.
"""

import subprocess

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16_000
CLIP_DURATION_S = 4.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)  # 64000

N_MELS = 128
N_FFT = 400        # 25 ms * 16000
HOP_LENGTH = 160   # 10 ms * 16000
WIN_LENGTH = 400

_mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    window_fn=torch.hann_window,
)


def _load_waveform_ffmpeg(filepath: str) -> torch.Tensor:
    """Load audio through ffmpeg when torchaudio needs optional codecs."""
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", filepath,
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if samples.size == 0:
        raise RuntimeError(f"No audio samples decoded from {filepath}")
    return torch.from_numpy(samples)


def load_waveform(filepath: str) -> torch.Tensor:
    """Load an audio file, resample to 16 kHz mono, and return 1-D tensor."""
    try:
        waveform, sr = torchaudio.load(filepath)
    except (ImportError, RuntimeError, OSError):
        return _load_waveform_ffmpeg(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    return waveform.squeeze(0)


def pad_or_trim(waveform: torch.Tensor, target_length: int = CLIP_SAMPLES) -> torch.Tensor:
    """Pad with zeros or trim to a fixed number of samples."""
    if waveform.shape[0] > target_length:
        return waveform[:target_length]
    if waveform.shape[0] < target_length:
        padding = target_length - waveform.shape[0]
        return torch.nn.functional.pad(waveform, (0, padding))
    return waveform


def waveform_to_mel(waveform: torch.Tensor) -> torch.Tensor:
    """Convert a 1-D waveform tensor to a 128-bin log-Mel spectrogram."""
    mel = _mel_transform(waveform.unsqueeze(0))          # (1, n_mels, time)
    log_mel = torch.log(mel.clamp(min=1e-9))
    return log_mel.squeeze(0)                            # (n_mels, time)


def normalize_spectrogram(spec: torch.Tensor) -> torch.Tensor:
    """Zero-mean, unit-variance normalization per spectrogram."""
    mean = spec.mean()
    std = spec.std()
    if std < 1e-6:
        return spec - mean
    return (spec - mean) / std


def process_audio(filepath: str) -> torch.Tensor:
    """Full pipeline: load -> pad/trim -> Mel spectrogram -> normalize."""
    waveform = load_waveform(filepath)
    waveform = pad_or_trim(waveform)
    mel = waveform_to_mel(waveform)
    mel = normalize_spectrogram(mel)
    return mel  # (128, T)
