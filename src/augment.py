"""
Podcast-style audio augmentation.

Simulates the lossy compression, ambient noise, and room acoustics
typical of real-world podcast production.  All augmentations operate
on raw waveform tensors (1-D, 16 kHz) so they can be composed with
the spectrogram pipeline in the Dataset.

Augmentation is controlled per-split with separate random seeds to
prevent data leakage between training and evaluation.
"""

import io
import random
from typing import Optional

import torch
import numpy as np
import scipy.signal

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

SAMPLE_RATE = 16_000


# ── Additive noise ──────────────────────────────────────────────────

def add_noise(waveform: torch.Tensor, snr_db: Optional[float] = None) -> torch.Tensor:
    """Add Gaussian noise at a random SNR between 10 and 30 dB."""
    if snr_db is None:
        snr_db = random.uniform(10.0, 30.0)
    signal_power = waveform.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return waveform + noise


# ── MP3 compression ─────────────────────────────────────────────────

def mp3_compress(waveform: torch.Tensor, bitrate: str = "128k") -> torch.Tensor:
    """
    Round-trip through MP3 at the given bitrate to simulate lossy codec
    artifacts.  Falls back to a no-op if pydub/ffmpeg is unavailable.
    """
    if not HAS_PYDUB:
        return waveform

    pcm = (waveform.numpy() * 32767).astype(np.int16).tobytes()
    audio = AudioSegment(
        data=pcm,
        sample_width=2,
        frame_rate=SAMPLE_RATE,
        channels=1,
    )
    buf = io.BytesIO()
    audio.export(buf, format="mp3", bitrate=bitrate)
    buf.seek(0)
    compressed = AudioSegment.from_mp3(buf)
    samples = np.array(compressed.get_array_of_samples(), dtype=np.float32) / 32767.0
    result = torch.from_numpy(samples)
    if result.shape[0] > waveform.shape[0]:
        result = result[: waveform.shape[0]]
    elif result.shape[0] < waveform.shape[0]:
        result = torch.nn.functional.pad(result, (0, waveform.shape[0] - result.shape[0]))
    return result


# ── Synthetic room impulse response ─────────────────────────────────

def apply_rir(waveform: torch.Tensor, rir_length: int = 4000) -> torch.Tensor:
    """
    Convolve with a simple synthetic exponentially-decaying impulse
    response to approximate room reverb.  This is a lightweight
    stand-in when real RIR datasets are not available.
    """
    decay = np.exp(-np.linspace(0, 8, rir_length))
    noise = np.random.randn(rir_length)
    rir = (noise * decay).astype(np.float32)
    rir = rir / np.abs(rir).max()
    convolved = scipy.signal.fftconvolve(waveform.numpy(), rir, mode="full")
    convolved = convolved[: waveform.shape[0]]
    convolved = convolved / (np.abs(convolved).max() + 1e-8)
    return torch.from_numpy(convolved.astype(np.float32))


# ── SpecAugment (applied on spectrograms, not waveforms) ────────────

def spec_augment(
    mel: torch.Tensor,
    freq_mask_param: int = 15,
    time_mask_param: int = 40,
    num_masks: int = 2,
) -> torch.Tensor:
    """
    Apply SpecAugment-style frequency and time masking to a log-Mel
    spectrogram tensor of shape (n_mels, time).
    """
    augmented = mel.clone()
    n_mels, n_time = augmented.shape

    for _ in range(num_masks):
        f = random.randint(0, min(freq_mask_param, n_mels - 1))
        f0 = random.randint(0, n_mels - f)
        augmented[f0 : f0 + f, :] = 0.0

    for _ in range(num_masks):
        t = random.randint(0, min(time_mask_param, n_time - 1))
        t0 = random.randint(0, n_time - t)
        augmented[:, t0 : t0 + t] = 0.0

    return augmented


# ── Pitch shifting ──────────────────────────────────────────────────

def pitch_shift(waveform: torch.Tensor, semitones: Optional[float] = None) -> torch.Tensor:
    """
    Shift pitch by resampling.  A random shift between -2 and +2
    semitones is applied if no value is provided.
    """
    if semitones is None:
        semitones = random.uniform(-2.0, 2.0)
    rate = 2.0 ** (semitones / 12.0)
    indices = torch.arange(0, waveform.shape[0], rate)
    indices = indices.long().clamp(max=waveform.shape[0] - 1)
    shifted = waveform[indices]
    if shifted.shape[0] > waveform.shape[0]:
        shifted = shifted[: waveform.shape[0]]
    elif shifted.shape[0] < waveform.shape[0]:
        shifted = torch.nn.functional.pad(shifted, (0, waveform.shape[0] - shifted.shape[0]))
    return shifted


# ── Composite augmentation for the training pipeline ────────────────

def podcast_augment(waveform: torch.Tensor) -> torch.Tensor:
    """
    Randomly apply podcast-realistic augmentations to a waveform:
    background noise, room response, and MP3 compression.
    """
    if random.random() < 0.4:
        waveform = add_noise(waveform)
    if random.random() < 0.3:
        waveform = apply_rir(waveform)
    if random.random() < 0.3:
        waveform = mp3_compress(waveform)
    return waveform
