"""Deezer's "fakeprint" (Afchar et al., "A Fourier Explanation of AI-music Artifacts", ISMIR 2025).

Re-implemented from the paper and github.com/deezer/ismir25-ai-music-detector
(compute_fakeprints.py), which is CC BY-NC 4.0 and covered by Deezer patent
applications: research use only.

Transposed-convolution upsampling in a neural decoder leaves narrow spectral
peaks at multiples of the decoder's internal frame rates. The fakeprint is the
time-averaged dB spectrum with its lower envelope (hull) removed, so only those
peaks remain; a logistic regression on it is the published detector.

Differences from the original, all forced by this project's protocol: input is
a 10 s excerpt at 48 kHz (Deezer averages up to 180 s at 44.1 kHz), so the
5-16 kHz band has 3754 bins instead of 4087.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

N_FFT = 1 << 14
HOP = N_FFT // 2
FMIN, FMAX = 5000.0, 16000.0
HULL_AREA = 10
MIN_DB, MAX_DB = -45.0, 5.0


def mean_db_spectrum(x: np.ndarray) -> np.ndarray:
    """Mean over frames and channels of 10*log10(|STFT|^2), torch.stft-style (centre, reflect padding)."""
    x = np.atleast_2d(x).astype(np.float64)  # [C, T]
    win = np.hanning(N_FFT + 1)[:-1]  # periodic Hann, = torch.hann_window
    acc, count = np.zeros(N_FFT // 2 + 1), 0
    for ch in x:
        xp = np.pad(ch, (N_FFT // 2, N_FFT // 2), mode="reflect")
        n_frames = 1 + len(ch) // HOP
        frames = np.lib.stride_tricks.sliding_window_view(xp, N_FFT)[::HOP][:n_frames] * win
        S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        acc += (10 * np.log10(np.clip(S, 1e-10, 1e6))).sum(axis=0)
        count += len(frames)
    return acc / count


def _lower_hull(c: np.ndarray, area: int = HULL_AREA) -> np.ndarray:
    n = len(c)
    W = np.lib.stride_tricks.sliding_window_view(c, area)
    idx = np.unique(np.arange(n - area + 1) + W.argmin(axis=1))
    if idx[0] != 0:
        idx = np.r_[0, idx]
    if idx[-1] != n - 1:
        idx = np.r_[idx, n - 1]
    return idx


def fakeprint(x: np.ndarray, sr: int = 48000, fmin: float = FMIN, fmax: float = FMAX) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, fakeprint vector in [0, 1]) for a signal [C, T] or [T]."""
    fp = mean_db_spectrum(x)
    freqs = np.linspace(0, sr / 2, len(fp))
    m = (freqs > fmin) & (freqs < fmax)
    f, c = freqs[m], fp[m]
    idx = _lower_hull(c)
    base = interp1d(f[idx], c[idx], kind="quadratic")(f)
    base = np.maximum(base, MIN_DB)
    r = np.clip(np.maximum(c - base, 0.0), 0, MAX_DB)
    return f, (r / (1e-6 + r.max())).astype(np.float32)


def predicted_peaks(sr_out: float, strides: list[int], fmax: float) -> np.ndarray:
    """Peak frequencies that a decoder with these upsampling strides should leave (Afchar et al. 2025, Sec. 3)."""
    f0 = sr_out / float(np.prod(strides))
    return np.arange(0, fmax + 1e-9, f0)


# Decoder strides from the public configs, for checking measured peaks against theory.
KNOWN_DECODERS = {
    "EnCodec 24 kHz (Bark, early Suno)": (24000, [8, 5, 4, 2]),
    "EnCodec 32 kHz (MusicGen)": (32000, [8, 5, 4, 4]),
    "EnCodec 48 kHz": (48000, [8, 5, 4, 2]),
    "DAC 44.1 kHz": (44100, [8, 8, 4, 2]),
    "Stable Audio Oobleck 44.1 kHz": (44100, [8, 8, 4, 4, 2]),
}
