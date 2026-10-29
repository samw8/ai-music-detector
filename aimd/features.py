"""Interpretable features for human-vs-AI music detection.

Every feature is a small function with a one-sentence docstring saying what it
measures and why it should differ between human and AI audio. The docstrings
are also used to print human-readable reasons in ``predict.py`` and to build the
feature table in the model card, so keep them to one sentence.

Input: a 10-second excerpt at 48 kHz, stereo if available (shape [2, n] or [n]).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sps

SR = 48000
EPS = 1e-12

# name -> (function, group)
REGISTRY: dict[str, tuple] = {}

# Groups whose values mostly describe the file format rather than the music.
# They are reported, but training scripts can exclude them when the human and
# AI sides of a dataset were stored in different formats (a confound).
FORMAT_GROUPS = {"stereo", "bandlimit"}


def feature(group: str):
    def deco(fn):
        REGISTRY[fn.__name__] = (fn, group)
        return fn
    return deco


def doc(name: str) -> str:
    return (REGISTRY[name][0].__doc__ or "").strip()


@dataclass
class Prepared:
    """Shared intermediate representations, computed lazily and cached."""

    stereo: np.ndarray  # [2, n] float32 (duplicated channel if the source was mono)
    sr: int = SR
    source_channels: int = 2
    cache: dict = field(default_factory=dict)

    @cached_property
    def mono(self) -> np.ndarray:
        return self.stereo.mean(axis=0)

    # ---- frame-level power spectrogram (n_fft 2048, ~23 Hz bins) ----
    @cached_property
    def stft_power(self) -> tuple[np.ndarray, np.ndarray]:
        f, _, Z = sps.stft(self.mono, fs=self.sr, nperseg=2048, noverlap=1536, window="hann", boundary=None, padded=False)
        return f, (np.abs(Z) ** 2).astype(np.float64)

    @cached_property
    def frame_total(self) -> np.ndarray:
        return self.stft_power[1].sum(axis=0) + EPS

    @cached_property
    def active(self) -> np.ndarray:
        """Frames that are not digital silence (a single 16-bit LSB of signal clears this floor)."""
        return self.stft_power[1].sum(axis=0) > 1e-10

    @cached_property
    def valid_hi(self) -> float:
        """Highest frequency with real content: 500 Hz below a band-limit cliff, or where the
        average spectrum has fallen 70 dB below its peak (gentle MP3 roll-offs), whichever is lower.

        Bands above this hold only the resampler's stop-band (e.g. a 16 kHz-native file
        re-decoded at 48 kHz), so statistics there would describe the file format, not the music.
        """
        f, _ = self.stft_power
        m = ndi.uniform_filter1d(self.mean_db, 21)  # ~500 Hz smoothing
        floor_db = 10 * np.log10(EPS)  # where empty bins of a 48 kHz PCM16 clip land in mean_db
        above = np.flatnonzero(m > max(m.max() - 70, floor_db + 3))  # quiet clips: anchor to the floor, not only the peak
        level_hi = float(f[above[-1]]) if len(above) else self.sr / 2
        edge_hi = self.edge[0] - 500.0 if self.edge[1] > 20 else self.sr / 2
        return min(level_hi, edge_hi)

    @cached_property
    def mean_db(self) -> np.ndarray:
        return 10 * np.log10(self.stft_power[1].mean(axis=1) + EPS)

    # ---- long-term average spectrum at fine resolution (n_fft 16384, ~2.9 Hz bins) ----
    @cached_property
    def fine(self) -> tuple[np.ndarray, np.ndarray]:
        f, P = sps.welch(self.mono, fs=self.sr, nperseg=16384, noverlap=8192, window="hann", average="mean")
        return f, 10 * np.log10(P + EPS)

    @cached_property
    def edge(self) -> tuple[float, float]:
        """(cutoff_hz, drop_db): steepest fall-off of the average spectrum above 8 kHz."""
        f, _ = self.stft_power
        m = ndi.uniform_filter1d(self.mean_db, 5)
        df = f[1] - f[0]
        w = int(round(500 / df))
        best_f, best_drop = float(f[-1]), 0.0
        for i in range(np.searchsorted(f, 8000), len(f) - w):
            drop = m[max(0, i - w):i].mean() - m[i:i + w].mean()
            if drop > best_drop:
                best_f, best_drop = float(f[i]), float(drop)
        return best_f, best_drop

    # ---- detrended fine spectrum and its second-order FFT ----
    @cached_property
    def residual(self) -> tuple[np.ndarray, float]:
        """Fine spectrum minus its ~150 Hz running median, between 500 Hz and the band-limit edge."""
        f, db = self.fine
        df = f[1] - f[0]
        hi = min(16000.0, self.valid_hi)  # never analyse the empty stop-band above the clip's real bandwidth
        if hi < 4000.0:
            return np.full(8, np.nan), float(df)
        band = (f >= 500) & (f <= hi)
        x = db[band]
        k = int(round(150 / df)) | 1
        trend = ndi.median_filter(x, size=k, mode="nearest")
        return x - trend, float(df)

    @cached_property
    def sofft(self) -> tuple[np.ndarray, np.ndarray]:
        """Second-order FFT: magnitude of the FFT of the detrended log spectrum, indexed by peak spacing in Hz."""
        r, df = self.residual
        if np.isnan(r).any():
            return np.array([np.nan]), np.array([np.nan])
        r = (r - r.mean()) * np.hanning(len(r))
        R = np.abs(np.fft.rfft(r))
        k = np.arange(len(R))
        with np.errstate(divide="ignore"):
            spacing = np.where(k > 0, len(r) * df / np.maximum(k, 1), np.inf)
        keep = (spacing >= 25) & (spacing <= 1500)
        return spacing[keep], R[keep]


MIN_ACTIVE = 0.25  # per-frame statistics need at least a quarter of the frames to be non-silent


def _band_mask(P: Prepared, lo: float, hi: float) -> np.ndarray | None:
    """Bins of [lo, hi) below the clip's real bandwidth; None if more than half the band lies above it."""
    f, _ = P.stft_power
    top = min(hi, P.valid_hi)
    if top - lo < 0.5 * (hi - lo):
        return None
    return (f >= lo) & (f < top)


def _flat(B: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Per-frame spectral flatness with a floor relative to each frame's level (not an absolute epsilon)."""
    B = B + floor
    return np.exp(np.log(B).mean(axis=0)) / B.mean(axis=0)


def _band_rel_db(P: Prepared, lo: float, hi: float) -> float:
    m = _band_mask(P, lo, hi)
    if m is None:
        return float("nan")
    _, S = P.stft_power
    e = S[m].sum(axis=0).mean()
    return float(10 * np.log10(e / P.frame_total.mean() + EPS))


def _band_frames_db(P: Prepared, lo: float, hi: float) -> np.ndarray | None:
    m = _band_mask(P, lo, hi)
    if m is None or P.active.mean() < MIN_ACTIVE:
        return None
    _, S = P.stft_power
    a = P.active
    return 10 * np.log10(S[m][:, a].sum(axis=0) / P.frame_total[a] + EPS)


def _band_flatness(P: Prepared, lo: float, hi: float) -> float:
    m = _band_mask(P, lo, hi)
    if m is None or P.active.mean() < MIN_ACTIVE:
        return float("nan")
    _, S = P.stft_power
    a = P.active
    floor = 1e-8 * P.frame_total[a] / S.shape[0]  # -80 dB re the frame's average bin
    return float(np.mean(_flat(S[m][:, a], floor)))


# ===================== second-order FFT (upsampling fingerprint) =====================

@feature("fourier")
def sofft_peak_ratio_db(P: Prepared) -> float:
    """Height of the tallest spike in the second-order FFT over its median, in dB: evenly spaced spectral peaks left by transposed-convolution upsampling in neural decoders make one tall spike, while human recordings give diffuse energy."""
    _, R = P.sofft
    return float(20 * np.log10(R.max() / (np.median(R) + EPS) + EPS))


@feature("fourier")
def sofft_top_spacing_hz(P: Prepared) -> float:
    """Spacing in Hz of the most periodic spectral-peak pattern; neural decoders repeat at their latent frame rate (e.g. 75 Hz for EnCodec-24k, 86 Hz for DAC-44k), human audio has no preferred spacing."""
    spacing, R = P.sofft
    return float(spacing[np.argmax(R)])


@feature("fourier")
def sofft_comb_db(P: Prepared) -> float:
    """Average height of the second-order FFT at multiples of its tallest spike, in dB over the median: a true upsampling grid repeats at every multiple, an accidental spike does not."""
    r, df = P.residual
    R = np.abs(np.fft.rfft((r - r.mean()) * np.hanning(len(r))))
    spacing, Rk = P.sofft
    k0 = int(round(len(r) * df / spacing[np.argmax(Rk)]))
    ks = [m * k0 for m in range(2, 6) if m * k0 < len(R)]
    if not ks:
        return 0.0
    med = np.median(Rk) + EPS
    return float(np.mean([20 * np.log10(R[max(k - 1, 0):k + 2].max() / med + EPS) for k in ks]))


@feature("fourier")
def sofft_kurtosis(P: Prepared) -> float:
    """Kurtosis of the second-order FFT: a few sharp spikes (synthetic periodic peaks) give a heavy-tailed shape, natural spectra give a light-tailed one."""
    _, R = P.sofft
    z = (R - R.mean()) / (R.std() + EPS)
    return float(np.mean(z ** 4))


@feature("fourier")
def fine_peak_rate(P: Prepared) -> float:
    """Narrow peaks per kHz that stick out at least 6 dB from the local spectral trend, a direct count of the tonal 'spikes' that decoder artifacts add to the long-term spectrum."""
    r, df = P.residual
    if np.isnan(r).any():
        return float("nan")
    peaks, _ = sps.find_peaks(r, prominence=6.0, width=(None, max(2, int(round(15 / df)))))
    return float(len(peaks) / (len(r) * df / 1000.0))


@feature("fourier")
def residual_std_db(P: Prepared) -> float:
    """Roughness of the long-term spectrum around its trend, in dB: decoder artifacts add fine ripple that a recorded acoustic mix averages away."""
    r, _ = P.residual
    return float("nan") if np.isnan(r).any() else float(np.std(r))


# ===================== named frequency bands =====================

@feature("bands")
def band_0_150_db(P: Prepared) -> float:
    """Share of energy below 150 Hz (dB re total); generators often under- or over-produce sub-bass compared with mastered human mixes."""
    return _band_rel_db(P, 0, 150)


@feature("bands")
def band_150_500_db(P: Prepared) -> float:
    """Share of energy in 150-500 Hz (dB re total), the low-mid body of instruments and voice, included as a spectral-balance reference."""
    return _band_rel_db(P, 150, 500)


@feature("bands")
def band_500_2k_db(P: Prepared) -> float:
    """Share of energy in 500 Hz-2 kHz (dB re total), where vocal fundamentals and most melody live, included as a spectral-balance reference."""
    return _band_rel_db(P, 500, 2000)


@feature("bands")
def band_2k_5k_db(P: Prepared) -> float:
    """Share of energy in 2-5 kHz (dB re total), the presence region, included as a spectral-balance reference."""
    return _band_rel_db(P, 2000, 5000)


@feature("bands")
def band_5k_8k_db(P: Prepared) -> float:
    """Share of energy in 5-8 kHz (dB re total), the band where vocoder limits distort vocal overtones and sibilance in AI vocals."""
    return _band_rel_db(P, 5000, 8000)


@feature("bands")
def band_8k_12k_db(P: Prepared) -> float:
    """Share of energy in 8-12 kHz (dB re total), the brilliance region, included as a spectral-balance reference."""
    return _band_rel_db(P, 8000, 12000)


@feature("bands")
def band_12k_18k_db(P: Prepared) -> float:
    """Share of energy in 12-18 kHz (dB re total), where real instruments put little energy and generators leave odd persistent energy."""
    return _band_rel_db(P, 12000, 18000)


@feature("bands")
def band_18k_24k_db(P: Prepared) -> float:
    """Share of energy from 18 kHz up to the band limit (dB re total; missing when the file stops below ~21 kHz): studio masters often carry air up here, many generators little."""
    return _band_rel_db(P, 18000, 24000)


@feature("bands")
def band_5k_8k_var_db(P: Prepared) -> float:
    """Frame-to-frame variability of the 5-8 kHz share in dB: real sibilants and breaths come and go, generated vocal bands are often smoother."""
    x = _band_frames_db(P, 5000, 8000)
    return float("nan") if x is None else float(np.std(x))


@feature("bands")
def band_12k_18k_var_db(P: Prepared) -> float:
    """Frame-to-frame variability of the 12-18 kHz share in dB: cymbals and hi-hats make it spiky in human mixes, a constant synthetic noise floor makes it flat."""
    x = _band_frames_db(P, 12000, 18000)
    return float("nan") if x is None else float(np.std(x))


@feature("bands")
def band_12k_18k_persistence_db(P: Prepared) -> float:
    """Gap between the quietest and loudest moments of the 12-18 kHz band (10th minus 90th percentile, dB): a small gap means energy that never goes away, typical of generator noise."""
    x = _band_frames_db(P, 12000, 18000)
    return float("nan") if x is None else float(np.percentile(x, 10) - np.percentile(x, 90))


@feature("bands")
def band_5k_8k_flatness(P: Prepared) -> float:
    """Spectral flatness inside 5-8 kHz: smeared, noise-like overtones from a vocoder are flatter than the harmonic overtones of a real voice."""
    return _band_flatness(P, 5000, 8000)


@feature("bands")
def band_12k_18k_flatness(P: Prepared) -> float:
    """Spectral flatness inside 12-18 kHz, below the band limit only (missing when the file stops below ~15 kHz): synthetic hiss is flat and stationary, real cymbals and air are uneven."""
    return _band_flatness(P, 12000, 18000)


# ===================== rolloff and flatness =====================

def _rolloff(P: Prepared, q: float) -> np.ndarray:
    """Per-frame rolloff over non-silent frames (silent frames would all read 0 Hz)."""
    f, S = P.stft_power
    S = S[:, P.active] if P.active.any() else S
    c = np.cumsum(S, axis=0) / (S.sum(axis=0, keepdims=True) + EPS)
    return f[np.argmax(c >= q, axis=0)]


FLAT_HI = 7000.0  # every source, even 16 kHz-native ones, has content up to here


def _fullband_flatness(P: Prepared) -> np.ndarray | None:
    """Per-frame flatness over a FIXED 20 Hz-7 kHz band (so it cannot encode the file's bandwidth), non-silent frames only."""
    if P.active.mean() < MIN_ACTIVE or P.valid_hi < FLAT_HI:
        return None
    f, S = P.stft_power
    a = P.active
    B = S[(f >= 20) & (f < FLAT_HI)][:, a]
    return _flat(B, 1e-8 * P.frame_total[a] / S.shape[0])


@feature("spectral")
def rolloff85_hz(P: Prepared) -> float:
    """Frequency below which 85% of the energy sits (mean over frames), a summary of overall brightness that shifts when a generator renders highs differently."""
    return float(_rolloff(P, 0.85).mean())


@feature("spectral")
def rolloff99_hz(P: Prepared) -> float:
    """Frequency below which 99% of the energy sits (mean over frames), which exposes the effective bandwidth a generator or codec actually produces."""
    return float(_rolloff(P, 0.99).mean())


@feature("spectral")
def rolloff99_std_hz(P: Prepared) -> float:
    """Frame-to-frame spread of the 99% rolloff: human mixes vary their bandwidth with the music, a fixed decoder bandwidth barely moves."""
    return float(_rolloff(P, 0.99).std())


@feature("spectral")
def flatness_mean(P: Prepared) -> float:
    """Average spectral flatness over 20 Hz-7 kHz (0 = pure tones, 1 = white noise); generated audio often carries a higher diffuse noise floor."""
    x = _fullband_flatness(P)
    return float("nan") if x is None else float(x.mean())


@feature("spectral")
def flatness_std(P: Prepared) -> float:
    """Frame-to-frame spread of flatness over 20 Hz-7 kHz: a real performance alternates tonal and noisy moments more than a smoothed generation."""
    x = _fullband_flatness(P)
    return float("nan") if x is None else float(x.std())


# ===================== codec band-limit edge =====================

@feature("bandlimit")
def bandlimit_edge_hz(P: Prepared) -> float:
    """Frequency of the steepest fall-off above 8 kHz, which reveals a brick-wall low-pass from a neural codec's output sample rate or from MP3/AAC encoding."""
    return P.edge[0]


@feature("bandlimit")
def bandlimit_drop_db(P: Prepared) -> float:
    """How many dB the spectrum falls across that edge within 1 kHz: a cliff points to a digital band limit, a gentle slope to natural roll-off."""
    return P.edge[1]


@feature("bandlimit")
def above_edge_db(P: Prepared) -> float:
    """Energy above the band-limit edge relative to the total (dB): near-silence above a cliff is a codec or decoder signature."""
    f, S = P.stft_power
    e = S[f >= P.edge[0]].sum(axis=0).mean()
    return float(10 * np.log10(e / P.frame_total.mean() + EPS))


# ===================== stereo =====================

def _mid_side(P: Prepared):
    L, R = P.stereo
    return (L + R) / 2, (L - R) / 2


@feature("stereo")
def stereo_width_db(P: Prepared) -> float:
    """Side-to-mid energy ratio in dB (-60 = mono): generators often produce narrow or artificially decorrelated stereo images unlike mixed multitrack recordings."""
    m, s = _mid_side(P)
    return float(np.clip(10 * np.log10((s ** 2).mean() / ((m ** 2).mean() + EPS) + EPS), -60, 20))


@feature("stereo")
def stereo_correlation(P: Prepared) -> float:
    """Correlation between left and right channels (1 = mono): a real stereo mix sits in a typical range set by panning and room sound."""
    L, R = P.stereo
    if np.allclose(L, R):
        return 1.0
    return float(np.corrcoef(L, R)[0, 1])


@feature("stereo")
def stereo_hf_coherence(P: Prepared) -> float:
    """Average left-right coherence between 4 and 16 kHz: decoders that generate channels jointly or upmix from mono leave unnaturally high or random high-frequency coherence."""
    L, R = P.stereo
    if np.allclose(L, R):
        return 1.0
    f, C = sps.coherence(L, R, fs=P.sr, nperseg=2048)
    band = (f >= 4000) & (f <= 16000)
    return float(C[band].mean())


@feature("stereo")
def stereo_width_hf_minus_lf_db(P: Prepared) -> float:
    """Stereo width above 4 kHz minus width below 500 Hz (dB): mixing engineers keep bass centred and spread highs, a pattern generators reproduce imperfectly."""
    m, s = _mid_side(P)
    if np.allclose(s, 0):
        return 0.0

    def width(lo, hi):
        sos = sps.butter(4, [max(lo, 20), min(hi, P.sr / 2 - 100)], btype="band", fs=P.sr, output="sos")
        mm, ss = sps.sosfilt(sos, m), sps.sosfilt(sos, s)
        return 10 * np.log10((ss ** 2).mean() / ((mm ** 2).mean() + EPS) + EPS)

    return float(np.clip(width(4000, 16000) - width(20, 500), -60, 60))


# ===================== optional: harmony (behind --harmony) =====================

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _key_profile_scores(chroma_vec: np.ndarray) -> np.ndarray:
    """24 Krumhansl-Kessler correlations (12 major, 12 minor)."""
    out = []
    for prof in (_MAJOR, _MINOR):
        for k in range(12):
            out.append(np.corrcoef(chroma_vec, np.roll(prof, k))[0, 1])
    return np.nan_to_num(np.array(out))


def _chords(P: Prepared):
    if "chords" in P.cache:
        return P.cache["chords"]
    import librosa

    y = librosa.resample(P.mono, orig_sr=P.sr, target_sr=22050)
    C = librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=2048)  # ~93 ms frames
    C = ndi.uniform_filter1d(C, 5, axis=1)  # ~0.5 s smoothing
    templates = []
    for root in range(12):
        for third in (4, 3):  # major, minor triads
            t = np.zeros(12)
            t[[root, (root + third) % 12, (root + 7) % 12]] = 1
            templates.append(t / np.linalg.norm(t))
    T = np.array(templates)
    Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + EPS)
    labels = np.argmax(T @ Cn, axis=0)
    key = int(np.argmax(_key_profile_scores(C.mean(axis=1))))
    P.cache["chords"] = (C, labels, key)
    return P.cache["chords"]


def _diatonic_triads(key: int) -> set[int]:
    tonic, minor = key % 12, key >= 12
    if minor:  # natural minor: i, III, iv, v, VI, VII (+ V from harmonic minor)
        spec = [(0, 3), (3, 4), (5, 3), (7, 3), (7, 4), (8, 4), (10, 4)]
    else:  # I, ii, iii, IV, V, vi
        spec = [(0, 4), (2, 3), (4, 3), (5, 4), (7, 4), (9, 3)]
    return {((tonic + r) % 12) * 2 + (0 if q == 4 else 1) for r, q in spec}


@feature("harmony")
def harm_nondiatonic_rate(P: Prepared) -> float:
    """Fraction of time the estimated chord is outside the estimated key: a qualitative MuseBench analysis claims generated tracks slip in non-diatonic chords, though corpus studies of Suno and Udio do not support it."""
    _, labels, key = _chords(P)
    allowed = _diatonic_triads(key)
    return float(np.mean([lab not in allowed for lab in labels]))


@feature("harmony")
def harm_chord_change_rate(P: Prepared) -> float:
    """Chord changes per second: implausibly jittery or static harmony is a possible sign of generation, though it varies a lot by genre."""
    _, labels, _ = _chords(P)
    dur = len(P.mono) / P.sr
    return float(np.sum(labels[1:] != labels[:-1]) / dur)


@feature("harmony")
def harm_key_instability(P: Prepared) -> float:
    """Fraction of 2.5-second windows whose best-fitting key differs from the excerpt's key, a proxy for the abrupt key shifts claimed (qualitatively, untested in corpus studies) for generated music."""
    C, _, key = _chords(P)
    w = 27  # ~2.5 s of 93 ms frames
    if C.shape[1] < 2 * w:
        return 0.0
    keys = [int(np.argmax(_key_profile_scores(C[:, i:i + w].mean(axis=1)))) for i in range(0, C.shape[1] - w + 1, w)]
    return float(np.mean([k != key for k in keys]))


DEFAULT_GROUPS = ("fourier", "bands", "spectral", "bandlimit", "stereo")


def names(groups=DEFAULT_GROUPS) -> list[str]:
    return [n for n, (_, g) in REGISTRY.items() if g in groups]


def extract(stereo: np.ndarray, sr: int = SR, groups=DEFAULT_GROUPS, source_channels: int | None = None) -> dict[str, float]:
    """Compute all features in ``groups`` for one excerpt. ``stereo`` is [2, n] or mono [n]."""
    x = np.asarray(stereo, dtype=np.float32)
    if x.ndim == 1:
        x = np.stack([x, x])
        source_channels = source_channels or 1
    if sr != SR:
        raise ValueError(f"features expect {SR} Hz input, got {sr}")
    P = Prepared(stereo=x, sr=sr, source_channels=source_channels or 2)
    out = {}
    for n in names(groups):
        try:
            out[n] = float(REGISTRY[n][0](P))
        except Exception:
            out[n] = float("nan")
    return out
