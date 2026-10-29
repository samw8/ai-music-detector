"""Real-world audio manipulations used for robustness testing (and training augmentation).

The Deezer paper (Afchar et al. 2025) shows detectors break under simple edits
like pitch shifting or re-encoding, so every model here is evaluated against
this list. Each transform takes and returns (mono float32 signal, sample rate)
at the same sample rate and roughly the same length.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import scipy.signal as sps

from .audio import _resample, ffmpeg_exe


def _codec_roundtrip(y: np.ndarray, sr: int, args: list[str], ext: str) -> np.ndarray:
    """Encode and decode with ffmpeg. y is mono [n] or stereo [2, n]; returns the same shape."""
    import soundfile as sf

    ch = 1 if y.ndim == 1 else y.shape[0]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        dst = Path(tmp) / f"out.{ext}"
        sf.write(str(src), np.clip(y if ch == 1 else y.T, -1, 1), sr, subtype="PCM_16")
        exe = ffmpeg_exe()
        subprocess.run([exe, "-nostdin", "-loglevel", "error", "-y", "-i", str(src), *args, str(dst)], check=True)
        out = subprocess.run(
            [exe, "-nostdin", "-loglevel", "error", "-i", str(dst), "-ac", str(ch), "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True, check=True,
        ).stdout
    z = np.frombuffer(out, dtype=np.float32).copy()
    n = y.shape[-1]
    if ch == 1:
        return _align(z, n)
    return np.stack([_align(c, n) for c in z.reshape(-1, ch).T])


def _align(z: np.ndarray, n: int) -> np.ndarray:
    return z[:n] if len(z) >= n else np.pad(z, (0, n - len(z)))


def _extend(y: np.ndarray, n_needed: int) -> np.ndarray:
    """Lengthen a clip by mirrored repeats (no digital silence) so shortening edits can be cropped, never zero-padded."""
    if len(y) >= n_needed:
        return y
    reps = int(np.ceil(n_needed / len(y)))
    return np.concatenate([y if i % 2 == 0 else y[::-1] for i in range(reps)])[:n_needed]


def mp3(y, sr, kbps: int = 128):
    return _codec_roundtrip(y, sr, ["-c:a", "libmp3lame", "-b:a", f"{kbps}k"], "mp3")


def aac(y, sr, kbps: int = 128):
    return _codec_roundtrip(y, sr, ["-c:a", "aac", "-b:a", f"{kbps}k"], "m4a")


def opus(y, sr, kbps: int = 64):
    return _codec_roundtrip(y, sr, ["-c:a", "libopus", "-b:a", f"{kbps}k"], "opus")


def ogg(y, sr, quality: int = 3):
    """Ogg Vorbis at quality q3 (~112 kbps), as in MuseBench."""
    return _codec_roundtrip(y, sr, ["-c:a", "libvorbis", "-q:a", str(quality)], "ogg")


def loudness(y, sr, lufs: float = -14.0):
    """Integrated-loudness normalisation (what Spotify/YouTube do on playback)."""
    import pyloudnorm

    meter = pyloudnorm.Meter(sr)
    cur = meter.integrated_loudness((y.T if y.ndim == 2 else y).astype(np.float64))
    if not np.isfinite(cur):
        return y
    out = y * 10 ** ((lufs - cur) / 20)
    peak = np.max(np.abs(out))
    return (out / peak * 0.999 if peak > 1 else out).astype(np.float32)


def speed(y, sr, factor: float = 1.1):
    """Playback speed change by resampling ('sped up' / 'slowed' edits): tempo and pitch move together."""
    n = len(y)
    src = _extend(y, int(np.ceil(n * factor)) + 1024)
    return _align(_resample(src, int(round(sr * factor)), sr), n)


def resample_roundtrip(y, sr, mid_sr: int = 22050):
    return _align(_resample(_resample(y, sr, mid_sr), mid_sr, sr), len(y))


def tape_speed(y, sr, semitones: float = 1.0):
    """Speed + pitch change by resampling (like a DJ pitch fader). Moves every frequency, artifacts included."""
    factor = 2 ** (semitones / 12)
    n = len(y)
    return _align(_resample(_extend(y, int(np.ceil(n * factor)) + 1024), int(round(sr * factor)), sr), n)


def pitch_shift(y, sr, semitones: float = 1.0):
    """Pitch shift keeping tempo (phase vocoder + resampling), what most DAWs do."""
    import librosa

    return _align(librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones).astype(np.float32), len(y))


def time_stretch(y, sr, rate: float = 1.05):
    import librosa

    n = len(y)
    src = _extend(y, int(np.ceil(n * rate)) + 4096)
    return _align(librosa.effects.time_stretch(src, rate=rate).astype(np.float32), n)


def add_noise(y, sr, snr_db: float = 30.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    p = np.mean(y.astype(np.float64) ** 2) + 1e-12
    noise = rng.standard_normal(len(y)) * np.sqrt(p / 10 ** (snr_db / 10))
    return (y + noise).astype(np.float32)


def lowpass(y, sr, cutoff: float = 8000.0):
    sos = sps.butter(8, cutoff, btype="low", fs=sr, output="sos")
    return sps.sosfiltfilt(sos, y).astype(np.float32)


def highpass(y, sr, cutoff: float = 200.0):
    sos = sps.butter(4, cutoff, btype="high", fs=sr, output="sos")
    return sps.sosfiltfilt(sos, y).astype(np.float32)


def eq_boost(y, sr, freq: float = 3000.0, gain_db: float = 6.0, q: float = 1.0):
    """Peaking EQ (RBJ cookbook biquad)."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return sps.lfilter(b, den, y).astype(np.float32)


def gain(y, sr, db: float = -6.0):
    return (y * 10 ** (db / 20)).astype(np.float32)


def hard_clip(y, sr, drive_db: float = 6.0):
    return np.clip(y * 10 ** (drive_db / 20), -1, 1).astype(np.float32)


def reverb(y, sr, rt60: float = 0.6, wet: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet_sig = sps.fftconvolve(y, ir)[: len(y)]
    return ((1 - wet) * y + wet * wet_sig).astype(np.float32)


# Named suite used by the robustness benchmark: name -> (function, kwargs)
SUITE: dict[str, tuple] = {
    "clean": (None, {}),
    "mp3_320": (mp3, {"kbps": 320}),
    "mp3_128": (mp3, {"kbps": 128}),
    "mp3_64": (mp3, {"kbps": 64}),
    "aac_128": (aac, {"kbps": 128}),
    "opus_64": (opus, {"kbps": 64}),
    "resample_22k": (resample_roundtrip, {"mid_sr": 22050}),
    "resample_16k": (resample_roundtrip, {"mid_sr": 16000}),
    "pitch_+1": (pitch_shift, {"semitones": 1}),
    "pitch_-2": (pitch_shift, {"semitones": -2}),
    "tape_+0.5": (tape_speed, {"semitones": 0.5}),
    "stretch_1.05": (time_stretch, {"rate": 1.05}),
    "noise_30dB": (add_noise, {"snr_db": 30}),
    "noise_20dB": (add_noise, {"snr_db": 20}),
    "lowpass_8k": (lowpass, {"cutoff": 8000}),
    "highpass_200": (highpass, {"cutoff": 200}),
    "eq_+6dB_3k": (eq_boost, {"freq": 3000, "gain_db": 6}),
    "gain_-6dB": (gain, {"db": -6}),
    "clip_+6dB": (hard_clip, {"drive_db": 6}),
    "reverb": (reverb, {}),
}


# The brief's robustness list (MuseBench's post-processing set), applied to BOTH classes.
BRIEF_SUITE: dict[str, tuple] = {
    "clean": (None, {}),
    "MP3 128 kbps": (mp3, {"kbps": 128}),
    "OGG Vorbis q3": (ogg, {"quality": 3}),
    "AAC 128 kbps": (aac, {"kbps": 128}),
    "pitch +2 semitones": (pitch_shift, {"semitones": 2}),
    "pitch -2 semitones": (pitch_shift, {"semitones": -2}),
    "speed 0.9x": (speed, {"factor": 0.9}),
    "speed 1.1x": (speed, {"factor": 1.1}),
    "tempo 0.9x (pitch kept)": (time_stretch, {"rate": 0.9}),
    "tempo 1.1x (pitch kept)": (time_stretch, {"rate": 1.1}),
    "loudness -14 LUFS": (loudness, {"lufs": -14.0}),
    # not in the brief: old, phone-quality or low-sample-rate recordings. Almost every clip stored at 16 or 32 kHz
    # is AI in this data, so these test whether a detector treats missing high frequencies as a sign of AI.
    "bandwidth 8 kHz (via 16 kHz)": (resample_roundtrip, {"mid_sr": 16000}),
    "bandwidth 16 kHz (via 32 kHz)": (resample_roundtrip, {"mid_sr": 32000}),
}


def apply(name: str, y: np.ndarray, sr: int, suite: dict = None) -> np.ndarray:
    """Apply a named transform to a mono [n] or stereo [2, n] signal."""
    fn, kw = (suite or {**SUITE, **BRIEF_SUITE})[name]
    if fn is None:
        return y
    if y.ndim == 2 and fn in (mp3, aac, opus, ogg, loudness):
        return fn(y, sr, **kw)  # these handle stereo jointly (joint-stereo coding, overall loudness)
    if y.ndim == 2:
        return np.stack([fn(c, sr, **kw) for c in y])
    return fn(y, sr, **kw)
