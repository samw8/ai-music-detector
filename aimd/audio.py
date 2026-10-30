"""Audio loading and segmentation.

Everything in this project works on mono float32 numpy arrays in [-1, 1].
Files are decoded with soundfile when possible (wav/flac/ogg/mp3) and fall
back to the ffmpeg binary bundled with imageio-ffmpeg for anything else
(m4a, aac, mp4, webm, ...), so no system ffmpeg install is needed.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".mp4", ".webm", ".aiff", ".aif"}


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    import soxr

    return soxr.resample(y, sr_in, sr_out, quality="HQ").astype(np.float32)


def _ffmpeg_decode(path: Path, sr: int | None, offset: float, duration: float | None) -> tuple[np.ndarray, int]:
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error"]
    if offset:
        cmd += ["-ss", f"{offset:.3f}"]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1"]
    if sr:
        cmd += ["-ar", str(sr)]
    else:
        sr = 44100
        cmd += ["-ar", str(sr)]
    cmd += ["-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy(), sr


def load(path: str | Path, sr: int | None = None, offset: float = 0.0, duration: float | None = None) -> tuple[np.ndarray, int]:
    """Load a file as mono float32. If ``sr`` is given, resample to it."""
    path = Path(path)
    try:
        info = sf.info(str(path))
        start = int(offset * info.samplerate)
        frames = int(duration * info.samplerate) if duration else -1
        y, file_sr = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        return _ffmpeg_decode(path, sr, offset, duration)
    if sr is None:
        return y, file_sr
    return _resample(y, file_sr, sr), sr


_STREAM = re.compile(r"Audio: (\w+)[^,]*, (\d+) Hz, ([^,]+),[^,]*(?:, (\d+) kb/s)?")
_DUR = re.compile(r"Duration: (\d+):(\d+):([\d.]+)")


def probe(path: str | Path) -> dict:
    """Codec, sample rate, channels, bitrate and duration from ffmpeg's banner (no ffprobe needed)."""
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace").stderr
    out = {"codec": "", "orig_sr": 0, "orig_ch": 0, "kbps": 0, "duration": 0.0}
    m = _STREAM.search(err)
    if m:
        out["codec"], out["orig_sr"] = m.group(1), int(m.group(2))
        layout = m.group(3).strip()
        out["orig_ch"] = 1 if layout == "mono" else 2 if layout == "stereo" else int(re.sub(r"\D", "", layout) or 2)
        out["kbps"] = int(m.group(4) or 0)
    d = _DUR.search(err)
    if d:
        out["duration"] = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
    return out


def decode_stereo(path: str | Path, offset: float, dur: float | None, sr_native: int, sr: int = 48000,
                  max_frames: int | None = None) -> np.ndarray:
    """THE decoding path for every clip, training and prediction alike: ffmpeg at the native rate,
    upmixed to 2 channels, then soxr HQ to ``sr``. Returns [2, n] float32.

    ``max_frames`` (used by the web page, never in training): refuse a file that yields more native-rate frames
    than that, i.e. whose real sample rate is higher than ``sr_native`` (ffmpeg's banner can be imitated by tags)."""
    cmd = [ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-ss", f"{offset:.3f}"]
    if dur:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "2"]
    if max_frames:
        cmd += ["-fs", str(8 * max_frames + 8)]  # ffmpeg stops writing just past the limit
    cmd += ["-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if max_frames and len(raw) > 8 * max_frames:
        raise ValueError("the file holds more audio than its stated sample rate allows")
    y = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    if sr_native and sr_native != sr:
        y = np.stack([_resample(c, sr_native, sr) for c in y])
    return y


def to_pcm16(y: np.ndarray) -> np.ndarray:
    """Round to the 16-bit grid exactly as the stored training clips were (libsndfile writes x*32768 with clipping,
    reads k/32768), so applying it to a stored clip changes nothing."""
    return (np.clip(np.round(np.asarray(y, np.float64) * 32768), -32768, 32767) / 32768).astype(np.float32)


def duration_of(path: str | Path) -> float:
    try:
        return sf.info(str(path)).duration
    except Exception:
        y, sr = _ffmpeg_decode(Path(path), 16000, 0.0, None)
        return len(y) / sr


def fix_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def segments(y: np.ndarray, sr: int, seg_seconds: float, hop_seconds: float | None = None, min_fraction: float = 0.5) -> list[np.ndarray]:
    """Split a signal into fixed-length windows (last partial window padded if long enough)."""
    n = int(seg_seconds * sr)
    hop = int((hop_seconds or seg_seconds) * sr)
    if len(y) <= n:
        return [fix_length(y, n)]
    out = []
    for start in range(0, len(y) - int(n * min_fraction) + 1, hop):
        out.append(fix_length(y[start:start + n], n))
    return out


def is_silent(y: np.ndarray, thresh_db: float = -50.0) -> bool:
    rms = np.sqrt(np.mean(y.astype(np.float64) ** 2) + 1e-12)
    return 20 * np.log10(rms) < thresh_db


def peak_normalize(y: np.ndarray, peak: float = 0.95) -> np.ndarray:
    m = np.max(np.abs(y)) if len(y) else 0.0
    return y if m < 1e-8 else (y * (peak / m)).astype(np.float32)


def write(path: str | Path, y: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1, 1), sr, subtype="PCM_16")


def list_audio(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
