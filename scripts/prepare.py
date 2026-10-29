"""Step 2 - cut one 10-second excerpt from every downloaded track and build the manifest.

    python scripts/prepare.py

Every track, human or AI, goes through the same decoder (ffmpeg) and resampler
(soxr) to 48 kHz stereo, so the two classes are handled identically. The
excerpt starts 40% of the way into the track (skipping intros); if that spot is
near-silent another position is tried. Output:

    %USERPROFILE%\\.aimd\\data\\clips\\<dataset>\\<id>.flac
    data\\manifest.csv   (one row per excerpt, with provenance tier and format info)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker process; parallelism comes from the pool

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO  # noqa: E402
from aimd.audio import decode_stereo, probe  # noqa: E402,F401  (probe is re-exported for older callers)
from aimd.sources import SOURCES  # noqa: E402

SR = 48000
SEG = 10.0
POSITIONS = (0.4, 0.2, 0.6, 0.8, 0.0)
MANIFEST = REPO / "data" / "manifest.csv"


def decode(path: Path, offset: float, dur: float, sr_native: int) -> np.ndarray:
    return decode_stereo(path, offset, dur, sr_native, SR)


def rms_db(y: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-12))


def work(job: tuple) -> dict | None:
    import soundfile as sf

    src_name, row = job
    s = SOURCES[src_name]
    path = s.raw / row["file"]
    if not path.exists():
        return None
    clip_id = f"{src_name}-{hashlib.md5(row['track_id'].encode()).hexdigest()[:10]}"
    out = DATA / "clips" / src_name / f"{clip_id}.flac"
    info = probe(path)
    if info["duration"] < SEG * 0.8:
        return None
    best = None
    for pos in POSITIONS:
        off = max(0.0, pos * (info["duration"] - SEG))
        try:
            y = decode(path, off, SEG, info["orig_sr"])
        except subprocess.CalledProcessError:
            continue
        if y.shape[1] < SR * SEG * 0.8:
            continue
        level = rms_db(y)
        if best is None or level > best[2]:
            best = (y, off, level)
        if level > -40:
            break
    if best is None or best[2] < -55 or rms_db(best[0].mean(axis=0)) < -55:
        return None  # silent, or silent once mixed to mono (phase-inverted stereo): every model listens in mono
    y, off, level = best
    n = int(SR * SEG)
    y = y[:, :n] if y.shape[1] >= n else np.pad(y, ((0, 0), (0, n - y.shape[1])))
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        sf.write(str(out), np.clip(y.T, -1, 1), SR, subtype="PCM_16")
    return {
        "clip_id": clip_id, "source": src_name, "label": s.label,
        "generator": row.get("generator") or ("human" if s.label == 0 else src_name),
        "family": s.family, "tier": row.get("tier") or s.tier, "artist_id": row.get("artist_id") or row["track_id"],
        "track_id": row["track_id"], "genre_meta": row.get("genre", ""), "tags": str(row.get("tags", ""))[:300],
        "vocal_hint": "" if row.get("vocal_hint") is None else int(bool(row["vocal_hint"])),
        "created": row.get("created", ""), "license": row.get("license", s.license),
        "path": str(out.relative_to(DATA)).replace("\\", "/"), "offset": round(off, 2), "rms_db": round(level, 1),
        "stereo_identical": int(np.allclose(y[0], y[1], atol=1e-4)), **info,
    }


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    names = [n for n in a.only.split(",") if n] or [n for n in SOURCES if (SOURCES[n].raw / "meta.jsonl").exists()]
    jobs = []
    for n in names:
        meta = SOURCES[n].raw / "meta.jsonl"
        if not meta.exists():
            print(f"skip {n}: not downloaded")
            continue
        jobs += [(n, json.loads(line)) for line in open(meta, encoding="utf-8")]
    print(f"{len(jobs)} tracks from {len(names)} datasets -> 10 s excerpts", flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            if r:
                rows.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows).sort_values(["label", "source", "clip_id"])
    old = MANIFEST.exists() and pd.read_csv(MANIFEST)
    if old is not False and a.only:
        df = pd.concat([old[~old.source.isin(names)], df])
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST, index=False)
    print(f"\nwrote {MANIFEST} with {len(df)} excerpts")
    print(df.groupby(["label", "source"]).size().to_string())
    print("\nFormat check (a big difference between human and AI here is a confound):")
    print(df.groupby("label")[["orig_sr", "orig_ch", "kbps"]].describe().T.to_string())


if __name__ == "__main__":
    main()
