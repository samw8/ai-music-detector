"""Step 5 - compute the interpretable features and Deezer fakeprints for every excerpt.

    python scripts/extract_features.py              # default feature groups
    python scripts/extract_features.py --harmony    # also the optional chord/key features (slower)
    python scripts/extract_features.py --codec mp3  # codec-equalised copy: every clip through MP3 192 kbps first

Output: data/features.csv (one row per clip, one column per feature) and
data/fakeprints.npy + data/fakeprint_ids.txt (one 3755-value vector per clip).
With --codec mp3 the files are features_mp3.csv / fakeprints_mp3.npy / fakeprint_mp3_ids.txt.

Why the codec-equalised copy: the human datasets are mostly MP3 while several AI
datasets are lossless WAV, so "has MP3 artefacts" could pass for "is human". Passing
BOTH classes through the same encoder removes that shortcut.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker process; parallelism comes from the pool

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO  # noqa: E402
from aimd import features as F  # noqa: E402
from aimd.fakeprint import fakeprint  # noqa: E402

GROUPS = list(F.DEFAULT_GROUPS)


def featurize(stereo: np.ndarray, sr: int, groups=None) -> tuple[dict, np.ndarray]:
    """Features + fakeprint for one [2, n] 48 kHz excerpt (also used by predict.py and the robustness test)."""
    feats = F.extract(stereo, sr, groups=groups or GROUPS)
    _, fp = fakeprint(stereo, sr)
    return feats, fp


def codec_equalise(stereo: np.ndarray, sr: int) -> np.ndarray:
    """The shared encoder used by the codec-equalised variant (both classes, training and test)."""
    from aimd.augment import mp3

    return np.clip(mp3(stereo, sr, kbps=192), -1, 1).astype(np.float32)


def _work(args):
    import soundfile as sf

    clip_id, path, groups, codec = args
    y, sr = sf.read(str(DATA / path), dtype="float32", always_2d=True)
    x = y.T
    if codec == "mp3":
        x = codec_equalise(x, sr)
    feats, fp = featurize(x, sr, groups)
    return clip_id, feats, fp


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--harmony", action="store_true", help="add the optional chord/key irregularity features")
    ap.add_argument("--codec", choices=["mp3"], default=None, help="codec-equalised variant")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    groups = GROUPS + (["harmony"] if a.harmony else [])
    sfx = f"_{a.codec}" if a.codec else ""
    man = pd.read_csv(REPO / "data" / "manifest.csv")
    feat_csv, fp_npy, fp_ids = (REPO / "data" / f"features{sfx}.csv", REPO / "data" / f"fakeprints{sfx}.npy",
                                REPO / "data" / f"fakeprint{sfx}_ids.txt")
    rows, fps, ids = [], [], []
    if feat_csv.exists() and fp_ids.exists() and not a.harmony:  # incremental: keep clips already done
        old = pd.read_csv(feat_csv)
        old_ids = fp_ids.read_text().split("\n")
        keep = set(old.clip_id) & set(man.clip_id) & set(old_ids)
        old_fp = np.load(fp_npy)
        pos = {c: i for i, c in enumerate(old_ids)}
        old = old[old.clip_id.isin(keep)]
        rows = old.to_dict("records")
        ids = list(old.clip_id)
        fps = [old_fp[pos[c]].astype(np.float32) for c in ids]
    done = set(ids)
    jobs = [(r.clip_id, r.path, groups, a.codec) for r in man.itertuples() if r.clip_id not in done]
    print(f"{len(jobs)} clips to do ({len(done)} already done), feature groups: {groups}, codec: {a.codec or 'native'}", flush=True)
    with Pool(a.workers) as pool:
        for i, (cid, feats, fp) in enumerate(pool.imap(_work, jobs, chunksize=8)):
            rows.append({"clip_id": cid, **feats})
            fps.append(fp)
            ids.append(cid)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)
    pd.DataFrame(rows).to_csv(feat_csv, index=False)
    np.save(fp_npy, np.stack(fps).astype(np.float16))
    fp_ids.write_text("\n".join(ids))
    print(f"wrote data/features{sfx}.csv ({len(rows)} x {len(rows[0]) - 1}) and data/fakeprints{sfx}.npy")


if __name__ == "__main__":
    main()
