"""Find clips whose MONO mix is silent (e.g. phase-inverted stereo, R = -L) and drop them from the manifest.

    python scripts/check_mono.py

All models listen in mono, so for them such a clip is digital silence. prepare.py now
rejects these itself; this script cleans an existing manifest.
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO  # noqa: E402

MONO_FLOOR_DB = -55.0


def _mono_db(path: str) -> float:
    import soundfile as sf

    y, _ = sf.read(str(DATA / path), dtype="float32", always_2d=True)
    return float(20 * np.log10(np.sqrt(np.mean(y.mean(axis=1).astype(np.float64) ** 2)) + 1e-12))


def main() -> None:
    import pandas as pd

    man = pd.read_csv(REPO / "data" / "manifest.csv")
    with Pool(6) as pool:
        man["mono_rms_db"] = pool.map(_mono_db, man.path, chunksize=32)
    bad = man[man.mono_rms_db < MONO_FLOOR_DB]
    print(f"{len(bad)} clips with a silent mono mix:")
    print(bad[["clip_id", "source", "rms_db", "mono_rms_db"]].to_string(index=False))
    man = man[man.mono_rms_db >= MONO_FLOOR_DB].round({"mono_rms_db": 1})
    man.to_csv(REPO / "data" / "manifest.csv", index=False)
    print(f"manifest now has {len(man)} clips")


if __name__ == "__main__":
    main()
