"""Step 6 - log-mel spectrograms for the ResNet18 model (cached so training doesn't redo them).

    python scripts/melspec.py

Resumable: progress is saved every 500 clips. Rebuilds from scratch if the manifest changed.
Output: data/logmel.npy (clips x 128 x 501, float16) and data/logmel_ids.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO  # noqa: E402
from aimd.cnn import SETTINGS, logmel  # noqa: E402


def main() -> None:
    import pandas as pd
    import soundfile as sf
    import torch

    torch.set_num_threads(2)
    man = pd.read_csv(REPO / "data" / "manifest.csv")
    first = logmel(np.zeros(480000, dtype=np.float32))
    path, ids_path, prog = REPO / "data" / "logmel.npy", REPO / "data" / "logmel_ids.txt", REPO / "data" / "logmel_progress.txt"
    settings_path = REPO / "data" / "logmel_settings.txt"  # a cache built with other spectrogram settings is rebuilt
    shape = (len(man), *first.shape)
    start = 0
    planned = "\n".join(man.clip_id)
    if (path.exists() and prog.exists() and ids_path.exists() and ids_path.read_text() == planned
            and settings_path.exists() and settings_path.read_text() == SETTINGS):
        out = np.lib.format.open_memmap(path, mode="r+")
        if out.shape == shape:
            start = int(prog.read_text() or 0)
    if start == 0:
        out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        ids_path.write_text(planned)
        settings_path.write_text(SETTINGS)
        prog.write_text("0")
    print(f"{len(man)} clips, starting at {start}", flush=True)
    for i, r in enumerate(man.itertuples()):
        if i < start:
            continue
        y, sr = sf.read(str(DATA / r.path), dtype="float32", always_2d=True)
        out[i] = logmel(y.mean(axis=1))
        if (i + 1) % 500 == 0 or i + 1 == len(man):
            out.flush()
            prog.write_text(str(i + 1))
            print(f"  {i + 1}/{len(man)}", flush=True)
    print(f"wrote data/logmel.npy {out.shape}")


if __name__ == "__main__":
    main()
