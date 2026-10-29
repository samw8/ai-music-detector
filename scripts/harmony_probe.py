"""Step 12 (optional) - test the "harmonic grammar" claim on a sample (the harmony features are too slow for every clip on CPU).

    python scripts/harmony_probe.py            # 100 clips per dataset

The brief suggested generated tracks "insert non-diatonic chords and abrupt key shifts".
Published corpus studies found the opposite for Suno/Udio (as diatonic or more, with stock
progressions; Cros Vila et al. TISMIR 2025, Dalmazzo et al. TISMIR 2026). This measures
it on our data: for each harmony feature, human vs AI distributions and how well the
feature alone separates them (AUC; 0.5 = no information, below 0.5 = AI scores LOWER).

Not implemented, deliberately: "physical playability" checks (>10 simultaneous piano
notes, guitar notes outside E2-E6, wind lines without breaths). They need reliable note
transcription AND instrument identification inside dense mixes, which no CPU-friendly
open model provides; a fake version would produce confident nonsense. No published
detector uses them either.

Output: results/harmony_probe.csv and results/harmony_probe.md
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO, RESULTS  # noqa: E402
from aimd import features as F  # noqa: E402


def _work(args):
    import soundfile as sf

    cid, path = args
    y, sr = sf.read(str(DATA / path), dtype="float32", always_2d=True)
    return cid, F.extract(y.T, sr, groups=("harmony",))


def main() -> None:
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from aimd.splits import load_table

    df = load_table(REPO)
    # (not groupby().apply: recent pandas drops the grouping column from its result)
    sample = pd.concat([g.sample(min(len(g), 100), random_state=0) for _, g in df.groupby("source")])
    print(f"{len(sample)} clips ({sample.label.eq(0).sum()} human, {sample.label.eq(1).sum()} AI)", flush=True)
    with Pool(8) as pool:
        res = pool.map(_work, list(zip(sample.clip_id, sample.path)), chunksize=4)
    H = pd.DataFrame([{"clip_id": c, **f} for c, f in res]).merge(sample[["clip_id", "label", "source", "model"]], on="clip_id")
    H.to_csv(RESULTS / "harmony_probe.csv", index=False)
    lines = ["# Harmony probe: do AI tracks break harmonic conventions?", "",
             f"{len(H)} ten-second clips (100 per dataset). AUC > 0.5: AI scores higher; < 0.5: AI scores lower.", "",
             "| Feature | Human median | AI median | AUC (AI vs human) | What it measures |", "|---|---|---|---|---|"]
    for n in F.names(("harmony",)):
        h, a = H[H.label == 0][n].dropna(), H[H.label == 1][n].dropna()
        auc = roc_auc_score(H.label[H[n].notna()], H[n].dropna())
        lines.append(f"| `{n}` | {h.median():.3f} | {a.median():.3f} | {auc:.3f} | {F.doc(n)} |")
    lines += ["", "Per generator (median non-diatonic rate):", "",
              H.groupby("model").harm_nondiatonic_rate.median().round(3).to_frame().to_markdown()]
    (RESULTS / "harmony_probe.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
