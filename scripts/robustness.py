"""Step 10 - robustness: re-score the matched test set after everyday audio edits.

    python scripts/robustness.py            # default: 800 test clips (400 human, 400 AI)
    python scripts/robustness.py --n 2000

Edits (the brief's list): MP3 128 kbps, OGG, AAC, +/-2 semitone pitch shift, 0.9x and
1.1x speed, loudness normalisation to -14 LUFS, plus tempo changes that keep pitch.
Unlike MuseBench (which edits only the AI tracks), the edits are applied to human
AND AI clips, so we see both missed AI (FNR) and newly accused humans (FPR).

Output: results/robustness.csv and results/figures/robustness.png (drawn by evaluate.fig_robustness, the one place
that figure is drawn, so it always has the same short, unique detector labels)
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/, for evaluate.fig_robustness (any working folder)

from aimd import DATA, REPO, RESULTS  # noqa: E402
from aimd.augment import BRIEF_SUITE, apply  # noqa: E402

SR = 48000


def _init_worker():
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        os.environ["OMP_NUM_THREADS"] = "1"


def _transform(args):
    """Apply one edit to one clip and compute its hand-made detector inputs here, in the worker, in parallel."""
    import soundfile as sf

    from aimd.audio import to_pcm16
    from aimd.infer import hand_reps

    path, name, feature_names, tags = args
    y, sr = sf.read(str(DATA / path), dtype="float32", always_2d=True)
    z = to_pcm16(np.clip(apply(name, y.T.copy(), sr, BRIEF_SUITE), -1, 1).astype(np.float32))
    return z, hand_reps(z, feature_names, tags)


def main() -> None:
    import pandas as pd

    from aimd.infer import Detectors
    from aimd.metrics import summarize
    from aimd.splits import base_split, load_table

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    df = load_table(REPO)
    test = df[base_split(df) == "test"]
    rng = np.random.default_rng(0)
    pick = []
    for lab in (0, 1):
        g = test[test.label == lab]
        pick.append(g.iloc[rng.permutation(len(g))[: a.n // 2]])
    sel = pd.concat(pick)
    print(f"{len(sel)} matched-test clips ({(sel.label == 0).sum()} human, {(sel.label == 1).sum()} AI) x {len(BRIEF_SUITE)} conditions", flush=True)

    det = Detectors()
    out_csv = RESULTS / "robustness.csv"
    rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    have = {(r["transform"], r["detector"]) for r in rows}  # resumable per (edit, detector): a retrained model is re-tested alone
    with Pool(a.workers, initializer=_init_worker) as pool:
        for name in BRIEF_SUITE:
            todo = [m for m in det.models if (name, m) not in have]
            if not todo:
                continue
            d = det if len(todo) == len(det.models) else Detectors(todo)
            out = pool.map(_transform, [(p, name, d.feature_names, d.hand_tags()) for p in sel.path], chunksize=4)
            reps = d.represent([z for z, _ in out], hand=[h for _, h in out])
            preds = d.predict(reps)
            for m, p in preds.items():
                s = summarize(sel.label.to_numpy(), p)
                rows.append({"transform": name, "detector": m, **{k: s[k] for k in ("balanced_accuracy", "fnr", "fpr", "auc", "eer")}})
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f"  {name}: " + ", ".join(f"{m} bal.acc {summarize(sel.label, p)['balanced_accuracy']:.3f}" for m, p in preds.items()), flush=True)

    # the figure is drawn in one place, with short unique detector labels, from the CSV just written
    from evaluate import fig_robustness

    fig_robustness()
    print(f"wrote {out_csv} and results/figures/robustness.png")


if __name__ == "__main__":
    main()
