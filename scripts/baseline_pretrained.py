"""Step 9 - published, already-trained detectors scored as they come (no retraining).

    python scripts/baseline_pretrained.py

    spectttra_alpha5s   SONICS SpecTTTra-alpha, 5 s model (Rahman et al., ICLR 2025; MIT weights,
                        trained on Suno v3.5 / Udio-130 vs YouTube songs). Two 5 s windows per clip, averaged.
    lofcz_fakeprint     a public logistic regression on Deezer's fakeprint at 16 kHz (huggingface.co/lofcz/ai-music-detector,
                        MIT; trained on FMA + SONICS + private data).

They are evaluated on exactly the same test clips as our models, so they appear in
the same tables. Not everything is unseen to them: SpecTTTra was trained on Suno v3.5 /
Udio-130, and lofcz on FMA-medium (which includes fma_small) plus SONICS, so every FMA
clip is in-domain for lofcz - the results table marks those cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO, RESULTS  # noqa: E402
from aimd.audio import _resample  # noqa: E402


def lofcz_scorer():
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file
    from scipy.ndimage import minimum_filter1d

    d = load_file(hf_hub_download("lofcz/ai-music-detector", "model.safetensors"))
    w, b = d["weights"].astype(np.float64), d["bias"].astype(np.float64)
    win = np.hanning(8193)[:-1]

    def score(mono48: np.ndarray) -> float:
        x = _resample(mono48, 48000, 16000).astype(np.float64)
        xp = np.pad(x, (4096, 4096), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(xp, 8192)[::4096][: 1 + len(x) // 4096] * win
        db = (10 * np.log10(np.clip(np.abs(np.fft.rfft(frames, axis=1)) ** 2, 1e-10, 1e6))).mean(axis=0)
        f = np.linspace(0, 8000, len(db))
        v = db[(f >= 1000) & (f <= 8000)]
        r = np.clip(np.clip(v - np.clip(minimum_filter1d(v, 10, mode="nearest"), -45, None), 0, None), 0, 5)
        fp = r / (r.max() + 1e-6)
        return float(1 / (1 + np.exp(-(fp @ w.T + b).item())))

    return score


def spectttra_scorer(model_id="awsaf49/sonics-spectttra-alpha-5s"):
    import torch
    from sonics import HFAudioClassifier

    torch.set_num_threads(8)
    model = HFAudioClassifier.from_pretrained(model_id, map_location="cpu").eval()
    sr = model.config.audio.sample_rate
    win = int(model.config.audio.max_time * sr)

    def score(mono48: np.ndarray) -> float:
        y = _resample(mono48, 48000, sr)
        segs = [y[s:s + win] for s in range(0, max(1, len(y) - win + 1), win)] or [y]
        segs = [np.pad(s, (0, win - len(s))) / max(float(np.std(s)), 1e-6) for s in segs]
        with torch.inference_mode():
            z = model(torch.from_numpy(np.stack(segs).astype(np.float32))).reshape(-1)
        return float(torch.sigmoid(z).mean())

    return score


def main() -> None:
    import pandas as pd
    import soundfile as sf

    from aimd.splits import load_table, make_rounds

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_experiments import ALL_CONDITIONS, slug

    df = load_table(REPO)
    out = REPO / "data" / "pretrained_scores.csv"
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["clip_id"])
    todo = df[~df.clip_id.isin(have.clip_id)]
    print(f"{len(todo)} clips to score with the published detectors", flush=True)
    scorers = {"spectttra_alpha5s": spectttra_scorer(), "lofcz_fakeprint": lofcz_scorer()}
    rows = []
    for i, r in enumerate(todo.itertuples()):
        y, _ = sf.read(str(DATA / r.path), dtype="float32", always_2d=True)
        mono = y.mean(axis=1)
        rows.append({"clip_id": r.clip_id, **{k: s(mono) for k, s in scorers.items()}})
        if (i + 1) % 500 == 0 or i + 1 == len(todo):
            have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
            have.to_csv(out, index=False)
            rows = []
            print(f"  {i + 1}/{len(todo)}", flush=True)

    scores = pd.read_csv(out).set_index("clip_id")
    conds = ALL_CONDITIONS if df.vocal.notna().any() else [c for c in ALL_CONDITIONS if c not in ("vocal_instr", "genre")]
    if len(conds) < len(ALL_CONDITIONS):
        print("tags.csv missing: vocal/genre rounds skipped for now; re-run after scripts/tag.py (scores are cached)")
    for r in make_rounds(df, which=conds):
        ids = df.clip_id.to_numpy()[r.test]
        for name in scorers:
            path = RESULTS / "predictions" / name / f"{r.condition}__{slug(r.name)}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"clip_id": ids, "label": df.label.to_numpy()[r.test], "p_ai": scores.loc[ids, name].to_numpy()}).to_csv(path, index=False)
    print("Wrote prediction files for every round. Re-run scripts/evaluate.py to include them.")


if __name__ == "__main__":
    main()
