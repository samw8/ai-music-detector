"""Check one song (or a folder of songs) for signs of AI generation.

    python predict.py path\\to\\song.mp3
    python predict.py path\\to\\folder          # every audio file inside, plus a CSV summary
    python predict.py song.mp3 --all-models    # show every detector, not just the main one

What you get for each song:
  * a probability that it is AI-generated (from the main detector, plus the others for comparison),
  * which measurable features pushed a simpler, interpretable model (features + LR) toward AI or
    human; when the main detector is a different model these are NOT its reasons, and the two
    scores can disagree,
  * an explicit confidence caveat, including how often this detector wrongly accuses
    real musicians on music unlike its training data,
  * a warning for low-sample-rate or low-bitrate files, or files with almost no sound above
    8 kHz, printed only if `run evaluate` found that band-limiting makes the main detector call
    human music AI (models/config.json then has bandlimit_warning). The current main detector
    passed that test, so no warning is printed; genuinely old or lo-fi recordings are still
    flagged more often (see MODEL_CARD.md).

A score is evidence, not proof. Never use it on its own to accuse an artist.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimd import MODELS as MODEL_DIR  # noqa: E402
from aimd import RESULTS  # noqa: E402
from aimd.audio import AUDIO_EXTS, decode_stereo, probe, to_pcm16  # noqa: E402
from aimd.models import label  # noqa: E402

SR, SEG = 48000, 10.0
POSITIONS = (0.25, 0.5, 0.75)


def load_stereo(path: Path, info: dict) -> np.ndarray:
    """Whole file as [2, n] at 48 kHz through exactly the training-data path:
    ffmpeg at the native rate, soxr HQ resampling, then the 16-bit grid of the stored clips."""
    return to_pcm16(decode_stereo(path, 0.0, None, info["orig_sr"], SR))


def excerpts(y: np.ndarray) -> list[np.ndarray]:
    n = int(SR * SEG)
    if y.shape[1] <= n:
        return [np.pad(y, ((0, 0), (0, n - y.shape[1])))]
    out = []
    for pos in POSITIONS:
        s = int(pos * (y.shape[1] - n))
        seg = y[:, s:s + n]
        if np.sqrt(np.mean(seg ** 2)) > 10 ** (-50 / 20):
            out.append(seg)
    return out or [y[:, :n]]


def config() -> dict:
    p = MODEL_DIR / "config.json"
    return json.loads(p.read_text()) if p.exists() else {"primary": "feat_lr"}


def verdict(p: float, cfg: dict) -> str:
    lo, hi = cfg.get("uncertain_band", [0.2, 0.8])
    if p >= hi:
        return "LIKELY AI-GENERATED"
    if p <= lo:
        return "LIKELY HUMAN-MADE"
    return "UNCERTAIN"


def shown(p: float, cfg: dict) -> float:
    """p rounded to 2 decimals for display, kept on its verdict's side of a cut-off (0.797 shows as 0.79, so the
    screen never says "0.80 -> UNCERTAIN"). The verdict itself always uses the unrounded p."""
    lo, hi = cfg.get("uncertain_band", [0.2, 0.8])
    r = round(p, 2)
    if lo < p < hi:
        r = min(max(r, round(lo + 0.01, 2)), round(hi - 0.01, 2))
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--all-models", action="store_true")
    a = ap.parse_args()

    from aimd.infer import Detectors

    target = Path(a.path)
    files = sorted(p for p in target.rglob("*") if p.suffix.lower() in AUDIO_EXTS) if target.is_dir() else [target]
    if not files:
        sys.exit(f"No audio files found at {target}")
    cfg = config()
    primary = cfg.get("primary", "feat_lr")
    names = None if a.all_models else sorted({primary, "feat_lr"})
    det = Detectors(names)
    rows = []
    for f in files:
        try:
            info = probe(f)
            clips = excerpts(load_stereo(f, info))
        except Exception as e:  # noqa: BLE001
            print(f"\n{f.name}: could not read audio ({e})")
            continue
        reps = det.represent(clips)
        per_excerpt = det.predict(reps)
        preds = {m: float(np.mean(p)) for m, p in per_excerpt.items()}
        p = preds.get(primary, next(iter(preds.values())))
        print("\n" + "=" * 78)
        print(f"{f.name}")
        print(f"  P(AI) = {shown(p, cfg):.2f}   ->  {verdict(p, cfg)}   (main detector: {label(primary)})")
        print(f"  scored {len(clips)} ten-second excerpt(s); per-excerpt: " + ", ".join(f"{x:.2f}" for x in np.ravel(per_excerpt[primary])))
        if len(preds) > 1:
            for m, v in preds.items():
                if m != primary:
                    print(f"    {label(m)[:60]:60s} P(AI) = {v:.2f}")
        explainer = primary if primary.split("@")[0] == "feat_lr" else "feat_lr"
        why = det.explain(reps, explainer)
        if why:
            same = explainer == primary
            print(f"  Why ({'the main detector' if same else 'a DIFFERENT, interpretable model: ' + label(explainer)}, "
                  f"P(AI) = {preds.get(explainer, float('nan')):.2f}; averaged over the excerpts; + pushes toward AI, - toward human):")
            if not same:
                print("    (the main detector's own inputs - a spectral fingerprint, embedding or spectrogram - have no per-feature explanation)")
            for name, c, d in why:
                print(f"    {'+' if c > 0 else '-'} {name} ({c:+.2f}): {d}")
        print("  Caveat: " + cfg.get("caveat", "Detector accuracy on music unlike its training data is limited; see MODEL_CARD.md."))
        # missing high frequencies look like AI to the main detector: check the file format AND what is actually in the audio
        b8 = (np.array([d.get("band_8k_12k_db", np.nan) for d in reps.get("feature_dicts", [])], float)
              if "band_8k_12k_db" in det.feature_names else np.array([]))
        content_limited = b8.size > 0 and float(np.median(np.nan_to_num(b8, nan=-99.0))) < -45  # NaN = real bandwidth ends below ~10 kHz
        # evaluate.py writes bandlimit_warning only if the main detector's robustness test showed the shortcut
        if cfg.get("bandlimit_warning") and (0 < info["orig_sr"] < 44100 or 0 < info["kbps"] <= 96 or content_limited):
            print(f"  WARNING: this file is {info['orig_sr']} Hz / {info['kbps'] or '?'} kbps"
                  + (" and has almost no sound above 8 kHz" if content_limited else "") + ". " + cfg["bandlimit_warning"])
        rows.append({"file": str(f), "p_ai": round(p, 4), "verdict": verdict(p, cfg), **{f"p_{m}": round(v, 4) for m, v in preds.items()}})
    if target.is_dir() and rows:
        import pandas as pd

        out = RESULTS / f"predictions_{target.name}.csv"
        out.parent.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
