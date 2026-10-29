"""Case study on named artists (qualitative demo set - never training data).

    python scripts/demo.py

Put your OWN copies of tracks (bought, downloaded legally, or ripped from your own
CDs) in folders named after the artist:

    demo\\audio\\The Strokes\\Is This It.mp3
    demo\\audio\\Breaking Rust\\Livin On Borrowed Time.m4a

The folder name is matched to demo\\artists.csv to know whether the artist is
treated as human or AI and with which provenance tier. Use only the releases
listed in the which_releases column of demo\\artists.csv: it names the exact
albums and warns about look-alike profiles and name collisions; leave every
other track by that artist out. For the two '(track-level only)' rows (Josh
Fawaz, mikeeysmind) only the named track is labelled AI; any other file in that
folder is labelled unknown. Every prediction is
reported, including the wrong ones - finding where the detector fails is the result.
The audio itself must not be shared or published; only the scores are.

Output: results/demo_case_study.csv and results/demo_case_study.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aimd import RESULTS  # noqa: E402
from aimd.audio import AUDIO_EXTS, probe  # noqa: E402


def match_artist(folder: str, table) -> dict | None:
    key = folder.lower().strip()
    for r in table.to_dict("records"):
        names = [r["artist"].lower()] + [s.strip(" ')").lower() for s in r["artist"].replace("(", ",").split(",")]
        if key in names or any(key == n.split(" - ")[0] for n in names):
            return r
    return None


def main() -> None:
    import pandas as pd

    from aimd.infer import Detectors
    from aimd.models import label
    from predict import config, excerpts, load_stereo, verdict

    table = pd.read_csv(ROOT / "demo" / "artists.csv")
    audio_root = ROOT / "demo" / "audio"
    files = sorted(p for p in audio_root.rglob("*") if p.suffix.lower() in AUDIO_EXTS) if audio_root.exists() else []
    if not files:
        sys.exit(f"No audio found. Put your own tracks in {audio_root}\\<artist name>\\ (see the docstring at the top of this file).")
    cfg = config()
    det = Detectors()
    rows = []
    for f in files:
        artist = f.parent.name
        m = match_artist(artist, table)
        if m is not None and "track-level only" in m["artist"]:
            title = re.sub(r"\s*\(.*?\)", "", re.search(r"'(.+?)'", m["artist"]).group(1)).lower()  # 'like a prayer', 'papaoutai'
            if title not in f.stem.lower():
                m = None  # only the named track carries the AI label; the rest of the catalogue is not confirmed AI
        meta = m or {"treat_as": "unknown", "tier": "D", "genre": ""}
        reps = det.represent(excerpts(load_stereo(f, probe(f))))
        preds = {m: float(np.mean(p)) for m, p in det.predict(reps).items()}
        p = preds[cfg.get("primary", "feat_lr")]
        call = verdict(p, cfg)
        expected = meta["treat_as"]
        correct = (expected == "AI" and call.startswith("LIKELY AI")) or (expected == "human" and call.startswith("LIKELY HUMAN"))
        rows.append({"artist": artist, "track": f.stem, "treated_as": expected, "tier": meta["tier"], "genre": meta.get("genre", ""),
                     "p_ai": round(p, 3), "call": call,
                     "agrees_with_label": "n/a (no label)" if expected == "unknown" else ("yes" if correct else ("uncertain" if call == "UNCERTAIN" else "NO")),
                     **{f"p_{m}": round(v, 3) for m, v in preds.items()}})
        print(f"{artist:28s} {f.stem[:40]:40s} {expected:6s} P(AI)={p:.2f} {call}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(RESULTS / "demo_case_study.csv", index=False)
    lines = ["# Case study: named artists", "", f"Main detector: {label(cfg.get('primary', 'feat_lr'))}. Labels are tier C/D "
             "(public reporting / assumed), so a disagreement is a finding to examine, not automatically a detector error. "
             "Rows treated as 'unknown' (other tracks by the track-level-only artists, or unmatched folders) have no label, "
             "so 'Agrees?' is n/a for them.", "",
             "| Artist | Track | Treated as | Tier | P(AI) | Call | Agrees? |", "|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        lines.append(f"| {r.artist} | {r.track} | {r.treated_as} | {r.tier} | {r.p_ai:.2f} | {r.call} | {r.agrees_with_label} |")
    for side in ("human", "AI"):
        s = D[D.treated_as == side]
        if len(s):
            lines.append(f"\n{side}: {len(s)} tracks, called AI {np.mean(s.p_ai >= 0.5):.0%} of the time at a 0.5 threshold.")
    (RESULTS / "demo_case_study.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'demo_case_study.csv'} and .md")


if __name__ == "__main__":
    main()
