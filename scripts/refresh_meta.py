"""Re-derive labels and leakage groups from the stored platform metadata (no downloading).

    python scripts/refresh_meta.py

Use after changing the grouping/filter rules in aimd/sources.py. It rewrites
DATA/raw/<dataset>/meta.jsonl from the sidecar files already on disk and patches
data/manifest.csv to match:
  * Suno/Sonauto tracks that are user-uploaded audio (not generator output) are removed;
  * HAIM clips (Lyria 3, ACE-Step) are grouped by their generation prompt;
  * MuseBench clips are grouped by the source artist of their caption.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import REPO  # noqa: E402
from aimd import sources as S  # noqa: E402


def rewrite_tar_meta(name: str, to_row, accept) -> set[str]:
    src = S.SOURCES[name]
    state = json.loads((src.raw / "tar.state.json").read_text())
    rows, dropped = [], set()
    for k in state["kept"]:
        meta = json.loads((src.raw / f"{k['stem']}.json").read_text(encoding="utf-8"))
        row = to_row(meta)
        row["file"] = k["file"]
        if not accept(meta):
            dropped.add(row["track_id"])
            continue
        rows.append(row)
    S._write_meta(src, rows)
    return dropped


def regroup_meta(name: str, fn) -> dict:
    src = S.SOURCES[name]
    rows = [json.loads(line) for line in open(src.raw / "meta.jsonl", encoding="utf-8")]
    new = {}
    for r in rows:
        g = fn(r)
        if g:
            r["artist_id"] = g
            new[r["track_id"]] = g
    S._write_meta(src, rows)
    return new


def haim_group(r: dict) -> str | None:
    src = "lyria3" if r["track_id"].startswith("lyria3:") else "acestep"
    side = S.SOURCES[src].raw / Path(r["file"]).with_suffix(".json").name
    if not side.exists():
        return None
    out = S._haim_json(r["hf_path"], side.with_suffix(".mp3"))
    return out.get("artist_id")


def main() -> None:
    import pandas as pd

    dropped = rewrite_tar_meta("suno_dedup", S._suno_row, S._NOT_UPLOAD_SUNO)
    dropped |= rewrite_tar_meta("sonauto", S._sonauto_row, S._NOT_UPLOAD_SONAUTO)
    groups = {}
    for name in ("lyria3", "acestep"):
        groups.update(regroup_meta(name, haim_group))
    for name in [n for n in S.SOURCES if n.startswith("mb_")]:
        groups.update(regroup_meta(name, lambda r, h=(name == "mb_jamendo"): S.mb_group(r["member"], h)))

    man = pd.read_csv(REPO / "data" / "manifest.csv")
    n0 = len(man)
    man = man[~man.track_id.isin(dropped)].copy()
    hit = man.track_id.isin(groups)
    changed = int((man.loc[hit, "artist_id"] != man.loc[hit, "track_id"].map(groups)).sum())
    man.loc[hit, "artist_id"] = man.loc[hit, "track_id"].map(groups)
    man.to_csv(REPO / "data" / "manifest.csv", index=False)
    print(f"removed {n0 - len(man)} user-uploaded clips; {changed} leakage groups updated")
    for src in ("lyria3", "acestep", "mb_lyria_rt", "mb_magnet"):
        g = man[man.source == src]
        print(f"  {src}: {len(g)} clips in {g.artist_id.nunique()} groups")


if __name__ == "__main__":
    main()
