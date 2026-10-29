"""Step 3 - pretrained audio embeddings (CLAP) for every excerpt.

    python scripts/embed.py

Model: laion/clap-htsat-unfused (48 kHz input, 512-dim output). The detector built
on these is "model 4: pretrained embedding + linear head". Resumable.
Output: data/emb_clap.npy and data/emb_clap_ids.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO  # noqa: E402
from aimd.clap import embed_waves  # noqa: E402


def main() -> None:
    import pandas as pd
    import soundfile as sf

    man = pd.read_csv(REPO / "data" / "manifest.csv")
    out_npy, out_ids = REPO / "data" / "emb_clap.npy", REPO / "data" / "emb_clap_ids.txt"
    have = [x for x in out_ids.read_text().split("\n") if x] if out_ids.exists() else []
    embs = list(np.load(out_npy)) if have else []
    if len(embs) > len(have):  # interrupted between saving the vectors and the ids: drop the unrecorded tail
        embs = embs[:len(have)]
    if len(embs) < len(have) or len(set(have)) != len(have):
        raise SystemExit("data/emb_clap.npy and emb_clap_ids.txt are out of step; delete both and re-run")
    todo = man[~man.clip_id.isin(set(have))]
    print(f"{len(todo)} clips to embed ({len(have)} done)", flush=True)
    recs = list(todo.itertuples())
    step = 200
    for i in range(0, len(recs), step):
        chunk = recs[i:i + step]
        waves = [sf.read(str(DATA / r.path), dtype="float32", always_2d=True)[0].mean(axis=1) for r in chunk]
        embs += list(embed_waves(waves))
        have += [r.clip_id for r in chunk]
        np.save(out_npy, np.stack(embs))  # vectors first, ids second: a crash in between leaves extra vectors, trimmed on resume
        out_ids.write_text("\n".join(have))
        print(f"  {len(have)}/{len(man)}", flush=True)


if __name__ == "__main__":
    main()
