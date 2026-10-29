"""Step 4 - label every excerpt as vocal/instrumental and with a coarse genre.

    python scripts/tag.py          # resumable; ~0.7 s per clip on CPU

Platform metadata is inconsistent across datasets (FMA genres, Jamendo tags, AI
prompts), so clips are tagged the same way whatever their source:
  * genre: zero-shot CLAP (the step-3 embeddings vs text such as "rock music"); it agrees
    with FMA's own genre labels on about 47% of FMA clips across 9 coarse genres (chance ~11%;
    make_report.py recomputes the exact figure), coarse but tagged the same way for human and AI;
  * vocals: the supervised AudioSet classifier AST (MIT/ast-finetuned-audioset-16-16-0.442,
    its "Singing", "Rapping", "Choir"... classes), run only on clips without a known label.

Vocal label, in order of trust:
  1. the dataset's own metadata (Jamendo "voice" tag, FMA "Instrumental" genre,
     Suno/Udio/Riffusion instrumental flags, ...);
  2. what the generator can do: lyrics-to-song models (YuE, DiffRhythm) always sing;
     MusicGen, MAGNeT, JASCO, AudioLDM2, Mustango, Stable Audio Open and Lyria
     RealTime cannot (already in their metadata);
  3. otherwise the tagger, with the threshold that best agrees with labels 1-2.

Tried and rejected for vocals: zero-shot CLAP ("a song with a singer" vs "instrumental
music") agreed with the metadata labels barely above chance (63% balanced) and called the
non-singing generators "vocal".

Output: data/tags.csv (clip_id, p_vocal, vocal, vocal_source, genre, genre_p)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO  # noqa: E402
from aimd.audio import _resample  # noqa: E402

MODEL = "MIT/ast-finetuned-audioset-16-16-0.442"
VOCAL_LABELS = {"Singing", "Male singing", "Female singing", "Child singing", "Synthetic singing", "Rapping", "Choir",
                "A capella", "Vocal music", "Chant", "Yodeling"}
GENRE_PROMPTS = {
    "rock_metal": ["rock music", "heavy metal music", "punk rock"],
    "pop": ["pop music", "a catchy pop song"],
    "hiphop": ["hip hop music", "rap music", "trap beat"],
    "electronic": ["electronic dance music", "techno music", "house music"],
    "rnb_soul_gospel": ["r&b music", "soul music", "gospel music"],
    "country_folk": ["country music", "folk music", "acoustic folk song"],
    "jazz_blues": ["jazz music", "blues music"],
    "classical": ["classical music", "orchestral music", "solo piano classical music"],
    "ambient_newage": ["ambient music", "new age meditation music"],
    "world_latin_reggae": ["latin music", "reggae music", "traditional world music"],
}


def clap_genres(man):
    """Zero-shot genre from the CLAP embeddings: the closest averaged text prompt."""
    import pandas as pd
    import torch

    from aimd.clap import load_model

    model, proc = load_model()
    ids = (REPO / "data" / "emb_clap_ids.txt").read_text().splitlines()
    E = np.load(REPO / "data" / "emb_clap.npy")
    pos = {c: i for i, c in enumerate(ids)}
    names = list(GENRE_PROMPTS)
    rows = []
    for g in names:
        inp = proc(text=GENRE_PROMPTS[g], return_tensors="pt", padding=True)
        with torch.inference_mode():
            e = model.get_text_features(**inp)
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        rows.append(torch.nn.functional.normalize(e, dim=-1).numpy().mean(axis=0))
    G = np.stack(rows)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    have = man[man.clip_id.isin(pos)]
    S = E[[pos[c] for c in have.clip_id]] @ G.T
    P = np.exp(30 * (S - S.max(axis=1, keepdims=True)))
    P /= P.sum(axis=1, keepdims=True)
    return pd.DataFrame({"clip_id": have.clip_id.to_numpy(), "genre": [names[i] for i in P.argmax(axis=1)],
                         "genre_p": P.max(axis=1).round(4)})
ALWAYS_VOCAL = {"yue", "diffrhythm"}  # lyrics-to-song generators


def main() -> None:
    import pandas as pd
    import soundfile as sf
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification

    torch.set_num_threads(6)
    man = pd.read_csv(REPO / "data" / "manifest.csv")
    raw_csv = REPO / "data" / "tags_raw.csv"
    done = pd.read_csv(raw_csv) if raw_csv.exists() else pd.DataFrame(columns=["clip_id"])
    hint0 = pd.to_numeric(man.vocal_hint, errors="coerce").where(lambda h: h.notna(), np.where(man.source.isin(ALWAYS_VOCAL), 1.0, np.nan))
    known_vocal = hint0.notna()
    # also score a calibration sample of known-label clips (both classes) to set the tagger's threshold
    calib = pd.concat([man[hint0 == c].sample(min(250, int((hint0 == c).sum())), random_state=0) for c in (0.0, 1.0)])
    todo = man[~man.clip_id.isin(done.clip_id) & (~known_vocal | man.clip_id.isin(calib.clip_id))]
    print(f"{len(todo)} clips need the vocal tagger ({len(done)} already done; {int(known_vocal.sum())} have known labels)", flush=True)
    if len(todo):
        fe = ASTFeatureExtractor.from_pretrained(MODEL)
        model = ASTForAudioClassification.from_pretrained(MODEL).eval()
        name2id = {v: int(k) for k, v in model.config.id2label.items()}
        vid = [name2id[n] for n in VOCAL_LABELS if n in name2id]
        rows, recs = [], list(todo.itertuples())
        for i in range(0, len(recs), 8):
            chunk = recs[i:i + 8]
            waves = [_resample(sf.read(str(DATA / r.path), dtype="float32", always_2d=True)[0].mean(axis=1), 48000, 16000)
                     for r in chunk]
            with torch.inference_mode():
                p = torch.sigmoid(model(**fe(waves, sampling_rate=16000, return_tensors="pt")).logits).numpy()
            for r, pr in zip(chunk, p):
                rows.append({"clip_id": r.clip_id, "p_vocal": float(pr[vid].max())})
            if len(rows) >= 400 or i + 8 >= len(recs):
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_csv(raw_csv, index=False)
                rows = []
                print(f"  {len(done)}/{len(man)}", flush=True)

    tags = pd.read_csv(raw_csv)[["clip_id", "p_vocal"]]
    m = man[["clip_id", "source", "label", "vocal_hint"]].merge(tags, on="clip_id", how="left")
    m = m.merge(clap_genres(man), on="clip_id", how="left")
    hint = pd.to_numeric(m.vocal_hint, errors="coerce")
    hint = hint.where(hint.notna(), np.where(m.source.isin(ALWAYS_VOCAL), 1.0, np.nan))
    known = hint.notna()
    thr = 0.1
    cal = known & m.p_vocal.notna()  # known-label clips the tagger also scored (from the first, full pass)
    if cal.sum() > 50 and hint[cal].nunique() == 2:
        yk = hint[cal].to_numpy()
        pk = m.p_vocal[cal].to_numpy()
        cands = np.linspace(0.005, 0.6, 120)
        bal = [np.mean([np.mean((pk >= t)[yk == c] == c) for c in (0, 1)]) for t in cands]
        thr = float(cands[int(np.argmax(bal))])
        print(f"vocal threshold {thr:.3f}: balanced agreement {max(bal):.1%} with {int(cal.sum())} known labels "
              f"({int((yk == 1).sum())} vocal, {int((yk == 0).sum())} instrumental)")
    m["vocal"] = np.where(known, hint.fillna(0), (m.p_vocal >= thr).astype(float)).astype(int)
    m["vocal_source"] = np.where(known, "metadata/generator", f"AST p>={thr:.3f}")
    m[["clip_id", "p_vocal", "vocal", "vocal_source", "genre", "genre_p"]].to_csv(REPO / "data" / "tags.csv", index=False)
    print(pd.crosstab([m.label.map({0: "human", 1: "AI"}), m.vocal.map({0: "instrumental", 1: "vocal"})], m.genre).to_string())
    print("\nvocal share by dataset:\n" + m.groupby("source").vocal.mean().round(2).to_string())


if __name__ == "__main__":
    main()
