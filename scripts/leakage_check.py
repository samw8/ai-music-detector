"""Step 7 - prove the splits are clean before believing any good number.

    python scripts/leakage_check.py

Checks (results/leakage_report.md):
  1. Every round re-passes the artist / generator / human-source disjointness checks.
  2. No duplicate audio across train and test (identical decoded excerpts, and
     near-duplicates by CLAP embedding similarity >= 0.98).
  3. Shuffled-label control: a model trained on randomly permuted labels must score ~50%.
  4. Format shortcut probe: how well do file-format facts alone (sample rate, channels,
     bitrate, codec) predict human vs AI? High = the data has a format confound.
  5. Dataset-identity probe: can the features tell which DATASET a clip came from?
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO, RESULTS  # noqa: E402
from aimd import features as F  # noqa: E402
from aimd.metrics import summarize  # noqa: E402
from aimd.splits import base_split, check_round, load_table, make_rounds  # noqa: E402


def _lead_sheet(track_id: str) -> str:
    """GuitarSet 'guitarset:04_Rock2-142-D_comp_mic' -> 'Rock2-142-D' (the lead sheet; every player plays all of them)."""
    import re

    return re.sub(r"_(comp|solo)(_mic)?$", "", re.sub(r"^\d+_", "", track_id.split(":", 1)[1]))


def near_copy_text(df, tr, te, best, arg, thr: float = 0.98) -> str:
    """Line 2b: how many matched test clips have a CLAP near-copy in training, and what those near-copies are.
    best/arg: each test clip's highest cosine similarity to a training clip, and that training clip's position among them."""
    te_i, tr_i = np.flatnonzero(te), np.flatnonzero(tr)
    hit = np.flatnonzero(best >= thr)
    out = (f"2b. **Near-duplicates (CLAP cosine >= {thr}):** {len(hit)} of {len(te_i)} test clips have a near-copy in training "
           f"(median nearest-neighbour similarity {np.median(best):.3f}).")
    if not len(hit):
        return out
    t, n = df.iloc[te_i[hit]], df.iloc[tr_i[arg[hit]]]
    gs = (t.source.to_numpy() == "guitarset") & (n.source.to_numpy() == "guitarset")
    parts = []
    if gs.any():
        other_player = int((t.artist_id.to_numpy()[gs] != n.artist_id.to_numpy()[gs]).sum())
        pairs = [(a.split(":", 1)[1], b.split(":", 1)[1]) for a, b in zip(t.track_id[gs], n.track_id[gs])
                 if _lead_sheet(a) == _lead_sheet(b)]
        sheet = ("a different lead sheet" if not pairs else "mostly a different lead sheet" if len(pairs) < gs.sum() / 2
                 else "often the same lead sheet")
        parts.append(f"{int(gs.sum())} of the {len(hit)} are GuitarSet clips whose nearest training clip is "
                     + ("another player's recording" if other_player == gs.sum() else f"a GuitarSet recording ({other_player} by another player)")
                     + f" with the same instrument and recording setup and {sheet}"
                     + (f" ({len(pairs)} pair{'s' if len(pairs) > 1 else ''} share{'' if len(pairs) > 1 else 's'} a lead sheet: "
                        + "; ".join(f"{a} and {b}" for a, b in pairs) + ")" if pairs else ""))
    rest = t[~gs].source.value_counts()
    if len(rest):
        parts.append(("the other " if gs.any() else "") + ", ".join(f"{k} {'is' if k == 1 else 'are'} {s}" for s, k in rest.items()))
    out += " " + "; ".join(parts) + ". They are left in the scoring (only exact duplicates are removed)."
    g = df[df.source == "guitarset"]
    g_tr, g_te = g[tr[g.index.to_numpy()]], g[te[g.index.to_numpy()]]
    if len(g_tr) and len(g_te):
        shared = int(g_te.track_id.map(_lead_sheet).isin(set(g_tr.track_id.map(_lead_sheet))).sum())
        same_set = g.groupby("artist_id").track_id.apply(lambda s: frozenset(s.map(_lead_sheet))).nunique() == 1
        players = ", ".join(sorted(a.split(":", 1)[1] for a in g_te.artist_id.unique()))
        out += (f" GuitarSet's {len(g_te)} matched test clips (players {players}) "
                + ("all" if shared == len(g_te) else f"include {shared} that") + " play lead sheets that the training players also play"
                + (f" (every GuitarSet player plays the same {g.track_id.map(_lead_sheet).nunique()} lead sheets)" if same_set else "")
                + ", so the GuitarSet part of the matched test measures new players, not new compositions.")
    return out


def shuffle_text(bas, aucs) -> str:
    """Line 3. A label-shuffle control only shows the scoring code gives chance when the labels carry no signal; it cannot
    reveal artist or composition leakage (checks 1, 2a and 2b look for those)."""
    return (f"3. **Shuffled-label control (10 permutations):** balanced accuracy {np.mean(bas):.3f} ± {np.std(bas):.3f}, "
            f"AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}; it should centre on 0.5. This shows only that the evaluation code gives "
            "chance-level scores when the labels carry no signal (features + LR). It cannot detect artist or composition leakage: a "
            "model trained on shuffled labels has nothing to exploit either way. Checks 1, 2a and 2b look for those.")


def main() -> None:
    import pandas as pd
    import soundfile as sf
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer

    RESULTS.mkdir(parents=True, exist_ok=True)  # before the slow hashing pass, so a missing folder never wastes it
    df = load_table(REPO)
    split = base_split(df)
    tr, te = (split == "train").to_numpy(), (split == "test").to_numpy()
    y = df.label.to_numpy()
    lines = ["# Leakage and confound report", ""]

    rounds = make_rounds(df)
    for r in rounds:
        check_round(df, r)
    lines += [f"1. **Disjointness:** all {len(rounds)} rounds pass (no shared artist, track, held-out generator or held-out human source "
              "between training and test).", ""]

    print("hashing excerpts...", flush=True)
    hashes = []
    for p in df.path:
        y16, _ = sf.read(str(DATA / p), dtype="int16")
        hashes.append(hashlib.md5(y16.tobytes()).hexdigest())
    df["audio_md5"] = hashes
    dup = df[df.duplicated("audio_md5", keep=False)]
    cross = set(df[tr].audio_md5) & set(df[te].audio_md5)
    lines += [f"2a. **Identical audio:** {len(dup)} clips share an exact duplicate somewhere in the data; "
              f"{len(cross)} duplicate(s) span the matched train/test split.", ""]
    if len(dup):
        lines.append("    Duplicate groups (first 10): " + "; ".join(
            ", ".join(g.clip_id) for _, g in list(dup.groupby("audio_md5"))[:10]))
        lines.append("")

    ids = (REPO / "data" / "emb_clap_ids.txt").read_text().split("\n")
    E = np.load(REPO / "data" / "emb_clap.npy")
    pos = {c: i for i, c in enumerate(ids)}
    Etr, Ete = E[[pos[c] for c in df.clip_id[tr]]], E[[pos[c] for c in df.clip_id[te]]]
    best, arg = np.zeros(len(Ete)), np.zeros(len(Ete), dtype=int)
    for i in range(0, len(Ete), 512):
        sim = Ete[i:i + 512] @ Etr.T
        best[i:i + 512], arg[i:i + 512] = sim.max(axis=1), sim.argmax(axis=1)
    lines += [near_copy_text(df, tr, te, best, arg), ""]

    feats = pd.read_csv(REPO / "data" / "features.csv").set_index("clip_id").loc[df.clip_id, F.names()].to_numpy()
    # 2c. exact duplicates that straddle training and test in ANY round
    import re as _re

    straddle, excluded = [], []
    dup_groups = [set(g.index) for _, g in dup.groupby("audio_md5")]
    for r in rounds:
        fit, test = set(np.union1d(r.train, r.val)), set(r.test)
        for g in dup_groups:
            if g & fit and g & test:
                straddle.append(f"{r.condition}/{r.name}: " + ", ".join(df.clip_id[sorted(g)]))
                for i in g & test:  # the test-side copy is dropped from that round's scoring by evaluate.py
                    excluded.append({"condition": r.condition, "round": _re.sub(r"[^A-Za-z0-9._=-]+", "_", r.name),
                                     "clip_id": df.clip_id[i], "reason": "exact duplicate of a training clip"})
    pd.DataFrame(excluded, columns=["condition", "round", "clip_id", "reason"]).to_csv(RESULTS / "excluded_test_clips.csv", index=False)
    lines += [f"2c. **Duplicates straddling a split in any round:** {len(straddle)}."
              + ("" if not straddle else " " + "; ".join(straddle[:10])
                 + f". The test-side copies ({len(excluded)} clip-round pairs) are listed in results/excluded_test_clips.csv "
                   "and left out of scoring by evaluate.py."), ""]
    mixed = [g for g in dup_groups if df.model[sorted(g)].nunique() > 1]
    if mixed:
        lines += ["    Duplicates carrying different generator labels (the same audio filed under two versions): "
                  + "; ".join(", ".join(f"{df.clip_id[i]} ({df.model[i]})" for i in sorted(g)) for g in mixed)
                  + ". When one of those versions is held out, its copy under the other label stays in training: one track per "
                    "round, too small to move a result, but it means one of the two labels is wrong.", ""]

    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    rng = np.random.default_rng(0)
    bas, aucs = [], []
    for _ in range(10):  # one permutation is a noisy control: a random direction can align with a real feature by chance
        perm = y[tr].copy()
        rng.shuffle(perm)
        lr.fit(feats[tr], perm)
        s = summarize(y[te], lr.predict_proba(feats[te])[:, 1])
        bas.append(s["balanced_accuracy"])
        aucs.append(s["auc"])
    lines += [shuffle_text(bas, aucs), ""]

    fmt_cols = ["orig_sr", "orig_ch", "kbps", "codec"]
    X = df[fmt_cols].copy()
    ct = ColumnTransformer([("num", make_pipeline(SimpleImputer(), StandardScaler()), ["orig_sr", "orig_ch", "kbps"]),
                            ("cat", OneHotEncoder(handle_unknown="ignore"), ["codec"])])
    probe = make_pipeline(ct, LogisticRegression(max_iter=3000, class_weight="balanced"))
    probe.fit(X[tr], y[tr])
    s = summarize(y[te], probe.predict_proba(X[te])[:, 1])
    lines += [f"4. **Format shortcut probe:** file format alone (sample rate, channels, bitrate, codec) separates human from AI "
              f"with balanced accuracy {s['balanced_accuracy']:.3f} (AUC {s['auc']:.3f}). The models never see these metadata, "
              "and every clip is re-decoded to 48 kHz, but traces of the original format (band limits, mono) survive in the audio - "
              "which is why the 'no file-format features' ablation exists.", ""]
    lines.append("    Format by class:")
    lines.append("")
    lines.append(df.groupby("label")[["orig_sr", "orig_ch", "kbps"]].median().rename(index={0: "human", 1: "AI"}).to_markdown())
    lines.append("")
    lines.append(pd.crosstab(df.codec, df.label.map({0: "human", 1: "AI"})).to_markdown())
    lines.append("")

    src = df.source.to_numpy()
    lr2 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=3000))
    lr2.fit(feats[tr], src[tr])
    acc = float((lr2.predict(feats[te]) == src[te]).mean())
    lines += [f"5. **Dataset-identity probe:** the interpretable features identify which of the {df.source.nunique()} datasets a test clip "
              f"came from with accuracy {acc:.3f} (chance = {1 / df.source.nunique():.3f}). Datasets have recognisable recording/format "
              "signatures; this is why unseen-human and unseen-generator conditions, not the matched split, are the real test.", ""]

    out = RESULTS / "leakage_report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
