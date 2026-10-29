"""The evaluation conditions, as train/validation/test splits that are checked for leakage.

Every round produced here passes ``check_round`` or the program stops with an
error: no artist may appear on both sides of a split, a held-out generator may
not appear in training, and a held-out human source may not appear in training.

Conditions (the brief's four, plus the genre experiment from the original hypothesis):
    matched          artist-disjoint 70/10/20 split of everything
    unseen_generator leave one AI model out (each with >= MIN_GEN_CLIPS clips), and each multi-version family out
    unseen_human     train on one human source, test on human music from another
    vocal_instr      train on vocal music, test on instrumental (and the reverse)
    genre            leave one genre out vs. the same genre seen in training
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEED = 0
TRAIN_FRAC, VAL_FRAC = 0.7, 0.1  # test = 0.2 (brief says 70/20/10; we keep the larger share for the reported test set)
# Both Jamendo datasets come from the same platform (and may share artists), so they are held out together.
HUMAN_SOURCE_GROUP = {"fma_train": "fma", "fma_pre2010": "fma", "mtg_jamendo": "jamendo", "mb_jamendo": "jamendo", "guitarset": "guitarset"}
MIN_GEN_CLIPS = 60


class LeakageError(RuntimeError):
    pass


@dataclass
class Round:
    condition: str
    name: str
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    held_out_generators: list = field(default_factory=list)
    held_out_human_sources: list = field(default_factory=list)
    note: str = ""


def normalise_generator(g: str) -> str:
    """One name per model version, so leave-one-model-out holds out each version cleanly."""
    g = str(g)
    if g.startswith("riffusion_fuzz") or g == "riffusion_unknown":
        return "riffusion_fuzz1.1"  # 2025 platform crawl (FUZZ era); riffusion_v1 is the separate 2022 research model
    if g in ("suno_v3.5", "suno_chirp-v3-5", "suno_v3"):
        return "suno_v3.5"
    if g.startswith("suno_v4.5"):
        return "suno_v4.5"
    if g.startswith("suno_chirp-") and g not in ("suno_chirp-v3-5",):
        return "suno_other"
    if g.startswith("sonauto_"):
        return "sonauto_v2"
    if g.startswith("mureka_") and g != "mureka_V6":
        return "mureka_other"
    return g


def _norm_name(s) -> str:
    import unicodedata

    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return " ".join(s.split())


def cross_platform_artist(df: pd.DataFrame) -> pd.Series:
    """One leakage key per artist NAME across platforms (FMA, MTG-Jamendo, MuseBench Jamendo and the
    MuseBench clips captioned from that artist), so the same musician can never sit on both sides.
    Two different acts with the same name get merged too, which only costs a little training data."""
    import json

    from . import DATA

    names: dict[str, str] = {}
    for src in ("fma_train", "fma_pre2010"):
        meta = DATA / "raw" / src / "meta.jsonl"
        if meta.exists():
            for line in open(meta, encoding="utf-8"):
                r = json.loads(line)
                if r.get("artist_name") and str(r["artist_name"]) != "nan":
                    names[r["track_id"]] = r["artist_name"]
    mtg = DATA / "mtg" / "raw.meta.tsv"
    if (df.source == "mtg_jamendo").any() and not mtg.exists():
        from .fetch import download_file
        from .sources import MTG_META

        download_file(MTG_META, mtg)  # artist names; without them MTG artists could not be matched across platforms
    if mtg.exists():
        for line in mtg.read_text(encoding="utf-8").splitlines()[1:]:
            p = line.split("\t")
            if len(p) > 4:
                names[p[0]] = p[4]
    out = df.artist_id.copy()
    by_track = df.track_id.map(names)
    has = by_track.notna()
    out[has] = "name:" + by_track[has].map(_norm_name)
    mbj = df.artist_id.str.startswith("mb_jamendo_artist:")
    # JamendoMaxCaps appends "_<track id>" to colliding artist names in filenames: strip it
    out[mbj] = "name:" + df.artist_id[mbj].str.split(":", n=1).str[1].str.replace(r"_\d{4,}\s*$", "", regex=True).map(_norm_name)
    return out


def load_table(repo) -> pd.DataFrame:
    man = pd.read_csv(repo / "data" / "manifest.csv")
    tag_csv = repo / "data" / "tags.csv"
    if tag_csv.exists():
        tags = pd.read_csv(tag_csv)[["clip_id", "p_vocal", "vocal", "vocal_source", "genre", "genre_p"]]
        df = man.merge(tags, on="clip_id", how="left")
    else:
        print("[splits] data/tags.csv not found yet: vocal/genre conditions unavailable until scripts/tag.py has run")
        df = man.assign(p_vocal=np.nan, vocal=np.nan, vocal_source="", genre=np.nan, genre_p=np.nan)
    df["model"] = df.generator.map(normalise_generator)
    df["human_group"] = df.source.map(HUMAN_SOURCE_GROUP).fillna("")
    df["artist_id_platform"] = df.artist_id
    df["artist_id"] = cross_platform_artist(df)
    if (df.tier == "D").any():
        n = int((df.tier == "D").sum())
        df = df[df.tier != "D"]
        print(f"[splits] dropped {n} tier-D (assumed-label) clips: tier D never enters training or evaluation")
    return df.reset_index(drop=True)


def base_split(df: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """Assign every clip to train/val/test by ARTIST (~70/10/20 of each dataset's clips).

    An artist (or leakage group) gets ONE side for all its clips, even when it appears in
    several datasets - e.g. MuseBench generations from one source artist's captions. Groups
    are shuffled and filled within the dataset where they have most clips.
    """
    import hashlib

    counts = df.groupby("artist_id").size()
    primary = df.groupby("artist_id").source.agg(lambda s: s.value_counts().index[0])
    amap = {}
    for _, artists in primary.groupby(primary):
        # order by a hash of the group key: a group's side then depends only on its own dataset's groups,
        # so a small change elsewhere (one regrouped artist) cannot reshuffle every other split
        a = np.array(sorted(artists.index, key=lambda k: hashlib.md5(f"{seed}:{k}".encode()).hexdigest()))
        cum = np.cumsum(counts.loc[a].to_numpy()) / counts.loc[a].sum()
        lab = np.where(cum <= TRAIN_FRAC, "train", np.where(cum <= TRAIN_FRAC + VAL_FRAC, "val", "test"))
        amap.update(zip(a, lab))
    return df.artist_id.map(amap)


def check_round(df: pd.DataFrame, r: Round) -> None:
    """Fail loudly on any leakage between the training side (train+val) and the test side."""
    fit = np.union1d(r.train, r.val)
    if len(np.intersect1d(fit, r.test)):
        raise LeakageError(f"[{r.condition}/{r.name}] the same clip is in training and test")
    shared = set(df.loc[fit, "artist_id"]) & set(df.loc[r.test, "artist_id"])
    if shared:
        raise LeakageError(f"[{r.condition}/{r.name}] {len(shared)} artist(s) on both sides, e.g. {sorted(shared)[:3]}")
    tracks = set(df.loc[fit, "track_id"]) & set(df.loc[r.test, "track_id"])
    if tracks:
        raise LeakageError(f"[{r.condition}/{r.name}] {len(tracks)} track(s) on both sides")
    for g in r.held_out_generators:
        if (df.loc[fit, "model"] == g).any():
            raise LeakageError(f"[{r.condition}/{r.name}] held-out generator {g} appears in training")
    for s in r.held_out_human_sources:
        if (df.loc[fit, "human_group"] == s).any():
            raise LeakageError(f"[{r.condition}/{r.name}] held-out human source {s} appears in training")
    for part, idx in (("train", r.train), ("test", r.test)):
        if len(idx) == 0:
            raise LeakageError(f"[{r.condition}/{r.name}] {part} set is empty")
        if df.loc[idx, "label"].nunique() < 2 and not (part == "test" and r.condition == "unseen_human"):
            raise LeakageError(f"[{r.condition}/{r.name}] {part} set has only one class")


def _idx(mask) -> np.ndarray:
    return np.flatnonzero(np.asarray(mask))


def make_rounds(df: pd.DataFrame, which=("matched", "unseen_generator", "unseen_human", "vocal_instr", "genre")) -> list[Round]:
    split = base_split(df)
    tr, va, te = split == "train", split == "val", split == "test"
    ai, hu = df.label == 1, df.label == 0
    rounds: list[Round] = []

    if "matched" in which:
        rounds.append(Round("matched", "all", _idx(tr), _idx(va), _idx(te), note="artist-disjoint 70/10/20 split of every dataset"))

    if "unseen_generator" in which:
        counts = df[ai].model.value_counts()
        for g in sorted(counts[counts >= MIN_GEN_CLIPS].index):
            out = df.model == g
            rounds.append(Round("unseen_generator", g, _idx(tr & ~out), _idx(va & ~out), _idx((out & ai) | (te & hu)),
                                held_out_generators=[g], note=f"all {int(counts[g])} {g} clips held out"))
        # families whose versions share a company / decoder (e.g. MusicGen, MAGNeT and JASCO all decode with EnCodec-32k):
        # holding out one member while training on its siblings overstates generalisation, so also hold out whole families
        for fam in sorted(df[ai].family.unique()):
            gens = sorted(df[ai & (df.family == fam)].model.unique())
            if len(gens) > 1:
                out = df.family == fam
                rounds.append(Round("unseen_generator", f"family:{fam}", _idx(tr & ~out), _idx(va & ~out),
                                    _idx((out & ai) | (te & hu)), held_out_generators=gens, note=f"every {fam} version held out"))

    if "unseen_human" in which:
        # U1 (the brief's headline): no FMA in training at all; test on FMA uploaded before 2010.
        # U2 (the reverse): no Jamendo in training; test on Jamendo. GuitarSet is not trained on in these rounds
        # (the matched split does train on 4 of its 6 players).
        for held, train_src in (("fma", "jamendo"), ("jamendo", "fma")):
            keep_h = hu & (df.human_group == train_src)
            test_h = hu & df.human_group.isin([held, "guitarset"])
            rounds.append(Round("unseen_human", f"train_human={train_src}|test={held}", _idx((tr & ai) | (tr & keep_h)),
                                _idx((va & ai) | (va & keep_h)), _idx(test_h | (te & ai)),
                                held_out_human_sources=[held, "guitarset"],
                                note=f"human training data only from {train_src}; {held} and GuitarSet are unseen"))

    if "vocal_instr" in which:
        voc, ins = df.vocal == 1, df.vocal == 0  # untagged (NaN) clips are neither
        if voc.sum() == 0 or ins.sum() == 0:
            print("[splits] vocal_instr skipped: no vocal/instrumental tags yet (run scripts/tag.py)")
            pairs = ()
        else:
            if df.vocal.isna().any():
                print(f"[splits] vocal_instr: {int(df.vocal.isna().sum())} untagged clips left out")
            pairs = (("vocal", voc, "instrumental", ins), ("instrumental", ins, "vocal", voc))
        for a_name, a_mask, b_name, b_mask in pairs:
            rounds.append(Round("vocal_instr", f"train={a_name}|test={b_name}", _idx(tr & a_mask), _idx(va & a_mask), _idx(te & b_mask),
                                note=f"trained only on {a_name} clips (both classes), tested on {b_name}"))
            rounds.append(Round("vocal_instr", f"train={a_name}|test={a_name}", _idx(tr & a_mask), _idx(va & a_mask), _idx(te & a_mask),
                                note="in-domain reference for the row above"))

    if "genre" in which:
        import hashlib

        n_src = df[tr].source.value_counts()
        for g in sorted(df.genre.dropna().unique()):
            gm = df.genre == g
            # A dataset that is nearly all one genre (GuitarSet: 311 of 360 clips tagged country_folk) leaves training
            # together with the genre, so its test clips would measure "dataset unseen" (the unseen_human question),
            # not "genre unseen". Such datasets (under a quarter of their training clips left) are not scored here.
            kept = df[tr & ~gm].source.value_counts().reindex(n_src.index, fill_value=0) / n_src
            gone = sorted(kept[kept < 0.25].index)
            gtest = te & gm & ~df.source.isin(gone)
            if (gtest & ai).sum() < 30 or (gtest & hu).sum() < 30:
                continue
            extra = f"; {', '.join(gone)} not scored (almost all {g}, it would leave training with the genre)" if gone else ""
            # "seen" trains on a random subset the same size as "held_out", so any gap is the genre, not the amount of data
            rng = np.random.default_rng(int(hashlib.md5(f"{SEED}:{g}".encode()).hexdigest()[:8], 16))
            seen = np.sort(rng.choice(_idx(tr), int((tr & ~gm).sum()), replace=False))
            rounds.append(Round("genre", f"{g}|seen", seen, _idx(va), _idx(gtest),
                                note="genre present in training (training set subsampled to the held-out size)" + extra))
            rounds.append(Round("genre", f"{g}|held_out", _idx(tr & ~gm), _idx(va & ~gm), _idx(gtest),
                                note="genre removed from training" + extra))

    for r in rounds:
        _purge(df, r)
        check_round(df, r)
    return rounds


def _purge(df: pd.DataFrame, r: Round) -> None:
    """Drop from train/val any clip whose leakage group is in this round's test set.

    Needed when a test set takes a whole generator or human source (all splits): e.g. a
    MAGNeT clip in test and an AudioLDM2 clip made from the same artist's caption in training.
    """
    test_groups = set(df.artist_id.to_numpy()[r.test])
    n0 = len(r.train) + len(r.val)
    r.train = r.train[~np.isin(df.artist_id.to_numpy()[r.train], list(test_groups))]
    r.val = r.val[~np.isin(df.artist_id.to_numpy()[r.val], list(test_groups))]
    dropped = n0 - len(r.train) - len(r.val)
    if dropped:
        r.note += f"; {dropped} training clips removed because their artist/caption group is in the test set"
