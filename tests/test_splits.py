"""The leakage checks must fail loudly. Run: run test  (or python tests/test_splits.py, or pytest if installed)."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aimd  # noqa: E402
from aimd.splits import LeakageError, Round, check_round, cross_platform_artist, load_table, make_rounds  # noqa: E402


def toy() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for src, group in (("fma_train", "fma"), ("fma_pre2010", "fma"), ("mtg_jamendo", "jamendo"), ("guitarset", "guitarset")):
        for i in range(120):
            rows.append(dict(clip_id=f"{src}-{i}", source=src, label=0, model="human", family="human", human_group=group,
                             artist_id=f"{src}:a{i // 2}", track_id=f"{src}:t{i}", vocal=int(rng.random() < 0.5),
                             genre=rng.choice(["rock_metal", "pop"]), tier="B"))
    for gen, fam in (("suno_v4", "suno"), ("suno_v5.5", "suno"), ("udio_v1", "udio"), ("musicgen", "audiocraft"), ("magnet", "audiocraft")):
        for i in range(120):
            rows.append(dict(clip_id=f"{gen}-{i}", source=gen, label=1, model=gen, family=fam, human_group="",
                             artist_id=f"{gen}:u{i // 3}", track_id=f"{gen}:t{i}", vocal=int(gen.startswith(("suno", "udio"))),
                             genre=rng.choice(["rock_metal", "pop"]), tier="B"))
    return pd.DataFrame(rows)


def test_rounds_are_clean():
    df = toy()
    rounds = make_rounds(df)
    conds = {r.condition for r in rounds}
    assert {"matched", "unseen_generator", "unseen_human", "vocal_instr", "genre"} <= conds
    assert any(r.name == "family:audiocraft" for r in rounds)


def test_artist_overlap_fails():
    df = toy()
    r = make_rounds(df, which=("matched",))[0]
    bad = Round("matched", "bad", np.r_[r.train, r.test[:1]], r.val, r.test)
    try:
        check_round(df, bad)
    except LeakageError:
        return
    raise AssertionError("artist overlap was not caught")


def test_held_out_generator_in_training_fails():
    df = toy()
    r = next(r for r in make_rounds(df, which=("unseen_generator",)) if r.name == "musicgen")
    leak = np.flatnonzero((df.model == "musicgen").to_numpy())[:1]
    bad = Round("unseen_generator", "bad", np.r_[r.train, leak], r.val, np.setdiff1d(r.test, leak), held_out_generators=["musicgen"])
    try:
        check_round(df, bad)
    except LeakageError:
        return
    raise AssertionError("held-out generator in training was not caught")


def test_held_out_human_source_in_training_fails():
    df = toy()
    r = make_rounds(df, which=("unseen_human",))[0]
    leak = np.flatnonzero((df.human_group == "fma").to_numpy())[:1]
    bad = Round("unseen_human", "bad", np.r_[r.train, leak], r.val, np.setdiff1d(r.test, leak), held_out_human_sources=["fma"])
    try:
        check_round(df, bad)
    except LeakageError:
        return
    raise AssertionError("held-out human source in training was not caught")


def test_tier_d_never_loaded():
    rows = [dict(clip_id=f"c{i}", source="suno_dedup", label=1, generator="suno_v4", artist_id=f"suno:u{i}",
                 track_id=f"suno:t{i}", tier="D" if i % 3 == 0 else "B") for i in range(30)]
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "data").mkdir()
        pd.DataFrame(rows).to_csv(Path(tmp) / "data" / "manifest.csv", index=False)
        df = load_table(Path(tmp))
    assert len(df) == 20 and not (df.tier == "D").any(), "tier-D clips reached the split table"


def test_same_artist_name_across_platforms_is_one_group():
    """The same musician on FMA, MTG-Jamendo and MuseBench's Jamendo set must get one leakage key."""
    real = aimd.DATA
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "raw" / "fma_train").mkdir(parents=True)
        (Path(tmp) / "raw" / "fma_train" / "meta.jsonl").write_text(
            json.dumps({"track_id": "fma:1", "artist_name": "Foo  Bar"}) + "\n", encoding="utf-8")
        (Path(tmp) / "mtg").mkdir()
        (Path(tmp) / "mtg" / "raw.meta.tsv").write_text("TRACK_ID\tARTIST_ID\tALBUM_ID\tTRACK_NAME\tARTIST_NAME\n"
                                                        "track_0000001\tartist_1\talbum_1\tSong\tFOO BAR\n", encoding="utf-8")
        df = pd.DataFrame(dict(source=["fma_train", "mtg_jamendo", "mb_jamendo", "fma_train"],
                               track_id=["fma:1", "track_0000001", "mb:9", "fma:2"],
                               artist_id=["fma:artist:7", "mtg:artist_1", "mb_jamendo_artist:Foo Bar_123456", "fma:artist:8"]))
        aimd.DATA = Path(tmp)
        try:
            keys = cross_platform_artist(df)
        finally:
            aimd.DATA = real
    assert keys.iloc[0] == keys.iloc[1] == keys.iloc[2] == "name:foo bar", list(keys)
    assert keys.iloc[3] == "fma:artist:8", "a track without a known name must keep its own artist key"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
