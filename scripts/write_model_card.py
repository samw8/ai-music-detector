"""Step 14 - write MODEL_CARD.md from the current results (re-run after any change).

    python scripts/write_model_card.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimd import MODELS as MODEL_DIR  # noqa: E402
from aimd import RESULTS  # noqa: E402
from aimd import features as F  # noqa: E402
from aimd.models import label  # noqa: E402
from aimd.sources import SOURCES  # noqa: E402
from aimd.cnn import FMAX  # noqa: E402
from aimd.metrics import MIN_CLUSTERS  # noqa: E402
from aimd.splits import MIN_GEN_CLIPS, load_table  # noqa: E402
from evaluate import vocal_generator_confound  # noqa: E402  (same numbers as the caveat in models/config.json)

WEBSITE = "https://who-gets-called-ai.streamlit.app"
HIST_LICENCE = "Sound Recording Common Law Protection"  # how FMA tags its digitised early commercial 78-rpm records
# Measured OUTSIDE this pipeline (an audit script matched HAIM's published MusicGen prompts to these files by file order and
# md5): of the matched-test MusicGen clips, how many share one of HAIM's prompts with a training clip. The pipeline cannot
# recompute it (it never reads those prompts), so it is quoted only while the matched test set has the same MusicGen count.
MUSICGEN_PROMPT_CHECK = {"n_test": 81, "n_shared": 53, "n_prompts": 40}


def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def main() -> None:
    cfg = json.loads((MODEL_DIR / "config.json").read_text())
    T = pd.read_csv(RESULTS / "results_table.csv").set_index("detector")
    UH = pd.read_csv(RESULTS / "unseen_human_breakdown.csv")
    df = load_table(ROOT)
    p = cfg["primary"]
    row = T.loc[p]
    rob = pd.read_csv(RESULTS / "robustness.csv") if (RESULTS / "robustness.csv").exists() else None
    # the unseen-generator rounds, described from the same data and rule make_rounds() uses
    ai_df = df[df.label == 1]
    fams = sorted(f for f, g in ai_df.groupby("family") if g.model.nunique() > 1)
    vc = ai_df.model.value_counts()
    small = sorted(vc[vc < MIN_GEN_CLIPS].index)
    ug = (f"- **Unseen generator:** each model version with at least {MIN_GEN_CLIPS} clips is removed from training in turn and "
          f"tested alone; every family with more than one version is also removed as a whole ({', '.join(fams)})."
          + (f" {', '.join(small)} (under {MIN_GEN_CLIPS} clips) get no round of their own and are held out only with their family."
             if small else ""))
    lo, hi = cfg.get("uncertain_band", [0.2, 0.8])
    vb = cfg.get("unseen_human_verdicts", {})
    verdicts = ""
    if vb:
        (fh, fu), (jh, ju) = vb["fma_pre2010"], vb["jamendo"]
        verdicts = (f" These rates count a P(AI) of 0.5 or above as an AI call. predict.py and the website say LIKELY AI-GENERATED only "
                    f"at {hi} or above, LIKELY HUMAN-MADE at {lo} or below and UNCERTAIN in between; at {hi} or above it still flagged "
                    f"{pct(fh)} of those FMA tracks and {pct(jh)} of those Jamendo tracks, and put {pct(fu)} and {pct(ju)} of them in UNCERTAIN.")
    is_resnet = p.split("@")[0] == "resnet18_mel"
    # what the trained (private) model saw: the matched round's training clips (and, for the ResNet, a balanced 3,000 of them)
    from aimd.cnn import _balanced_subset
    from aimd.splits import make_rounds

    mr = next(r for r in make_rounds(df, which=("matched",)) if r.condition == "matched")
    y = df.label.to_numpy()
    seen_txt = (f"The matched split puts {len(mr.train):,} clips in training, {len(mr.val):,} in validation and {len(mr.test):,} in test; "
                f"the trained models (kept private: the public repository has only models/config.json) were trained on the "
                f"{len(mr.train):,} training clips only.")
    if is_resnet:
        sub, _ = _balanced_subset(np.asarray(mr.train), y[mr.train], 3000, 0)
        s = df.iloc[sub]
        hc = s[s.label == 0].source.value_counts()
        seen_txt += (f" The trained ResNet18, the copy the website uses, saw a class-balanced random {len(sub):,} of them "
                     f"({int((s.label == 0).sum()):,} human: "
                     + ", ".join(f"{n} {k}" for k, n in hc.items())
                     + f"; {int((s.label == 1).sum()):,} AI from {s[s.label == 1].model.nunique()} generators); a balanced 600 of the "
                       f"{len(mr.val):,} validation clips picked its best epoch.")
    gs = df[df.source == "guitarset"]
    gs_seen, gs_all = df.iloc[mr.train][df.iloc[mr.train].source == "guitarset"].artist_id.nunique(), gs.artist_id.nunique()
    num = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    fma = df[df.source.str.startswith("fma")]
    lic = fma.license.astype(str).str.lower()
    is_hist = lic.str.contains(HIST_LICENCE.lower(), regex=False)
    n_hist = int(is_hist.sum())
    hist_by = fma[is_hist].source.value_counts().to_dict()
    hist_lic = ", ".join(sorted(fma[is_hist].license.unique()))
    hist_ot = int((is_hist & (fma.genre_meta == "Old-Time / Historic")).sum())
    hist_where = ("all in its 'Old-Time / Historic' genre" if hist_ot == n_hist else
                  f"{hist_ot} of them in its 'Old-Time / Historic' genre")
    n_nd, n_nc = int(lic.str.contains("noderiv|no deriv").sum()), int(lic.str.contains("noncommercial").sum())
    n_scraped = int(df.license.astype(str).str.startswith("unclear").sum())
    scraped_src = ", ".join(sorted(df[df.license.astype(str).str.startswith("unclear")].source.unique()))
    # the AI datasets' licence mix, from the same SOURCES entries the Datasets table prints
    ai_lic = (df[df.label == 1].groupby("source").size().rename("n").reset_index()
              .assign(lic=lambda d: d.source.map(lambda s: SOURCES[s].license if s in SOURCES else "unknown")))
    ai_lic = ai_lic[~ai_lic.lic.str.startswith("unclear")].copy()
    ai_lic[["base", "note"]] = ai_lic.lic.str.extract(r"^(.*?)(?:\s*\((model|claimed|per MuseBench card)\))?$").fillna("")
    note_txt = {"model": "the generating model's licence", "claimed": "as claimed by the uploader", "per MuseBench card": "per the MuseBench card"}
    mix = []
    for base, g in sorted(ai_lic.groupby("base"), key=lambda kv: -kv[1].n.sum()):
        tot, notes = int(g.n.sum()), g[g.note != ""].groupby("note").n.sum()
        extra = "".join(f"; {note_txt[k]}" if c == tot else f"; for {c:,} of them, {note_txt[k]}" for k, c in notes.items())
        mix.append(f"{base} ({tot:,} clips{extra})")
    lic_mix = "; ".join(mix)
    # Jamendo: is the per-track licence actually recorded, or only a generic note?
    jam = df[df.source.isin(["mtg_jamendo", "mb_jamendo"])]
    jam_generic = len(jam) and jam.license.astype(str).str.contains("per track").all()
    # AI datasets with a different group key for every clip (no uploader or prompt recorded): split per track only
    ai_df_all = df[df.label == 1]
    per_track = sorted(s for s, g in ai_df_all.groupby("source") if g.artist_id_platform.nunique() == len(g))
    n_per_track = int(ai_df_all.source.isin(per_track).sum())
    # which rounds share every test generator with training (matched, unseen human), for the generator-disjointness note
    same_gen = []
    for r in make_rounds(df, which=("matched", "unseen_human")):
        f_ = np.union1d(r.train, r.val)
        seen_g = set(df.iloc[f_][df.iloc[f_].label == 1].model)
        tg = df.iloc[r.test][df.iloc[r.test].label == 1].model.unique()
        same_gen.append((r.condition, int(pd.Series(tg).isin(seen_g).sum()), len(tg)))
    all_same_gen = all(a == b for _, a, b in same_gen) and len({b for _, _, b in same_gen}) == 1
    # the unseen-human rates are per 10-second excerpt of each source file: how long those files are
    fma_s = df[df.source == "fma_pre2010"].duration.median()
    jam_min = df[df.source.isin(["mtg_jamendo", "mb_jamendo"])].duration.median() / 60
    mt = df.iloc[mr.test]
    mt_ai = mt[mt.label == 1]
    fit_m = np.union1d(mr.train, mr.val)
    gen_fit = set(df.iloc[fit_m][df.iloc[fit_m].label == 1].model)
    n_mt_gen, n_mt_gen_shared = mt_ai.model.nunique(), int(mt_ai.model.drop_duplicates().isin(gen_fit).sum())
    n_mt_musicgen = int((mt_ai.source == "musicgen").sum())
    vc = vocal_generator_confound(df, p)
    vi, iv = vc.get("vocal_to_instr", {}), vc.get("instr_to_vocal", {})

    def cond4(r) -> str:
        """The other three conditions, for any sentence that quotes the unseen-human rates."""
        return (f"In the other conditions: on the matched test split (new artists, otherwise familiar music) "
                f"{pct(r.get('matched_balanced_accuracy'))} balanced accuracy, with {pct(r.get('matched_fpr'))} of human tracks called AI; "
                f"with one AI generator held out, {pct(r.get('unseen_gen_balanced_accuracy_mean'))} balanced accuracy on average "
                f"(it caught {pct(r.get('unseen_gen_recall_mean'))} of the held-out generator's tracks; worst "
                f"{r.get('unseen_gen_worst_generator', '?')}, {pct(r.get('unseen_gen_recall_worst'))}); trained on vocal music only, "
                f"{pct(r.get('vocal_to_instr_balanced_accuracy'))} on instrumentals, and trained on instrumentals only, "
                f"{pct(r.get('instr_to_vocal_balanced_accuracy'))} on vocal music.")

    L = [f"# Model card - {label(p)}", "",
         "> **Read this first.** Retrained with FMA left out of its training data, this detector called "
         f"**{pct(1 - row.get('unseen_human_acc_fma_pre2010'))} of FMA's pre-2010 tracks** AI-generated; retrained "
         f"without Jamendo, it called **{pct(1 - row.get('unseen_human_acc_jamendo'))} of Jamendo tracks** AI-generated. "
         "Every one of those is a false accusation of a real musician. Expect rates like these on music unlike its training data "
         f"(the trained copy, which is kept private and is the one the website uses, was trained on FMA, Jamendo and "
         f"{num.get(gs_seen, gs_seen)} of the {num.get(gs_all, gs_all)} GuitarSet players too). Rates are per 10-second excerpt of each source file "
         f"(a {fma_s:.0f}-second clip for FMA, the full track for Jamendo, median {jam_min:.1f} minutes)."
         + verdicts + " " + cond4(row)
         + (" **Old, phone-quality or low-sample-rate human recordings are called AI far more often** (see Known failure modes)."
            if cfg.get("bandlimit_warning") else "") + " A score is evidence, not proof.", "",
         "## What it is", "",
         f"- Binary classifier: P(AI-generated) for a 10-second excerpt at 48 kHz (predict.py averages up to three excerpts per song).",
         *(["- ResNet18 initialised from torchvision's ImageNet-1k weights (IMAGENET1K_V1), first layer summed to one input channel; "
            f"input = mono 128-band log-mel, 20 Hz-{FMAX / 1000:g} kHz (capped there so a missing top octave cannot separate "
            "the classes; see Known failure modes). It has no per-feature explanation: the 'why' lines predict.py prints come from a "
            "separate, simpler model (interpretable features + logistic regression), are not this detector's reasons, and can disagree with it."]
           if is_resnet else []),
         f"- Chosen as the main detector because it has the best *worst-case* score across the four evaluation conditions "
         f"({cfg['worst_case_score']:.3f}); see results/results_table.md for every model. "
         + ("That choice, and the redesign that capped its spectrogram at "
            f"{FMAX / 1000:g} kHz (an earlier version saw up to 16 kHz), were made after looking at the same test rounds reported here. "
            if is_resnet else "That choice was made after looking at the same test rounds reported here. ")
         + "There is no untouched final test set, so this detector's numbers may be slightly optimistic.",
         "- Intended use: research on AI-music detection; triage that a human then checks. **Not** for accusing, "
         "demonetising or removing an artist's music.", "",
         "## Results in all four conditions (never quote one without the others)", "",
         (RESULTS / "results_table.md").read_text(encoding="utf-8"), "",
         "Balanced accuracy = average of the share of AI tracks caught and the share of human tracks correctly passed. "
         "Unseen human = accuracy on human-only music from a source absent from training (= 1 - false-positive rate).", "",
         "## How it was tested", "",
         "- **Matched:** 70% train / 10% validation / 20% test, split by artist (a cross-platform artist key, so the same act "
         "on two sites stays on one side) and, for generated music, by uploader account or prompt/caption group where the dataset "
         "records one (some do not; see below). The brief asked for 70/20/10; "
         "the larger share went to test so the confidence intervals are tighter. Validation is used only to keep the ResNet18's "
         "best epoch; every detector calls a clip AI at P(AI) >= 0.5 (no threshold tuning).",
         ug,
         "- **Unseen human:** train on one human source (FMA or Jamendo) and test on the other, plus GuitarSet solo acoustic "
         "recordings, which are also left out of training in these rounds.",
         "- **Vocal / instrumental:** train on one, test on both, in both directions.",
         "- **Generator-disjointness applies only to the unseen-generator rounds.** The matched round and both unseen-human "
         "rounds train and test on clips from the same generators by design"
         + (f" (in each of these {len(same_gen)} rounds, all {same_gen[0][2]} generators in the test set also appear in training)."
            if all_same_gen else
            " (" + "; ".join(f"{c}: {a} of {b} test generators also in training" for c, a, b in same_gen) + ").")
         + (" Both vocal/instrumental cross rounds include test generators that are absent from training because of the data, "
            "not by design: " + " and ".join(f"{len(d['novel_generators'])} in {k} ({', '.join(d['novel_generators'])})"
                                            for k, d in (("vocal → instrumental", vi), ("instrumental → vocal", iv))
                                            if d.get("novel_generators"))
            + "; see Known failure modes." if vi.get("novel_generators") or iv.get("novel_generators") else "")
         + " Only the unseen-generator rounds hold out generators on purpose.",
         "- Every round re-checks that no artist group, held-out generator or held-out source appears on both sides, "
         "and stops with an error if one does. For generated music the artist group is the uploader account, or the text prompt "
         "or caption where the dataset's metadata records one."
         + (f" {num.get(len(per_track), str(len(per_track))).capitalize()} AI datasets ({', '.join(per_track)}; {n_per_track:,} clips) "
            "have a different group key for every clip (no uploader or prompt is recorded for them), so they are split per track "
            "only and clips made from the same prompt can sit on both sides of a round."
            if per_track else "")
         + (" MusicGen's source (the HAIM dataset) does publish its prompts, but the download step does not read them for MusicGen. "
            + (f"{MUSICGEN_PROMPT_CHECK['n_shared']} of the {MUSICGEN_PROMPT_CHECK['n_test']} matched-test MusicGen clips share one of "
               f"HAIM's {MUSICGEN_PROMPT_CHECK['n_prompts']} prompts with a training clip (matched by file order and md5 in a separate "
               "check, not reproduced by this pipeline)."
               if n_mt_musicgen == MUSICGEN_PROMPT_CHECK["n_test"] else
               "A separate check outside this pipeline found that many matched-test MusicGen clips share a prompt with a training clip.")
            + f" MusicGen is {n_mt_musicgen} of the {len(mt_ai):,} AI clips in the matched test set."
            if "musicgen" in per_track else "")
         + " 95% intervals come from resampling whole artists; a Wilson interval over "
         "artists is used when a set has no errors (or only errors), where resampling would give a zero-width interval. "
         f"No interval is reported for a test set with fewer than {MIN_CLUSTERS} artists (e.g. GuitarSet's 6 players): "
         "treat those accuracies as point estimates.",
         "- **Genre** (the original hypothesis): for each genre with at least 30 AI and 30 human test clips, each detector except "
         "the ResNet18 is trained with and without it (training sets of equal size) and tested on it; the ResNet18 was skipped "
         "in these rounds to save CPU time. A dataset that is almost all one genre (GuitarSet: country/folk) is not scored in "
         "that genre's rounds, since removing the genre would remove the dataset too.",
         "- Exact duplicate audio that ended up on both sides of a round is removed from that round's scoring "
         "(results/excluded_test_clips.csv); see results/leakage_report.md for the duplicate, near-duplicate, shuffled-label "
         "and format-shortcut checks.",
         "- One 10-second excerpt per track, every file through the same decoder to 48 kHz; fixed random seed 0. "
         "The ResNet18 was trained on CPU with a budget (3-second crops, at most 3,000 clips per round, 4 epochs), far less "
         "than published ResNet18 baselines. No longer run was tested: more training may raise its matched score, but it could "
         "also learn more dataset-specific format traces and flag more unseen human music, so do not read these numbers as a "
         "lower or upper bound.", "",
         "## Unseen human music, by source", "",
         "One condition broken down by test source; read it next to the other three conditions in the table above.", "",
         UH[UH.detector == p][["round", "test_set", "n", "n_artists", "accuracy", "lo", "hi"]].rename(
             columns={"lo": "95% CI low", "hi": "95% CI high"}).to_markdown(index=False, floatfmt=".3f", missingval="-"), "",
         "## Datasets (clip counts cover all splits)", "",
         "| Dataset | Human/AI | Generator(s) | Clips | Tier | License |", "|---|---|---|---|---|---|"]
    for src, g in df.groupby("source"):
        s = SOURCES.get(src)
        L.append(f"| {src} | {'human' if g.label.iloc[0] == 0 else 'AI'} | {', '.join(sorted(g.model.unique()))[:60]} | {len(g)} | "
                 f"{', '.join(sorted(g.tier.unique()))} | {s.license if s else ''} |")
    L += ["", f"Clip counts above cover every split ({len(df):,} clips). {seen_txt}", "",
          "Provenance tiers: A = generated by this project, B = dataset with verified metadata, C = public reporting, "
          "D = assumed. Tier D never enters training. No present-day commercial releases were downloaded for training or testing "
          "(the public website does fetch store previews of them for each check; see Ethics and legal notes). Exception: FMA "
          f"contributed {n_hist} clips ("
          + ", ".join(f"{n} {k}" for k, n in hist_by.items())
          + f") tagged '{hist_lic}' ({hist_where}) that are digitised early-20th-century commercial 78-rpm records; "
            "**have a lawyer confirm them (or drop them) before publishing.** The named-artist case study "
            "uses the researcher's own copies and only scores are kept.", "",
          "## Known failure modes", "",
          "Each item below is one condition or stress test; read it next to the four-condition table above.", ""]
    worst = T.loc[p, "unseen_gen_worst_generator"] if "unseen_gen_worst_generator" in T else "?"
    L += [f"- **Unseen generators:** with one model version held out it caught {pct(row.get('unseen_gen_recall_mean'))} of that "
          f"version's tracks on average (worst: {worst}, {pct(row.get('unseen_gen_recall_worst'))}). With a whole family held out it "
          f"caught as few as {pct(row.get('unseen_family_recall_worst'))} (all {row.get('unseen_family_worst', '?')} versions; every family "
          "is in results/unseen_generator_table.csv); that is the better guide to music from a new company or architecture. "
          "Fingerprints can differ even between versions of one service: the fakeprint detector of Afchar et al. (ISMIR 2025), "
          "trained with Suno v3.5 and Udio-130, caught only 39.8% of tracks from Udio-32, an older Udio model it had not seen.",
          f"- **Vocal vs instrumental:** trained on vocal music it scores {pct(row.get('vocal_to_instr_balanced_accuracy'))} balanced "
          f"accuracy on instrumentals (vs {pct(row.get('instr_to_instr_balanced_accuracy'))} when trained on instrumentals)."
          + (f" Much of that gap is a change of generator, not of vocals: {vi['n_novel']} of the {vi['n_ai']} AI clips in that "
             f"instrumental test set come from {len(vi['novel_generators'])} generators that never appear in vocal training "
             f"({', '.join(vi['novel_generators'])}). The vocal-trained model catches {pct(vi['cross_recall_novel'])} of those "
             f"against {pct(vi['cross_recall_seen'])} of the rest (instrumental-trained: {pct(vi['same_recall_novel'])} and "
             f"{pct(vi['same_recall_seen'])}). Counting only AI clips from generators present in vocal training, its balanced "
             f"accuracy is {pct(vi['cross_balanced_accuracy_seen_only'])} instead of {pct(vi['cross_balanced_accuracy'])}. "
             f"Separately, it wrongly calls {pct(vi['cross_fpr'])} of human instrumentals AI against {pct(vi['same_fpr'])} for the "
             "instrumental-trained model; generator novelty does not explain that, since it affects AI clips only. So this round "
             "mixes two shifts (vocal to instrumental, and seen to unseen generators) and does not isolate the vocal/instrumental "
             "effect." if vi.get("n_novel") else "")
          + (f" The reverse round has the same problem on a smaller scale: trained on instrumentals it scores "
             f"{pct(iv['cross_balanced_accuracy'])} on vocal music (vs {pct(row.get('vocal_to_vocal_balanced_accuracy'))} when "
             f"trained on vocal music), but {iv['n_novel']} of its {iv['n_ai']} AI test clips come from "
             f"{len(iv['novel_generators'])} generators absent from its training ({', '.join(iv['novel_generators'])}). It catches "
             f"{pct(iv['cross_recall_novel'])} of those against {pct(iv['cross_recall_seen'])} of the rest, and counting only "
             f"generators present in its training its balanced accuracy is {pct(iv['cross_balanced_accuracy_seen_only'])}."
             if iv.get("n_novel") else "")]
    # format shortcut, measured: the robustness 'bandwidth' edits and the unseen-FMA tracks stored below 44.1 kHz
    shortcut = "- **Format shortcuts:** the datasets differ in sample rate, channels and MP3 bitrate (results/leakage_report.md)."
    if rob is not None:
        r = rob[rob.detector == p].set_index("transform")
        bw = r[r.index.str.startswith("bandwidth")]
        if len(bw):
            lowsr = df[df.orig_sr.between(1, 32000)]
            f = RESULTS / "predictions" / p / "unseen_human__train_human=jamendo_test=fma.csv"
            extra = ""
            if f.exists():
                P = pd.read_csv(f).merge(df[["clip_id", "orig_sr", "source"]], on="clip_id")
                low = P[(P.label == 0) & P.source.str.startswith("fma") & P.orig_sr.between(1, 44099)]
                if len(low):
                    extra = (f" Unseen FMA tracks stored below 44.1 kHz were called AI in {int((low.p_ai >= 0.5).sum())} of {len(low)} "
                             "cases.")
            clean_fpr = r.loc["clean", "fpr"] if "clean" in r.index else np.nan
            # genuinely old recordings: the headline set (FMA pre-2010) split by content (almost no sound above 8 kHz vs the rest)
            old_rec = ""
            ff = ROOT / "data" / "features.csv"
            if f.exists() and ff.exists():
                Q = (pd.read_csv(f).merge(df[["clip_id", "source", "license"]], on="clip_id")
                     .merge(pd.read_csv(ff, usecols=["clip_id", "band_8k_12k_db"]), on="clip_id", how="left"))
                Q = Q[(Q.label == 0) & (Q.source == "fma_pre2010")]
                lim = Q.band_8k_12k_db.isna() | (Q.band_8k_12k_db < -45)
                n_78 = int(Q[lim].license.astype(str).str.contains(HIST_LICENCE, regex=False).sum())
                old_rec = (f" Genuinely old recordings are still harder: it calls {pct((Q[lim].p_ai >= 0.5).mean())} of the "
                           f"{int(lim.sum())} unseen pre-2010 FMA tracks with almost no sound above 8 kHz AI (old or lo-fi recordings, "
                           f"{n_78} of them tagged as historic 78-rpm records), against {pct((Q[~lim].p_ai >= 0.5).mean())} of the "
                           f"other {int((~lim).sum())}.")
            present = bw.fpr.max() > clean_fpr + 0.10  # same rule as evaluate.py's BAND_RISE
            measured = ("; ".join(f"{pct(x.fpr)} of human clips AI after {t}" for t, x in bw.iterrows())
                        + f" (unedited: {pct(clean_fpr)})")
            before = ""
            old = RESULTS / "resnet18_16k_before_robustness.csv"
            if is_resnet and old.exists():
                ob = pd.read_csv(old).set_index("transform")
                ob_bw = ob[ob.index.str.startswith("bandwidth")]
                before = (" An earlier version of this detector, which saw up to 16 kHz, had learned the cue: it wrongly called "
                          + "; ".join(f"{pct(x.fpr)} of human clips AI after {t}" for t, x in ob_bw.iterrows())
                          + f" (unedited: {pct(ob.loc['clean', 'fpr'])}). It was retrained with the spectrogram capped at "
                          f"{FMAX / 1000:g} kHz, below the lowest bandwidth of any file in the data (results/resnet18_16k_before*.csv).")
            shortcut = (f"- **Format shortcut ({'measured' if present else 'checked'}):** almost every clip whose original file was 32 kHz "
                        f"or lower is AI ({int((lowsr.label == 1).sum())} of {len(lowsr)}: "
                        + ", ".join(sorted(lowsr[lowsr.label == 1].model.value_counts().index[:5])) + ")." + before
                        + (f" This detector sees up to {FMAX / 1000:g} kHz." if is_resnet else "")
                        + f" On the robustness test clips it wrongly called {measured}." + extra
                        + (" Do not rely on it for old, phone-quality, low-sample-rate or low-bitrate recordings; predict.py warns "
                           "when a file is below 44.1 kHz, at 96 kbps or less, or has almost no sound above 8 kHz."
                           if present else
                           (f" So band-limiting at 8 or 16 kHz, above the {FMAX / 1000:g} kHz this detector sees, no longer changes its "
                            "verdict; narrower band limits inside its range (phone quality, for example) were not tested."
                            if is_resnet else " So band-limiting a recording at 8 or 16 kHz no longer changes its verdict.") + old_rec))
        e = r.drop(index="clean", errors="ignore")
        e = e[~e.index.str.startswith("bandwidth")].sort_values("balanced_accuracy").head(3)
        if len(e):
            L.append("- **Everyday edits:** worst cases " + "; ".join(
                f"{t}: balanced accuracy {pct(x.balanced_accuracy)} (missed AI {pct(x.fnr)}, accused humans {pct(x.fpr)})"
                for t, x in e.iterrows()) + f" vs clean {pct(r.loc['clean', 'balanced_accuracy']) if 'clean' in r.index else 'n/a'}.")
    L.append(shortcut)
    gt = RESULTS / "genre_table.csv"
    if gt.exists() and p in set(pd.read_csv(gt).detector):
        L.append("- **Genre:** see results/genre_table.csv - performance on a genre removed from training vs the same genre seen in training.")
    else:
        L.append("- **Genre:** not measured for this detector (it was skipped in the genre rounds to save CPU time); "
                 "results/genre_table.csv covers the other detectors only.")
    L += ["- **Hybrid tracks** (AI stems mixed with human performance) were not evaluated (out of scope).", "",
          "## Features the interpretable models use", "", "| Feature | Group | What it measures and why it should differ |", "|---|---|---|"]
    for n in F.names():
        L.append(f"| `{n}` | {F.REGISTRY[n][1]} | {F.doc(n)} |")
    L += ["", "The optional harmony features (harm_nondiatonic_rate, harm_chord_change_rate, harm_key_instability) are not used "
          "by any trained model; they were tested separately on a sample (results/harmony_probe.md).",
          "", "## Ethics and legal notes (not legal advice)", "",
          "- Deezer's fakeprint method is published under CC BY-NC 4.0 and covered by Deezer patent applications: fine for "
          "non-commercial research, **ask a lawyer before any commercial or public deployment.**",
          f"- The AI-music datasets carry mixed terms (see the Datasets table): {lic_mix}. A model's licence is not necessarily the "
          "licence of the audio it made, and a claimed licence is the uploader's own statement. A further "
          f"{n_scraped:,} AI clips ({scraped_src}) come from a scraped collection with no clear licence: research only, do not "
          "redistribute the audio.",
          f"- FMA human tracks are licensed per track: {n_nc:,} of {len(fma):,} are NonCommercial and {n_nd:,} are NoDerivatives; "
          f"{n_hist} are historic commercial records (see Datasets). Features, embeddings, fakeprints, scores, the manifest (which "
          "also holds scraped platform user IDs) and the trained models are all derived from this audio: **ask a lawyer before "
          "releasing any of them publicly.**",
          *([f"- The per-track licences of the {len(jam):,} Jamendo clips (" + ", ".join(sorted(jam.source.unique())) + ") were not "
             "recorded: the manifest holds only the generic note" + ("s " if jam.license.nunique() > 1 else " ")
             + " / ".join(f"'{x}'" for x in sorted(jam.license.astype(str).unique())) + ". So how many of them are NonCommercial "
             "or NoDerivatives is unknown. MTG-Jamendo publishes its per-track licences (audio_licenses.txt); they have not yet "
             "been joined to the manifest."] if jam_generic else []),
          "- The trained models (kept private; the public repository has only models/config.json) are not cleared for release."
          + (" resnet18_mel.pt starts from torchvision's ImageNet-pretrained ResNet18 (torchvision notes such weights may carry the "
             "terms of their training data; ImageNet's terms limit the images to non-commercial research and education)." if is_resnet else "")
          + " Whether the audio's licence terms carry over to trained weights is unsettled: **ask a lawyer before publishing the "
            "trained weights or using any of them commercially.**",
          f"- **The public website** ({WEBSITE}) went live before this legal review. For each check it fetches the song's official "
          "30-second preview from the Deezer API or Apple's iTunes Search API to a temporary file on the server and deletes it after "
          "scoring. Apple's Search API terms say previews may be used only to promote store content, must be streamed and not "
          "downloaded, saved or cached, must be shown near a store badge and credited 'provided courtesy of iTunes'. Deezer's API "
          "guidelines (developers.deezer.com/guidelines) forbid storing audio locally or offline, and its API Terms of Use "
          "(developers.deezer.com/termsofuse, section IV) limit use to non-commercial purposes. The site also reads song "
          "titles from public Spotify track pages and from YouTube's oEmbed service (falling back to the watch page), and it shows "
          "the artist's name next to a verdict. **A lawyer should look at all of it.**",
          "- AI generators' own terms of service may bind the people who built the third-party AI datasets used here. For example, "
          "Suno's terms (effective 3 September 2026) forbid using Outputs to train other AI or machine-learning models and forbid "
          "scraping, and free-tier outputs are for personal, non-commercial use. Whether that affects training or publishing a "
          "detector on those clips is a question for the legal review.",
          "- A public tool that labels named artists as AI can harm careers and could raise defamation questions: "
          "**get legal advice before publishing per-artist verdicts**, use a precision-tuned threshold and offer an appeals path.",
          "- US law has no text-and-data-mining exception; research use of copyrighted audio is a fair-use question "
          "(US Copyright Office, Part 3 report, 2025). The EU/UK research exceptions require lawful access."]
    (ROOT / "MODEL_CARD.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote MODEL_CARD.md")


if __name__ == "__main__":
    main()
