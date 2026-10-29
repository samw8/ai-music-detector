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

from aimd import MODELS as MODEL_DIR  # noqa: E402
from aimd import RESULTS  # noqa: E402
from aimd import features as F  # noqa: E402
from aimd.models import label  # noqa: E402
from aimd.sources import SOURCES  # noqa: E402
from aimd.cnn import FMAX  # noqa: E402
from aimd.metrics import MIN_CLUSTERS  # noqa: E402
from aimd.splits import MIN_GEN_CLIPS, load_table  # noqa: E402


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
        verdicts = (f" These rates count P(AI) >= 0.5 as an AI call. predict.py prints LIKELY AI-GENERATED only at P(AI) >= {hi}, "
                    f"LIKELY HUMAN-MADE at <= {lo} and UNCERTAIN in between; at {hi} it still flagged {pct(fh)} of those FMA tracks "
                    f"and {pct(jh)} of those Jamendo tracks, and put {pct(fu)} and {pct(ju)} of them in UNCERTAIN.")
    is_resnet = p.split("@")[0] == "resnet18_mel"
    # what the shipped model saw: the matched round's training clips (and, for the ResNet, a balanced 3,000 of them)
    from aimd.cnn import _balanced_subset
    from aimd.splits import make_rounds

    mr = next(r for r in make_rounds(df, which=("matched",)) if r.condition == "matched")
    y = df.label.to_numpy()
    seen_txt = (f"The matched split puts {len(mr.train):,} clips in training, {len(mr.val):,} in validation and {len(mr.test):,} in test; "
                f"the models in models/ were trained on the {len(mr.train):,} training clips only.")
    if is_resnet:
        sub, _ = _balanced_subset(np.asarray(mr.train), y[mr.train], 3000, 0)
        s = df.iloc[sub]
        hc = s[s.label == 0].source.value_counts()
        seen_txt += (f" The shipped ResNet18 saw a class-balanced random {len(sub):,} of them ({int((s.label == 0).sum()):,} human: "
                     + ", ".join(f"{n} {k}" for k, n in hc.items())
                     + f"; {int((s.label == 1).sum()):,} AI from {s[s.label == 1].model.nunique()} generators); a balanced 600 of the "
                       f"{len(mr.val):,} validation clips picked its best epoch.")
    fma = df[df.source.str.startswith("fma")]
    lic = fma.license.astype(str).str.lower()
    n_hist = int(lic.str.contains("common law").sum())
    hist_by = fma[lic.str.contains("common law")].source.value_counts().to_dict()
    n_nd, n_nc = int(lic.str.contains("noderiv|no deriv").sum()), int(lic.str.contains("noncommercial").sum())
    n_scraped = int(df.license.astype(str).str.startswith("unclear").sum())
    scraped_src = ", ".join(sorted(df[df.license.astype(str).str.startswith("unclear")].source.unique()))

    L = [f"# Model card - {label(p)}", "",
         "> **Read this first.** Retrained with FMA left out of its training data, this detector called "
         f"**{pct(1 - row.get('unseen_human_acc_fma_pre2010'))} of FMA tracks uploaded before 2010** AI-generated; retrained "
         f"without Jamendo, it called **{pct(1 - row.get('unseen_human_acc_jamendo'))} of Jamendo tracks** AI-generated. "
         "Every one of those is a false accusation of a real musician. Expect rates like these on music unlike its training data "
         "(the copy shipped in models/ was trained on FMA, Jamendo and four of the six GuitarSet players too). Rates are per 10-second excerpt."
         + verdicts + (" **Old, phone-quality or low-sample-rate human recordings are called AI far more often** (see Known failure modes)."
                       if cfg.get("bandlimit_warning") else "") + " A score is evidence, not proof.", "",
         "## What it is", "",
         f"- Binary classifier: P(AI-generated) for a 10-second excerpt at 48 kHz (predict.py averages up to three excerpts per song).",
         *(["- ResNet18 initialised from torchvision's ImageNet-1k weights (IMAGENET1K_V1), first layer summed to one input channel; "
            f"input = mono 128-band log-mel, 20 Hz-{FMAX / 1000:g} kHz (capped there so a missing top octave cannot separate "
            "the classes; see Known failure modes). It has no per-feature explanation: the 'why' lines predict.py prints come from a "
            "separate, simpler model (interpretable features + logistic regression), are not this detector's reasons, and can disagree with it."]
           if is_resnet else []),
         f"- Chosen as the main detector because it has the best *worst-case* score across the four evaluation conditions "
         f"({cfg['worst_case_score']:.3f}); see results/results_table.md for every model.",
         "- Intended use: research on AI-music detection; triage that a human then checks. **Not** for accusing, "
         "demonetising or removing an artist's music.", "",
         "## Results in all four conditions (never quote one without the others)", "",
         (RESULTS / "results_table.md").read_text(encoding="utf-8"), "",
         "Balanced accuracy = average of the share of AI tracks caught and the share of human tracks correctly passed. "
         "Unseen human = accuracy on human-only music from a source absent from training (= 1 - false-positive rate).", "",
         "## How it was tested", "",
         "- **Matched:** 70% train / 10% validation / 20% test, split by artist (a cross-platform artist key, so the same act "
         "on two sites stays on one side) and by prompt/caption group for generated music. The brief asked for 70/20/10; "
         "the larger share went to test so the confidence intervals are tighter. Validation is used only to keep the ResNet18's "
         "best epoch; every detector calls a clip AI at P(AI) >= 0.5 (no threshold tuning).",
         ug,
         "- **Unseen human:** train on one human source (FMA or Jamendo) and test on the other, plus GuitarSet solo acoustic "
         "recordings, which are also left out of training in these rounds.",
         "- **Vocal / instrumental:** train on one, test on both, in both directions.",
         "- Every round re-checks that no artist, prompt group, held-out generator or held-out source appears on both sides, "
         "and stops with an error if one does. 95% intervals come from resampling whole artists; a Wilson interval over "
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
          "D = assumed. Tier D never enters training. No present-day commercial releases were downloaded. Exception: FMA's "
          f"'Old-Time / Historic' genre contributed {n_hist} clips ("
          + ", ".join(f"{n} {k}" for k, n in hist_by.items())
          + ") that are digitised early-20th-century commercial 78-rpm records, which FMA tags 'Public Domain / Sound Recording "
            "Common Law Protection'; **have a lawyer confirm them (or drop them) before publishing.** The named-artist case study "
            "uses the researcher's own copies and only scores are kept.", "",
          "## Known failure modes", ""]
    worst = T.loc[p, "unseen_gen_worst_generator"] if "unseen_gen_worst_generator" in T else "?"
    L += [f"- **Unseen generators:** with one model version held out it caught {pct(row.get('unseen_gen_recall_mean'))} of that "
          f"version's tracks on average (worst: {worst}, {pct(row.get('unseen_gen_recall_worst'))}). With a whole family held out it "
          f"caught as few as {pct(row.get('unseen_family_recall_worst'))} (all {row.get('unseen_family_worst', '?')} versions; every family "
          "is in results/unseen_generator_table.csv); that is the better guide to music from a new company or architecture. "
          "Fingerprints can differ even between versions of one service: the fakeprint detector of Afchar et al. (ISMIR 2025), "
          "trained with Suno v3.5 and Udio-130, caught only 39.8% of tracks from Udio-32, an older Udio model it had not seen.",
          f"- **Vocal vs instrumental:** trained on vocal music it scores {pct(row.get('vocal_to_instr_balanced_accuracy'))} balanced "
          f"accuracy on instrumentals (vs {pct(row.get('instr_to_instr_balanced_accuracy'))} when trained on instrumentals)."]
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
                Q = (pd.read_csv(f).merge(df[["clip_id", "source"]], on="clip_id")
                     .merge(pd.read_csv(ff, usecols=["clip_id", "band_8k_12k_db"]), on="clip_id", how="left"))
                Q = Q[(Q.label == 0) & (Q.source == "fma_pre2010")]
                lim = Q.band_8k_12k_db.isna() | (Q.band_8k_12k_db < -45)
                old_rec = (f" Genuinely old recordings are still harder: of the {len(Q)} unseen pre-2010 FMA tracks, the {int(lim.sum())} with "
                           f"almost no sound above 8 kHz (old or lo-fi recordings, including 78-rpm transfers) are called AI {pct((Q[lim].p_ai >= 0.5).mean())} of the time, "
                           f"against {pct((Q[~lim].p_ai >= 0.5).mean())} for the rest.")
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
          "by any shipped model; they were tested separately on a sample (results/harmony_probe.md).",
          "", "## Ethics and legal notes (not legal advice)", "",
          "- Deezer's fakeprint method is published under CC BY-NC 4.0 and covered by Deezer patent applications: fine for "
          "non-commercial research, **ask a lawyer before any commercial or public deployment.**",
          f"- The AI-music datasets used here are CC BY-NC or research-only, and {n_scraped:,} AI clips ({scraped_src}) come from a "
          "scraped collection with no clear licence: research only, do not redistribute the audio.",
          f"- FMA human tracks are licensed per track: {n_nc:,} of {len(fma):,} are NonCommercial and {n_nd:,} are NoDerivatives; "
          f"{n_hist} are historic commercial records (see Datasets). Features, embeddings, fakeprints, scores, the manifest (which "
          "also holds scraped platform user IDs) and the trained models are all derived from this audio: **ask a lawyer before "
          "releasing any of them publicly.**",
          "- The trained models in models/ are not cleared for release."
          + (" resnet18_mel.pt starts from torchvision's ImageNet-pretrained ResNet18 (torchvision notes such weights may carry the "
             "terms of their training data; ImageNet's terms limit the images to non-commercial research and education)." if is_resnet else "")
          + " Whether the audio's licence terms carry over to trained weights is unsettled: **ask a lawyer before publishing models/ "
            "or using any shipped model commercially.**",
          "- A public tool that labels named artists as AI can harm careers and could raise defamation questions: "
          "**get legal advice before publishing per-artist verdicts**, use a precision-tuned threshold and offer an appeals path.",
          "- US law has no text-and-data-mining exception; research use of copyrighted audio is a fair-use question "
          "(US Copyright Office, Part 3 report, 2025). The EU/UK research exceptions require lawful access."]
    (ROOT / "MODEL_CARD.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote MODEL_CARD.md")


if __name__ == "__main__":
    main()
