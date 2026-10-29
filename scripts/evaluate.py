"""Step 13 - turn the saved predictions into the results tables and figures.

    python scripts/evaluate.py

Writes to results/:
    results_table.csv / .md       one row per model, all four conditions side by side (the paper table)
    metrics_by_round.csv          every model x every round, all metrics with 95% artist-bootstrap CIs
    unseen_human_breakdown.csv    accuracy on each unseen human source (FMA pre-2010, Jamendo, GuitarSet)
    unseen_generator_table.csv    share of each held-out generator's tracks caught, per model
    vocal_instr_table.csv, genre_table.csv
    figures/*.png                 the same numbers as figures, plus a confusion matrix per condition
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import REPO, RESULTS  # noqa: E402
from aimd import plotstyle as S  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402  (after plotstyle, which selects the headless backend)
from aimd.metrics import bootstrap_ci, summarize  # noqa: E402
from aimd.models import PRETRAINED, label, ordered  # noqa: E402
from aimd.splits import load_table, make_rounds  # noqa: E402

FIG = RESULTS / "figures"
# Published detectors whose own training data overlaps a test source: their score there is not "unseen".
IN_DOMAIN = {"lofcz_fakeprint": {"FMA uploaded before 2010", "FMA, all"}}
# ... and generators they were trained on (SONICS: Suno v3.5 and Udio-130 era) are not "unseen generators" for them
IN_DOMAIN_GEN = {"spectttra_alpha5s": {"suno_v3.5", "udio_v1"}, "lofcz_fakeprint": {"suno_v3.5", "udio_v1"}}


def panel_title(d: str, n: int = 30) -> str:
    """Short figure title that always keeps the variant tag (e.g. '[MP3-equalised]')."""
    base, _, var = d.partition("@")
    t = (label(base).replace("Off the shelf: ", "Off-shelf ").replace(" (published method, retrained here)", "")
         .replace(" (not retrained)", ""))
    t = t if len(t) <= n else t[: n - 1] + "…"
    return t + (" [MP3-equalised]" if var == "mp3" else (f" [{var}]" if var else ""))
CI_METRICS = ("balanced_accuracy", "fpr", "recall")


def load_predictions(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = ["clip_id", "artist_id", "source", "model", "family", "human_group", "vocal", "genre"]
    meta = table[cols]
    for f in sorted((RESULTS / "predictions").glob("*/*.csv")):
        cond, rnd = f.stem.split("__", 1)
        p = pd.read_csv(f).merge(meta, on="clip_id", how="left")
        p["detector"], p["condition"], p["round"] = f.parent.name, cond, rnd
        rows.append(p)
    if not rows:
        sys.exit("No predictions found - run scripts/run_experiments.py first.")
    P = pd.concat(rows, ignore_index=True)
    ex = RESULTS / "excluded_test_clips.csv"  # written by leakage_check.py: test clips with an exact copy in training
    if ex.exists():
        E = pd.read_csv(ex)
        if len(E):
            key = set(zip(E.condition, E["round"], E.clip_id))
            drop = [k in key for k in zip(P.condition, P["round"], P.clip_id)]
            print(f"[evaluate] left out {sum(drop)} test predictions whose clip has an exact duplicate in training")
            P = P[~np.array(drop)]
    return P


def score(g: pd.DataFrame) -> dict:
    s = summarize(g.label, g.p_ai)
    s["n_artists"] = int(g.artist_id.nunique())
    for m in CI_METRICS:
        if np.isnan(s.get(m, np.nan)):
            s[f"{m}_lo"] = s[f"{m}_hi"] = np.nan
        else:
            s[f"{m}_lo"], s[f"{m}_hi"] = bootstrap_ci(g.label, g.p_ai, g.artist_id, m)
    s["specificity_lo"], s["specificity_hi"] = 1 - s["fpr_hi"], 1 - s["fpr_lo"]
    return s


def by_round(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (d, c, r), g in P.groupby(["detector", "condition", "round"]):
        rows.append({"detector": d, "condition": c, "round": r, **score(g)})
    return pd.DataFrame(rows)


def unseen_human_breakdown(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    U = P[P.condition == "unseen_human"]
    subsets = {"FMA uploaded before 2010": lambda g: g.source == "fma_pre2010",
               "FMA, all": lambda g: g.human_group == "fma",
               "Jamendo (MTG + MuseBench)": lambda g: g.human_group == "jamendo",
               "GuitarSet (solo acoustic)": lambda g: g.human_group == "guitarset",
               "AI tracks (seen generators)": lambda g: g.label == 1}
    for (d, r), g in U.groupby(["detector", "round"]):
        for name, sel in subsets.items():
            h = g[sel(g)]
            if len(h) < 20:
                continue
            s = score(h)
            is_ai = name.startswith("AI")
            rows.append({"detector": d, "round": r, "test_set": name, "n": len(h), "n_artists": s["n_artists"],
                         "in_domain_for_detector": name in IN_DOMAIN.get(d, set()),
                         "accuracy": s["recall"] if is_ai else s["specificity"],
                         "lo": s["recall_lo"] if is_ai else s["specificity_lo"],
                         "hi": s["recall_hi"] if is_ai else s["specificity_hi"],
                         "fpr": np.nan if is_ai else s["fpr"]})
    return pd.DataFrame(rows)


def paper_table(R: pd.DataFrame, UH: pd.DataFrame, expected: pd.Series) -> pd.DataFrame:
    """expected: how many rounds make_rounds() builds for each condition."""
    out = []
    n_rounds = R.groupby(["detector", "condition"]).size()
    for d in ordered(R.detector):
        row = {"model": label(d), "detector": d}
        # a detector still training on some rounds is flagged, so a partial mean is never read as a full one
        conds = ["matched", "unseen_generator", "unseen_human", "vocal_instr"]
        if d.split("@")[0] != "resnet18_mel":  # run_experiments.py skips genre for the ResNet only
            conds.append("genre")
        row["partial"] = "; ".join(f"{c.replace('_', ' ')} {n_rounds.get((d, c), 0)}/{expected[c]} rounds"
                                   for c in conds if expected.get(c, 0) and n_rounds.get((d, c), 0) < expected[c])
        m = R[(R.detector == d) & (R.condition == "matched")]
        if len(m):
            m = m.iloc[0]
            for k in ("balanced_accuracy", "precision", "recall", "fpr", "fnr", "eer", "auc"):
                row[f"matched_{k}"] = m[k]
        g = R[(R.detector == d) & (R.condition == "unseen_generator") & ~R["round"].str.startswith("family")
              & ~R["round"].isin(IN_DOMAIN_GEN.get(d.split("@")[0], set()))]
        if len(g):
            row["unseen_gen_balanced_accuracy_mean"] = g.balanced_accuracy.mean()
            row["unseen_gen_recall_mean"] = g.recall.mean()
            worst = g.loc[g.recall.idxmin()]
            row["unseen_gen_recall_worst"] = worst.recall
            row["unseen_gen_worst_generator"] = worst["round"]
            row["unseen_gen_fpr_mean"] = g.fpr.mean()
            row["unseen_gen_eer_mean"] = g.eer.mean()
        # whole-family hold-outs (all Suno versions, all Udio versions, ...): the harder, more realistic case
        fam = R[(R.detector == d) & (R.condition == "unseen_generator") & R["round"].str.startswith("family")]
        if d.split("@")[0] in IN_DOMAIN_GEN:  # a published detector trained on Suno/Udio has not "unseen" those families
            fam = fam[~fam["round"].isin({"family_" + x.split("_")[0] for x in IN_DOMAIN_GEN[d.split("@")[0]]})]
        if len(fam):
            w = fam.loc[fam.recall.idxmin()]
            row["unseen_family_balanced_accuracy_mean"] = fam.balanced_accuracy.mean()
            row["unseen_family_recall_worst"] = w.recall
            row["unseen_family_worst"] = w["round"][len("family_"):]
        u = UH[(UH.detector == d) & ~UH.in_domain_for_detector.astype(bool)]
        for name, key in (("FMA uploaded before 2010", "unseen_human_acc_fma_pre2010"), ("Jamendo (MTG + MuseBench)", "unseen_human_acc_jamendo"),
                          ("GuitarSet (solo acoustic)", "unseen_human_acc_guitarset")):
            h = u[u.test_set == name]
            if len(h):
                h = h.iloc[0] if name != "GuitarSet (solo acoustic)" else h.sort_values("accuracy").iloc[0]
                row[key], row[key + "_lo"], row[key + "_hi"] = h.accuracy, h.lo, h.hi
        v = R[(R.detector == d) & (R.condition == "vocal_instr")].set_index("round")
        for rk, key in (("train=vocal_test=instrumental", "vocal_to_instr"), ("train=vocal_test=vocal", "vocal_to_vocal"),
                        ("train=instrumental_test=vocal", "instr_to_vocal"), ("train=instrumental_test=instrumental", "instr_to_instr")):
            if rk in v.index:
                row[f"{key}_balanced_accuracy"] = v.loc[rk, "balanced_accuracy"]
                row[f"{key}_fpr"] = v.loc[rk, "fpr"]
        out.append(row)
    return pd.DataFrame(out)


BAND_RISE = 0.10  # a band-limit edit that raises the human false-positive rate by more than this counts as a shortcut

WORST_CASE_COLS = ["matched_balanced_accuracy", "unseen_gen_balanced_accuracy_mean", "unseen_human_acc_fma_pre2010",
                   "unseen_human_acc_jamendo", "vocal_to_instr_balanced_accuracy", "instr_to_vocal_balanced_accuracy"]


def write_config(T: pd.DataFrame) -> dict:
    """Pick predict.py's main detector by its WORST condition (not its best), and write the caveat it prints."""
    import json

    from aimd import MODELS as MODEL_DIR
    from aimd.models import PRETRAINED

    cand = T[~T.detector.isin(PRETRAINED)].copy()
    cand = cand[(MODEL_DIR / (cand.detector + ".joblib")).map(lambda p: p.exists())
                | ((cand.detector == "resnet18_mel") & (MODEL_DIR / "resnet18_mel.pt").exists())]
    cols = [c for c in WORST_CASE_COLS if c in cand and cand[c].notna().any()]
    complete = cand[cols].notna().all(axis=1) & (cand.partial.fillna("") == "")
    if (~complete).any():
        print("[config] not eligible yet (missing conditions): " + ", ".join(cand.detector[~complete]))
    cand = cand[complete].copy()
    if cand.empty:
        print("[config] no detector has results for every condition yet; models/config.json not written")
        return {}
    cand["worst_case"] = cand[cols].min(axis=1, skipna=False)
    best = cand.sort_values("worst_case", ascending=False).iloc[0]
    pct = lambda x: "n/a" if pd.isna(x) else f"{100 * x:.0f}%"  # noqa: E731
    lo, hi = 0.2, 0.8  # predict.py: LIKELY HUMAN-MADE <= lo < UNCERTAIN < hi <= LIKELY AI-GENERATED
    edits, band_warn, band_all = "Pitch/speed edits and re-encoding change scores (results/robustness.csv).", "", ""
    rob = RESULTS / "robustness.csv"
    if rob.exists():  # name the edits that hurt this detector most, measured on the same test clips
        Rb = pd.read_csv(rob)
        Rb = Rb[Rb.detector == best.detector]
        clean = Rb.loc[Rb["transform"] == "clean", "balanced_accuracy"]
        bw = Rb[Rb["transform"].str.startswith("bandwidth")]
        Ed = Rb[(Rb["transform"] != "clean") & ~Rb["transform"].str.startswith("bandwidth")].sort_values("balanced_accuracy")
        if len(Ed):
            edits = ("Its worst everyday edits: " + "; ".join(f"after {t}, balanced accuracy {pct(b)}" for t, b in
                                                             zip(Ed["transform"][:3], Ed.balanced_accuracy[:3]))
                     + (f" (unedited: {pct(clean.iloc[0])}; " if len(clean) else " (") + "50% = coin flip; results/robustness.csv).")
        clean_fpr = float(Rb.loc[Rb["transform"] == "clean", "fpr"].iloc[0]) if len(clean) else np.nan
        # old, phone-quality or low-sample-rate recordings: almost every 16/32 kHz clip in the data is AI. Warn only if this
        # detector actually accuses band-limited humans much more often (more than 10 points above its unedited rate)
        if len(bw) and bw.fpr.max() > clean_fpr + BAND_RISE:
            w = bw.loc[bw.fpr.idxmax()]
            band_all = (f"Human recordings with little sound above 8 kHz (old, phone-quality or low-sample-rate audio, even when saved as an "
                        f"ordinary 44.1/48 kHz file) are often called AI: after {w['transform']}, {pct(w.fpr)} of human test clips "
                        f"(unedited: {pct(Rb.loc[Rb['transform'] == 'clean', 'fpr'].iloc[0]) if len(clean) else 'n/a'}). ")
            band_warn = ("The main detector treats missing high frequencies as a sign of AI: after "
                         + "; after ".join(f"{t}, it wrongly called {pct(f)} of human test clips AI" for t, f in zip(bw["transform"], bw.fpr))
                         + f" (unedited: {pct(Rb.loc[Rb['transform'] == 'clean', 'fpr'].iloc[0]) if len(clean) else 'n/a'}). "
                           "So a high P(AI) for this file is weak evidence.")
    # how the unseen-human clips fall into predict.py's three verdicts (the rates above count P(AI) >= 0.5 as an AI call)
    M = pd.read_csv(REPO / "data" / "manifest.csv", usecols=["clip_id", "source"])

    def bands(train: str, test: str, srcs: list[str]):
        f = RESULTS / "predictions" / best.detector / f"unseen_human__train_human={train}_test={test}.csv"
        if not f.exists():
            return np.nan, np.nan
        d = pd.read_csv(f).merge(M, on="clip_id")
        p = d[(d.label == 0) & d.source.isin(srcs)].p_ai
        return (p >= hi).mean(), ((p > lo) & (p < hi)).mean()

    fh, fu = bands("jamendo", "fma", ["fma_pre2010"])
    jh, ju = bands("fma", "jamendo", ["mtg_jamendo", "mb_jamendo"])
    # with no band-limit shortcut there is no per-file warning, so say in the caveat how genuinely old recordings fare
    old_rec = ""
    fu_csv, ff = RESULTS / "predictions" / best.detector / "unseen_human__train_human=jamendo_test=fma.csv", REPO / "data" / "features.csv"
    if not band_warn and fu_csv.exists() and ff.exists():
        Q = (pd.read_csv(fu_csv).merge(M, on="clip_id")
             .merge(pd.read_csv(ff, usecols=["clip_id", "band_8k_12k_db"]), on="clip_id", how="left"))
        Q = Q[(Q.label == 0) & (Q.source == "fma_pre2010")]
        lim = Q.band_8k_12k_db.isna() | (Q.band_8k_12k_db < -45)  # NaN = real bandwidth ends below ~10 kHz
        if lim.any() and (~lim).any():
            old_rec = (f"Old or lo-fi human recordings (almost no sound above 8 kHz) are still wrongly called AI more often: "
                       f"{pct((Q[lim].p_ai >= 0.5).mean())} of such old FMA tracks, against {pct((Q[~lim].p_ai >= 0.5).mean())} of the rest. ")
    caveat = (f"When this detector type was retrained with FMA left out, it wrongly called "
              f"{pct(1 - best.get('unseen_human_acc_fma_pre2010', np.nan))} of FMA tracks (uploaded before 2010) AI-generated; "
              f"retrained without Jamendo, {pct(1 - best.get('unseen_human_acc_jamendo', np.nan))} of Jamendo tracks "
              f"(counting P(AI) >= 0.5 as an AI call; this tool says LIKELY AI-GENERATED only at P(AI) >= {hi}, which still flagged "
              f"{pct(fh)} and {pct(jh)} of those tracks, and put {pct(fu)} and {pct(ju)} in UNCERTAIN). Treat those as "
              "estimates of how often it accuses real musicians whose music is unlike its training data (the copy shipped here did "
              f"train on FMA, Jamendo and some GuitarSet). Retrained without a given AI generator, it caught {pct(best.get('unseen_gen_recall_mean', np.nan))} "
              f"of that generator's tracks on average (worst: {best.get('unseen_gen_worst_generator', '?')}, "
              f"{pct(best.get('unseen_gen_recall_worst', np.nan))}); with a whole model family held out (like music from a new company), "
              f"as few as {pct(best.get('unseen_family_recall_worst', np.nan))} (all {best.get('unseen_family_worst', '?')} versions). "
              f"Retrained on vocal music only, it scored {pct(best.get('vocal_to_instr_balanced_accuracy', np.nan))} balanced accuracy on "
              f"instrumentals and wrongly called {pct(best.get('vocal_to_instr_fpr', np.nan))} of human instrumentals AI-generated "
              f"(retrained on instrumentals only: {pct(best.get('instr_to_vocal_balanced_accuracy', np.nan))} on vocal music). "
              f"On the matched test split (new artists, otherwise familiar music) it scored {pct(best.get('matched_balanced_accuracy', np.nan))} "
              f"and wrongly called {pct(best.get('matched_fpr', np.nan))} of human tracks AI-generated. "
              f"Rates are per 10-second excerpt. {edits} {band_all}{old_rec}"
              "A score is one piece of evidence, never proof that a musician used AI.")
    cfg = {"primary": best.detector, "selected_by": "highest worst-case score over: " + ", ".join(cols),
           "worst_case_score": float(best.worst_case), "uncertain_band": [lo, hi], "caveat": caveat,
           # share of unseen human tracks predict.py would call LIKELY AI (>= hi) and UNCERTAIN, for the model card
           "unseen_human_verdicts": {"fma_pre2010": [fh, fu], "jamendo": [jh, ju]}}
    if band_warn:
        cfg["bandlimit_warning"] = band_warn  # predict.py prints it for files below 44.1 kHz or at 96 kbps or less
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nMain detector for predict.py: {best.detector} (worst-case {best.worst_case:.3f})")
    return cfg


def md_table(T: pd.DataFrame) -> str:
    def f(x, pct=True):
        return "-" if pd.isna(x) else (f"{100 * x:.1f}%" if pct else f"{x:.3f}")

    lines = ["| Model | Matched: bal. acc / FPR | Unseen generator: bal. acc (mean) / worst caught | "
             "**Unseen human: accuracy on FMA pre-2010** | Unseen human: Jamendo | Vocal→Instr bal. acc | Instr→Vocal bal. acc |",
             "|---|---|---|---|---|---|---|"]
    for r in T.itertuples():
        g = getattr(r, "unseen_gen_worst_generator", "")
        lines.append(
            f"| {r.model}{f' *(partial: {r.partial})*' if r.partial else ''} | {f(getattr(r, 'matched_balanced_accuracy', np.nan))} / {f(getattr(r, 'matched_fpr', np.nan))} "
            f"| {f(getattr(r, 'unseen_gen_balanced_accuracy_mean', np.nan))} / {f(getattr(r, 'unseen_gen_recall_worst', np.nan))} ({g}) "
            f"| **{f(getattr(r, 'unseen_human_acc_fma_pre2010', np.nan))}**{'†' if r.detector in IN_DOMAIN else ''} "
            f"({f(getattr(r, 'unseen_human_acc_fma_pre2010_lo', np.nan))}-{f(getattr(r, 'unseen_human_acc_fma_pre2010_hi', np.nan))}) "
            f"| {f(getattr(r, 'unseen_human_acc_jamendo', np.nan))} "
            f"| {f(getattr(r, 'vocal_to_instr_balanced_accuracy', np.nan))} (in-domain {f(getattr(r, 'instr_to_instr_balanced_accuracy', np.nan))}) "
            f"| {f(getattr(r, 'instr_to_vocal_balanced_accuracy', np.nan))} (in-domain {f(getattr(r, 'vocal_to_vocal_balanced_accuracy', np.nan))}) |")
    if any(T.detector.isin(IN_DOMAIN)):
        lines += ["", "† The public lofcz model was trained on FMA, so FMA is not unseen for it: its FMA cells are left blank."]
    lines += ["", "Unseen generator: mean over leave-one-version-out rounds; 'worst caught' is the lowest detection rate in any of them. "
              "Whole-family hold-outs are in results/unseen_generator_table.csv. For the off-the-shelf detectors, Suno v3.5 and "
              "Udio v1 are left out of that mean because those detectors were trained on them."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures

def _dot_panel(ax, labels, vals, lo, hi, title, xlabel, ref=None):
    y = np.arange(len(labels))[::-1]
    ax.hlines(y, lo, hi, color=S.SERIES[0], lw=2, alpha=0.35)
    ax.plot(vals, y, "o", ms=8, color=S.SERIES[0], mec=S.SURFACE, mew=2)
    for yi, v in zip(y, vals):
        if not np.isnan(v):
            ax.annotate(f"{100 * v:.1f}%", (v, yi), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7, color=S.INK2)
    if ref is not None:
        ax.axvline(ref, color=S.AXIS, lw=1)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.02)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def fig_four_conditions(T: pd.DataFrame) -> None:
    labels = [m + (" (partial)" if p else "") for m, p in zip(T.model, T.partial)]
    get = lambda c: T[c].to_numpy(float) if c in T else np.full(len(T), np.nan)  # noqa: E731
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.55 * len(T) + 1.6), sharey=True)
    _dot_panel(axes[0], labels, get("matched_balanced_accuracy"), get("matched_balanced_accuracy"), get("matched_balanced_accuracy"),
               "Matched (lab setting)", "balanced accuracy", 0.5)
    _dot_panel(axes[1], labels, get("unseen_gen_balanced_accuracy_mean"), get("unseen_gen_recall_worst"), get("unseen_gen_balanced_accuracy_mean"),
               "Unseen generator", "balanced accuracy (mean); line to worst detection rate", 0.5)
    _dot_panel(axes[2], labels, get("unseen_human_acc_fma_pre2010"), get("unseen_human_acc_fma_pre2010_lo"), get("unseen_human_acc_fma_pre2010_hi"),
               "Unseen human music (headline)", "accuracy on FMA uploads before 2010 (95% CI)")
    _dot_panel(axes[3], labels, get("vocal_to_instr_balanced_accuracy"), get("vocal_to_instr_balanced_accuracy"), get("vocal_to_instr_balanced_accuracy"),
               "Vocal → instrumental", "balanced accuracy", 0.5)
    fig.suptitle("Every detector under the four evaluation conditions", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG / "four_conditions.png")
    plt.close(fig)


def fig_heatmap(M: pd.DataFrame, path: Path, title: str, cbar: str, fmt="{:.0%}") -> None:
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * M.shape[1], 0.9 + 0.42 * M.shape[0]))
    im = ax.imshow(M.to_numpy(float), cmap=S.SEQ, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if not pd.isna(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7, color=S.text_on(v))
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=40, ha="right")
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cbar)
    cb.outline.set_visible(False)
    ax.set_title(title, loc="left")
    fig.savefig(path)
    plt.close(fig)


def fig_confusions(P: pd.DataFrame) -> None:
    for cond, g in P.groupby("condition"):
        dets = ordered(g.detector)
        fig, axes = plt.subplots(1, len(dets), figsize=(2.6 * len(dets), 2.9))
        axes = np.atleast_1d(axes)
        for ax, d in zip(axes, dets):
            h = g[g.detector == d]
            pred = (h.p_ai >= 0.5).astype(int)
            cm = np.array([[((h.label == t) & (pred == p)).sum() for p in (0, 1)] for t in (0, 1)], dtype=float)
            rate = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(rate, cmap=S.SEQ, vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{rate[i, j]:.0%}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8, color=S.text_on(rate[i, j]))
            ax.set_xticks([0, 1], ["called human", "called AI"])
            ax.set_yticks([0, 1], ["human", "AI"])
            ax.grid(False)
            ax.set_title(panel_title(d), fontsize=8, loc="left")
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.suptitle(f"Confusion matrices - {cond.replace('_', ' ')} (all rounds pooled; rows sum to 100%)", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(FIG / f"confusion_{cond}.png")
        plt.close(fig)


def fig_dumbbell(D: pd.DataFrame, path: Path, title: str, a: str, b: str, xlabel: str) -> None:
    """Rows = categories; two dots per row (a = slot 1, b = slot 2) joined by a line."""
    dets = ordered(D.detector)
    fig, axes = plt.subplots(1, len(dets), figsize=(3.0 * len(dets), 0.38 * D["item"].nunique() + 1.6), sharey=True)
    axes = np.atleast_1d(axes)
    items = sorted(D["item"].unique())
    y = np.arange(len(items))[::-1]
    for ax, d in zip(axes, dets):
        h = D[D.detector == d].set_index("item").reindex(items)
        ax.hlines(y, h[a], h[b], color=S.AXIS, lw=2)
        ax.plot(h[a], y, "o", ms=8, color=S.SERIES[0], mec=S.SURFACE, mew=2, label=a.replace("_", " "))
        ax.plot(h[b], y, "o", ms=8, color=S.SERIES[1], mec=S.SURFACE, mew=2, label=b.replace("_", " "))
        ax.set_yticks(y, items)
        ax.set_xlim(0, 1.02)
        ax.set_title(panel_title(d), fontsize=8, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(xlabel)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.suptitle(title, x=0.01, ha="left", y=1.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # the table has arrows; a redirected console is cp1252
    S.apply()
    FIG.mkdir(parents=True, exist_ok=True)
    table = load_table(REPO)
    P = load_predictions(table)
    R = by_round(P)
    R.to_csv(RESULTS / "metrics_by_round.csv", index=False)
    UH = unseen_human_breakdown(P)
    UH.to_csv(RESULTS / "unseen_human_breakdown.csv", index=False)
    expected = pd.Series(Counter(r.condition for r in make_rounds(table)))  # the real round counts, not the best-covered detector's
    T = paper_table(R, UH, expected)
    T.to_csv(RESULTS / "results_table.csv", index=False)
    (RESULTS / "results_table.md").write_text(md_table(T), encoding="utf-8")
    print(md_table(T))
    write_config(T)

    G = R[R.condition == "unseen_generator"].pivot(index="detector", columns="round", values="recall")
    if len(G):
        versions = sorted(c for c in G.columns if not c.startswith("family"))
        families = sorted(c for c in G.columns if c.startswith("family"))
        G = G[versions + families].rename(columns={c: "ALL " + c.split("_", 1)[1] for c in families})
        G = G.reindex(ordered(G.index)).rename(index=label)
        G.to_csv(RESULTS / "unseen_generator_table.csv")
        for d, gens in IN_DOMAIN_GEN.items():  # a published detector's own training generators are not held out for it
            if label(d) in G.index:
                fams = {"ALL " + g.split("_", 1)[0] for g in gens}
                G.loc[label(d), [c for c in G.columns if c in gens or c in fams]] = np.nan  # left blank, like unseen_human.png
        fig_heatmap(G, FIG / "unseen_generator.png", "Share of each held-out generator's tracks caught (it was never seen in training)",
                    "detection rate (recall)")
    UHf = UH.copy()
    UHf.loc[UHf.in_domain_for_detector.astype(bool), "accuracy"] = np.nan  # not "unseen" for that detector: leave blank
    H = UHf.pivot_table(index="detector", columns=["test_set"], values="accuracy", aggfunc="min", dropna=False)
    if len(H):
        H = H.reindex(ordered(H.index)).rename(index=label)
        fig_heatmap(H, FIG / "unseen_human.png", "Accuracy on music from sources never seen in training (human sets: share correctly called human)",
                    "accuracy")
    V = R[R.condition == "vocal_instr"].copy()
    if len(V):
        V[["train", "test"]] = V["round"].str.extract(r"train=(\w+)_test=(\w+)")
        V.to_csv(RESULTS / "vocal_instr_table.csv", index=False)
        D = []
        for d, g in V.groupby("detector"):
            for tr in ("vocal", "instrumental"):
                other = "instrumental" if tr == "vocal" else "vocal"
                same = g[(g.train == other) & (g.test == other)].balanced_accuracy
                cross = g[(g.train == tr) & (g.test == other)].balanced_accuracy
                if len(same) and len(cross):
                    D.append({"detector": d, "item": f"test on {other}", "trained_on_same_kind": float(same.iloc[0]),
                              "trained_on_the_other_kind": float(cross.iloc[0])})
        if D:
            fig_dumbbell(pd.DataFrame(D), FIG / "vocal_instr.png", "Vocal and instrumental: is it one task or two?",
                         "trained_on_same_kind", "trained_on_the_other_kind", "balanced accuracy")
    Gn = R[R.condition == "genre"].copy()
    if len(Gn):
        held = Gn["round"].str.endswith("_held_out")
        Gn["setting"] = np.where(held, "held_out", "seen")
        Gn["genre"] = np.where(held, Gn["round"].str[:-len("_held_out")], Gn["round"].str[:-len("_seen")])
        Gn.to_csv(RESULTS / "genre_table.csv", index=False)
        D = Gn.pivot_table(index=["detector", "genre"], columns="setting", values="auc").reset_index().rename(columns={"genre": "item"})
        if {"seen", "held_out"} <= set(D.columns):
            D = D.rename(columns={"seen": "genre_seen_in_training", "held_out": "genre_held_out"})
            D = D[~D.detector.isin(PRETRAINED)]  # off-the-shelf detectors are never retrained: seen == held out by construction
            fig_dumbbell(D, FIG / "genre.png", "Your genre hypothesis: does a detector fail on a genre it never saw? (AUC)",
                         "genre_seen_in_training", "genre_held_out", "AUC")
    fig_four_conditions(T)
    fig_confusions(P)
    print(f"\nTables in {RESULTS}, figures in {FIG}")


if __name__ == "__main__":
    main()
