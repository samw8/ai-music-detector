"""Build results/report.html: a self-contained findings page (figures embedded) from the current results.

    python scripts/make_report.py
"""
from __future__ import annotations

import base64
import html
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aimd import MODELS as MODEL_DIR  # noqa: E402
from aimd import RESULTS  # noqa: E402
from aimd.cnn import FMAX  # noqa: E402
from aimd.models import PRETRAINED, label, ordered  # noqa: E402
from aimd.splits import MIN_GEN_CLIPS, base_split, load_table  # noqa: E402
from evaluate import vocal_generator_confound  # noqa: E402  (same numbers as the model card and the caveat)

FIG = RESULTS / "figures"
HIST_LICENCE = "Sound Recording Common Law Protection"  # how FMA tags its digitised early commercial 78-rpm records
WEBSITE = "https://who-gets-called-ai.streamlit.app"


def slug_round(s: str) -> str:
    """A round's name as run_experiments.py writes it in file names and metrics_by_round.csv."""
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)


def pct(x, d=1) -> str:
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def img(name: str, alt: str) -> str:
    p = FIG / name
    if not p.exists():
        return ""
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><div class="scroll"><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}"></div><figcaption>{html.escape(alt)}</figcaption></figure>'


def table(df: pd.DataFrame, cls: str = "") -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


CSS = """
:root {
  /* Layout: one reading column (~68ch) with wide figure breakouts; summary first, evidence after. */
  --bg: #f6f7f9; --surface: #ffffff; --ink: #121820; --ink2: #49525e; --muted: #7a838f; --rule: #dde1e7;
  --accent: #2a78d6; --accent-soft: #e3eefb; --warn: #c2410c; --warn-soft: #fdeee6;
  --display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --data: "JetBrains Mono", "Cascadia Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #12161c; --surface: #1a2028; --ink: #eef1f5; --ink2: #b9c1cc; --muted: #8b95a2; --rule: #2c343e;
  --accent: #5ea1f0; --accent-soft: #1b2e46; --warn: #f0895a; --warn-soft: #3a2218; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #12161c; --surface: #1a2028; --ink: #eef1f5; --ink2: #b9c1cc; --muted: #8b95a2; --rule: #2c343e;
  --accent: #5ea1f0; --accent-soft: #1b2e46; --warn: #f0895a; --warn-soft: #3a2218; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font: 17px/1.6 var(--body);
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
main { max-width: 1060px; margin: 0 auto; padding-block: 40px 80px; display: grid; gap: 44px; }
.col { max-width: 68ch; }
header { display: grid; gap: 12px; }
.kicker { font: 600 13px/1.2 var(--data); letter-spacing: .06em; text-transform: uppercase; color: var(--accent); }
h1 { font: 700 clamp(34px, 6vw, 56px)/1.05 var(--display); margin: 0; text-wrap: balance; letter-spacing: -.01em; }
h2 { font: 700 26px/1.2 var(--display); margin: 0 0 10px; text-wrap: balance; }
h3 { font: 650 18px/1.3 var(--display); margin: 18px 0 6px; }
p { margin: 0 0 12px; } .lede { font-size: 20px; color: var(--ink2); }
section { display: grid; gap: 14px; min-width: 0; }
.verdict { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
.stat { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; display: grid; gap: 4px; min-width: 0; }
.stat b { font: 700 34px/1.1 var(--display); }
.stat.bad b { color: var(--warn); } .stat.good b { color: var(--accent); }
.stat span { font-size: 15px; color: var(--ink2); }
.note { background: var(--accent-soft); border-radius: 8px; padding: 12px 16px; font-size: 15.5px; }
.caution { background: var(--warn-soft); border-radius: 8px; padding: 12px 16px; font-size: 15.5px; }
.scroll { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; font: 14px/1.4 var(--data); font-variant-numeric: tabular-nums; min-width: 560px; background: var(--surface); }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--ink2); background: var(--bg); position: sticky; top: 0; }
figure { margin: 0; display: grid; gap: 6px; }
figure img { display: block; max-width: none; width: 100%; min-width: 640px; border-radius: 6px; background: #fcfcfb; }
figcaption { font-size: 14px; color: var(--muted); }
ul { margin: 0 0 12px; padding-left: 22px; } li { margin: 4px 0; }
a { color: var(--accent); } code { font: 14px var(--data); background: var(--surface); padding: 1px 4px; border-radius: 4px; }
footer { color: var(--muted); font-size: 14px; border-top: 1px solid var(--rule); padding-top: 16px; }
@media (max-width: 520px) { body { font-size: 16px; } .lede { font-size: 18px; } }
"""


MARGINAL_DB = 0.15  # a comb less than this far above the human/codec ceiling is reported as "marginal"


def main() -> None:
    T = pd.read_csv(RESULTS / "results_table.csv").set_index("detector")
    UH = pd.read_csv(RESULTS / "unseen_human_breakdown.csv")
    cfg = json.loads((MODEL_DIR / "config.json").read_text()) if (MODEL_DIR / "config.json").exists() else {}
    df = load_table(ROOT)
    T["partial"] = T.partial.fillna("") if "partial" in T else ""
    # headline numbers only from detectors with every round finished
    trained = [d for d in ordered(T.index) if d not in PRETRAINED and not T.loc[d, "partial"]]
    if not trained:
        sys.exit("[report] no trained detector has finished every round yet ("
                 + "; ".join(f"{d}: {T.loc[d, 'partial']}" for d in ordered(T.index) if d not in PRETRAINED)
                 + "). Finish `run train` / `run train-mp3`, then `run evaluate`; report.html not written.")
    n_det = len({d.split("@")[0] for d in ordered(T.index) if d not in PRETRAINED})
    best_h = T.loc[trained, "unseen_human_acc_fma_pre2010"].idxmax()
    best_j = T.loc[trained, "unseen_human_acc_jamendo"].idxmax()
    best_m = T.loc[trained, "matched_balanced_accuracy"].idxmax()
    worst_h = T.loc[trained, "unseen_human_acc_fma_pre2010"].idxmin()
    primary = cfg.get("primary", best_m)
    R = pd.read_csv(RESULTS / "metrics_by_round.csv")
    man = pd.read_csv(ROOT / "data" / "manifest.csv")

    rows = []
    for d in ordered(T.index):
        r = T.loc[d]
        name = html.escape(label(d)) + (f"<br><span class='note'>partial: {html.escape(r.partial)}</span>" if r.partial else "")
        rows.append([name, pct(r.get("matched_balanced_accuracy")), pct(r.get("matched_fpr")),
                     pct(r.get("unseen_gen_balanced_accuracy_mean")), pct(r.get("unseen_gen_recall_worst"), 0) + f" ({html.escape(str(r.get('unseen_gen_worst_generator', '')))})",
                     f"<b>{pct(r.get('unseen_human_acc_fma_pre2010'))}</b>{'†' if d == 'lofcz_fakeprint' else ''}",
                     pct(r.get("unseen_human_acc_jamendo")), pct(r.get("vocal_to_instr_balanced_accuracy")), pct(r.get("instr_to_vocal_balanced_accuracy"))])
    four = pd.DataFrame(rows, columns=["Detector", "Matched bal. acc", "Matched FPR", "Unseen gen. bal. acc (mean)", "Worst gen. caught",
                                       "Unseen human: FMA pre-2010", "Unseen human: Jamendo", "Vocal→Instr", "Instr→Vocal"])

    fp = pd.read_csv(RESULTS / "fingerprints.csv") if (RESULTS / "fingerprints.csv").exists() else None
    fp_rows, fp_text = "", ""
    if fp is not None:
        ref = fp.group.str.startswith(("human", "control"))
        null = float(fp.ceiling_db.iloc[0])
        sel = fp[~ref].sort_values("comb_contrast_db", ascending=False)
        ctl_db = sel.format_control_db_at_spacing.map(lambda x: "-" if pd.isna(x) else f"{x:.2f}")
        marginal = sel.comb_found & (sel.comb_contrast_db < null + MARGINAL_DB)
        fp_tab = pd.DataFrame({"Generator": sel.group,
                               "Delivered as": sel.codec.str.upper().str.replace("PCM_.*", "WAV", regex=True) + " " + (sel.orig_sr / 1000).map("{:g} kHz".format),
                               "Strongest grid (Hz)": sel.comb_spacing_hz.round(2), "Contrast (dB)": sel.comb_contrast_db.round(2),
                               "Same grid in both halves?": np.where(sel.replicates, "yes", "no"),
                               "Human audio, same format, same grid (dB)": ctl_db,
                               "Comb found?": np.where(marginal, "marginal", np.where(sel.comb_found, "yes", "no"))})
        n_ref = int(fp[ref].n_clips.max())
        fp_rows = table(fp_tab) + (f"<p class='note'>Ceiling: averaged over {n_ref} clips per set, no human set (FMA, Jamendo, GuitarSet) and no "
                                   f"set of the same human clips re-encoded in these formats reaches more than {null:.2f} dB at any spacing from "
                                   f"12 to 400 Hz. This ceiling applies only to averages of many clips: single human recordings usually score "
                                   f"well above it, so it must never be used to judge one song. 'Comb found' = the generator's average is above "
                                   f"that ceiling and has the same spacing (within 1%) in two independent halves of its clips; 'marginal' = less "
                                   f"than {MARGINAL_DB} dB above the ceiling, too close to call. 'Delivered as' is each generator's most common "
                                   "file format.</p>")
        found = sel[sel.comb_found & ~marginal]
        weak = ", ".join(sorted(sel[marginal].group))
        grids = found.groupby(found.comb_spacing_hz.round(1)).group.apply(lambda g: ", ".join(sorted(g)))
        lines = [f"<li><b>{sp:g} Hz grid:</b> {html.escape(gs)}</li>" for sp, gs in grids.sort_index().items()]
        fsp = found.set_index("group").comb_spacing_hz
        open_ok = [g for g in ("musicgen", "magnet", "jasco") if g in fsp.index and abs(fsp[g] - 50 * round(fsp[g] / 50)) < 0.5]
        missing = ", ".join(sorted(sel[~sel.comb_found].group))
        sao = fp.set_index("group").reindex(["stable_audio_open"])
        sao_txt = ""
        if sao.comb_contrast_db.notna().all():
            s = sao.iloc[0]
            hit = s["contrast@21.53Hz"] > null
            sao_txt = (" Stable Audio Open, whose Oobleck decoder predicts a 21.5 Hz grid, "
                       + ("shows it." if hit else f"does <em>not</em> (contrast {s['contrast@21.53Hz']:.2f} dB at 21.5 Hz); its strongest grid is "
                          f"{s.comb_spacing_hz:.1f} Hz, which no published decoder explains. So a missing comb does not prove a model has no "
                          "transposed-convolution decoder, and an unexplained one is not attributable."))
        fp_text = (f"<p class='col'>Check on open models first: {len(open_ok)} of the 3 EnCodec-32k models (MusicGen, MAGNeT, JASCO), whose published "
                   "decoder predicts grids at multiples of 50 Hz, show a comb on a multiple of 50 Hz"
                   + (f" ({', '.join(open_ok)})." if open_ok else ".") + sao_txt + "</p>"
                   f"<ul class='col'>{''.join(lines)}</ul>"
                   + (f"<p class='col'>Marginal, too close to the ceiling to call: {html.escape(weak)}.</p>" if weak else ""))
        # spacings where the scan's own noise tends to peak (best spacings of the human and codec-control sets and their halves)
        noise_sp = np.concatenate([fp[ref][c].to_numpy() for c in ("comb_spacing_hz", "half_a_spacing_hz", "half_b_spacing_hz")])
        on_noise = [g for g, s in fsp.items() if np.any(np.abs(noise_sp - s) <= 0.01 * s)]
        if on_noise:
            spots = []
            for x in np.sort(noise_sp):  # merge spacings within 1% of each other for display
                if not spots or x > spots[-1] * 1.01:
                    spots.append(x)
            fp_text += (f"<p class='col'>Caution: {html.escape(', '.join(on_noise))} peak{'s' if len(on_noise) == 1 else ''} at a spacing where the scan's noise also "
                        f"peaks in human and codec-control sets (around {', '.join(f'{x:.0f}' for x in spots)} Hz). The comb is strong, "
                        "but its exact spacing is less trustworthy than its strength.</p>")
        # does "no comb" mean a Fourier detector cannot catch them? check the retrained fakeprint's own results
        no_comb = sorted(sel[~sel.comb_found].group)
        fpm = RESULTS / "predictions" / "fakeprint_lr" / "matched__all.csv"
        fourier = ""
        if fpm.exists() and no_comb:
            Pm = pd.read_csv(fpm).merge(df[["clip_id", "model"]], on="clip_id")
            seen_rec = Pm[Pm.label == 1].groupby("model").p_ai.apply(lambda s: (s >= 0.5).mean()).reindex(no_comb).dropna()
            Ru = R[(R.detector == "fakeprint_lr") & (R.condition == "unseen_generator")].set_index("round").recall
            held = Ru.reindex([slug_round(g) for g in no_comb]).dropna()
            strong = Ru.reindex([slug_round(g) for g in found.group]).dropna()
            if len(seen_rec) and len(held):
                fourier = (" A missing grid does not make a Fourier-based detector blind: the retrained Deezer fakeprint reads the whole "
                           f"5-16 kHz spectrum, not only grids. With these generators in training it caught {pct(seen_rec.min(), 0)}-"
                           f"{pct(seen_rec.max(), 0)} of their test clips; with one of them held out, {pct(held.min(), 0)}-{pct(held.max(), 0)}"
                           + (f"; and it caught only {pct(strong.min(), 0)} of held-out {html.escape(strong.idxmin())}, which has a clear comb"
                              if len(strong) else "")
                           + (f". So the scan is a rough guide, not a rule: held out, the fakeprint caught a median {pct(strong.median(), 0)} of the "
                              f"{len(strong)} generators with a clear comb but {pct(held.median(), 0)} of the {len(held)} without one, with exceptions both ways."
                              if len(strong) else "."))
        fp_text += f"<p class='col'>No comb that clears both tests: {html.escape(missing)}.{fourier}</p>"
        suno = [g for g in found.group if g.startswith("suno")]
        if suno and all(abs(fsp[g] - 200) < 1 for g in suno):
            yue = "yue" in fsp.index and abs(fsp["yue"] - 200) < 1
            ctl = found.set_index("group").format_control_db_at_spacing.reindex(suno)
            fmt_ok = bool(ctl.notna().all() and (ctl < null).all())
            fp_text += (f"<p class='col'>{', '.join(suno)} share one grid at 200 Hz{', as does the open-source YuE (its X-Codec tokenizer runs at 50 frames per second)' if yue else ''}. "
                        + ("Human music re-encoded as the same 48 kHz MP3 stays below the ceiling at 200 Hz (see the table), so the grid comes from the generator, "
                           "not the file format. That is consistent with a 50 Hz codec-token design across Suno's versions; it is evidence about the decoder, "
                           "not proof of Suno's architecture.</p>" if fmt_ok else
                           "But human music re-encoded as the same MP3 also shows a grid there, so the file format cannot be ruled out as its source.</p>"))

    gen_tab = ""
    gen_txt = ""
    if (RESULTS / "genre_table.csv").exists():
        G0 = pd.read_csv(RESULTS / "genre_table.csv")
        G = G0[G0.detector.isin(trained)].pivot_table(index=["detector", "genre"], columns="setting", values=["auc", "fpr"])
        G.columns = [f"{v}_{s}" for v, s in G.columns]
        G = G.reset_index()
        if {"auc_seen", "auc_held_out", "fpr_seen", "fpr_held_out"} <= set(G.columns):
            G["drop"] = G.auc_seen - G.auc_held_out
            G["fpr_rise"] = G.fpr_held_out - G.fpr_seen
            agg = G.groupby("detector").agg(mean_drop=("drop", "mean"), max_drop=("drop", "max"),
                                            mean_rise=("fpr_rise", "mean"), max_rise=("fpr_rise", "max")).reset_index()
            w, wf = G.loc[G["drop"].idxmax()], G.loc[G["fpr_rise"].idxmax()]
            n_lo, n_hi = int(G0.n.min()), int(G0.n.max())
            n_pos, n_pairs = int((G["drop"] > 0).sum()), int(G["drop"].notna().sum())
            n_up, n_dn, n_eq = int((G.fpr_rise > 0).sum()), int((G.fpr_rise < 0).sum()), int((G.fpr_rise == 0).sum())
            kk = G.groupby("detector").fpr_rise.agg(up=lambda s: int((s > 0).sum()), n="count")
            steady = [f"{html.escape(label(d))} ({int(r.up)} of {int(r.n)} genres)" for d, r in kk.iterrows() if r.up >= r.n - 1]
            n_gen = int(G0.genre.nunique())
            all_gen = int(pd.read_csv(ROOT / "data" / "tags.csv", usecols=["genre"]).genre.dropna().nunique())
            verdict = ("Drops this small do not support \"detectors only work per genre\", at least for genres this coarse"
                       if agg.mean_drop.max() < 0.05 else
                       "Drops this large support the hypothesis that detectors depend on the genres they were trained on")
            gen_txt = (f"<p class='col'><b>Result:</b> removing a genre from training lowers each detector's AUC on that genre by "
                       f"{agg.mean_drop.min():+.3f} to {agg.mean_drop.max():+.3f} on average (AUC runs from 0.5, guessing, to 1, perfect). "
                       f"The largest single drop is {w['drop']:+.3f} ({html.escape(label(w.detector))}, {html.escape(str(w.genre))}). "
                       f"{verdict}. The drops are small but consistent: {n_pos} of {n_pairs} detector-genre pairs drop, and both settings "
                       f"score the same {n_lo}-{n_hi} test clips per genre. "
                       + (f"At the 0.5 threshold the picture is mixed: holding a genre out raises the share of human clips accused in that genre in "
                          f"{n_up} of {n_pairs} pairs, lowers it in {n_dn} and leaves it unchanged in {n_eq} (mean {100 * G.fpr_rise.mean():+.1f} points per pair). "
                          f"The largest rise is {100 * wf.fpr_rise:+.1f} points ({html.escape(label(wf.detector))}, {html.escape(str(wf.genre))}: {pct(wf.fpr_seen)} of human "
                          f"clips flagged with the genre in training, {pct(wf.fpr_held_out)} with it held out). "
                          + (f"The rise is consistent only for {' and '.join(steady)}, so some detectors, not all, become more suspicious of a genre they never saw. "
                             if steady else "No detector's rise is consistent across genres. "))
                       + f"Only {n_gen} of the {all_gen} tagged genres have at least "
                       f"30 AI and 30 human test clips and are scored. ResNet18 was not run on this condition ({2 * n_gen} extra trainings were too "
                       "slow on CPU).</p>")
            agg["detector"] = agg.detector.map(label)
            gen_tab = table(agg.assign(mean_drop=agg.mean_drop.map(lambda x: f"{x:+.3f}"), max_drop=agg.max_drop.map(lambda x: f"{x:+.3f}"),
                                       mean_rise=agg.mean_rise.map(lambda x: f"{100 * x:+.1f} pts"), max_rise=agg.max_rise.map(lambda x: f"{100 * x:+.1f} pts"))
                            .rename(columns={"detector": "Detector", "mean_drop": "Mean AUC drop when genre unseen", "max_drop": "Worst genre's AUC drop",
                                             "mean_rise": "Mean rise in humans accused", "max_rise": "Worst genre's rise"}))

    rob_tab, band_note = "", ""
    if (RESULTS / "robustness.csv").exists():
        Rb = pd.read_csv(RESULTS / "robustness.csv")
        p = cfg.get("primary")
        if p in set(Rb.detector):
            r = Rb[Rb.detector == p][["transform", "balanced_accuracy", "fnr", "fpr"]]
            rob_tab = table(r.assign(balanced_accuracy=r.balanced_accuracy.map(pct), fnr=r.fnr.map(pct), fpr=r.fpr.map(pct))
                            .rename(columns={"transform": "Edit", "balanced_accuracy": "Bal. acc", "fnr": "AI missed", "fpr": "Humans accused"}))
            bw = r[r["transform"].str.startswith("bandwidth")]
            clean_fpr = r.loc[r["transform"] == "clean", "fpr"]
            lowsr = df[df.orig_sr.between(1, 32000)]
            ff = ROOT / "data" / "features.csv"
            feats8 = pd.read_csv(ff, usecols=["clip_id", "band_8k_12k_db"]) if ff.exists() else None

            def content_split(pred_csv):
                """The headline set (FMA pre-2010) split by content: tracks with almost no sound above 8 kHz vs the rest."""
                if feats8 is None or not pred_csv.exists():
                    return None
                Pf = (pd.read_csv(pred_csv).merge(man[["clip_id", "orig_sr", "source", "license"]], on="clip_id")
                      .merge(feats8, on="clip_id", how="left"))
                Pf = Pf[(Pf.label == 0) & (Pf.source == "fma_pre2010")]
                lim = Pf.band_8k_12k_db.isna() | (Pf.band_8k_12k_db < -45)  # NaN = real bandwidth ends below ~10 kHz
                Lm, Rest = Pf[lim], Pf[~lim]
                ot = Lm[Lm.license.astype(str).str.contains(HIST_LICENCE, regex=False)]  # FMA's tag for historic 78-rpm records
                return {"n": len(Pf), "n_lim": len(Lm), "n_lim44": int((Lm.orig_sr >= 44100).sum()), "share": lim.mean(),
                        "fpr_lim": (Lm.p_ai >= 0.5).mean(), "fpr_rest": (Rest.p_ai >= 0.5).mean(), "n_rest": len(Rest),
                        "fp_lim": int((Lm.p_ai >= 0.5).sum()), "fp": int((Pf.p_ai >= 0.5).sum()),
                        "fp_ot": int((ot.p_ai >= 0.5).sum()), "n_ot": len(ot)}

            uh = "unseen_human__train_human=jamendo_test=fma.csv"
            cs = content_split(RESULTS / "predictions" / p / uh)
            shortcut = len(bw) and len(clean_fpr) and bw.fpr.max() > clean_fpr.iloc[0] + 0.10
            if shortcut:
                hl = "" if cs is None else (
                    f" It matters for the headline. Of the {cs['n']} pre-2010 FMA tracks, {cs['n_lim']} ({pct(cs['share'])}) have almost no sound "
                    f"above 8 kHz, {cs['n_lim44']} of them stored as ordinary 44.1 kHz or higher files. The main detector calls "
                    f"{pct(cs['fpr_lim'])} of them AI against {pct(cs['fpr_rest'])} of the other {cs['n_rest']}, so they account for "
                    f"{cs['fp_lim']} of its {cs['fp']} false accusations there ({cs['fp_ot']} of them from the {cs['n_ot']} tagged as historic 78-rpm records).")
                band_note = (f"<p class='caution col'><b>A shortcut to know about.</b> In this data, {int((lowsr.label == 1).sum()):,} of the "
                             f"{len(lowsr):,} clips whose original file was 32 kHz or lower are AI. The main detector learned that missing high "
                             "frequencies mean AI: when the same human test clips are band-limited, it wrongly calls "
                             + " and ".join(f"{pct(x.fpr)} of them AI ({html.escape(t)})" for t, x in zip(bw["transform"], bw.itertuples()))
                             + f", against {pct(clean_fpr.iloc[0])} unedited.{hl} Do not trust it on old, phone-quality or low-sample-rate "
                               "recordings. The song checker warns when a file is stored below 44.1 kHz, at 96 kbps or less, or has almost no "
                               "sound above 8 kHz.</p>")
            old_row, old_rob = RESULTS / "resnet18_16k_before.csv", RESULTS / "resnet18_16k_before_robustness.csv"
            if p == "resnet18_mel" and old_row.exists() and old_rob.exists() and p in T.index:
                # the ResNet18 was retrained without the shortcut: show before and after side by side
                o, n = pd.read_csv(old_row).iloc[0], T.loc[p]
                orb = pd.read_csv(old_rob).set_index("transform")
                nrb = r.set_index("transform")
                cs_old = content_split(RESULTS / "archive" / "resnet18_mel_16k" / "predictions" / uh)
                worst = lambda rb: rb.drop(index=[i for i in rb.index if i == "clean" or i.startswith("bandwidth")]).balanced_accuracy.min()  # noqa: E731
                rows_ba = [("Matched: balanced accuracy", o.matched_balanced_accuracy, n.matched_balanced_accuracy),
                           ("Matched: humans wrongly called AI", o.matched_fpr, n.matched_fpr),
                           ("Unseen generator, one version held out: balanced accuracy", o.unseen_gen_balanced_accuracy_mean, n.unseen_gen_balanced_accuracy_mean),
                           ("Unseen generator, whole family held out: balanced accuracy", o.get("unseen_family_balanced_accuracy_mean"), n.get("unseen_family_balanced_accuracy_mean")),
                           ("Unseen human, FMA before 2010: wrongly called AI", 1 - o.unseen_human_acc_fma_pre2010, 1 - n.unseen_human_acc_fma_pre2010),
                           ("Unseen human, Jamendo: wrongly called AI", 1 - o.unseen_human_acc_jamendo, 1 - n.unseen_human_acc_jamendo),
                           ("Vocal → instrumental: balanced accuracy", o.vocal_to_instr_balanced_accuracy, n.vocal_to_instr_balanced_accuracy),
                           ("Instrumental → vocal: balanced accuracy", o.instr_to_vocal_balanced_accuracy, n.instr_to_vocal_balanced_accuracy)]
                for t in [i for i in orb.index if i.startswith("bandwidth")]:
                    rows_ba.append((f"Humans wrongly called AI after {t}", orb.loc[t, "fpr"], nrb.loc[t, "fpr"] if t in nrb.index else np.nan))
                rows_ba.append(("Humans wrongly called AI, same clips unedited", orb.loc["clean", "fpr"], nrb.loc["clean", "fpr"] if "clean" in nrb.index else np.nan))
                rows_ba.append(("Worst everyday edit: balanced accuracy", worst(orb), worst(nrb)))
                if cs_old and cs:
                    rows_ba.append((f"FMA before 2010 with almost no sound above 8 kHz ({cs['n_lim']} tracks): wrongly called AI",
                                    cs_old["fpr_lim"], cs["fpr_lim"]))
                    rows_ba.append((f"FMA before 2010, the other {cs['n_rest']} tracks: wrongly called AI", cs_old["fpr_rest"], cs["fpr_rest"]))
                ba = pd.DataFrame([(k, pct(a), pct(b)) for k, a, b in rows_ba],
                                  columns=["", "Before: log-mel up to 16 kHz", f"After: log-mel up to {FMAX / 1000:g} kHz"])
                still = "still has" if shortcut else "no longer has"
                band_note += (f"<p class='note col'><b>The main detector was retrained to remove a shortcut.</b> In this data, "
                              f"{int((lowsr.label == 1).sum()):,} of the {len(lowsr):,} clips whose original file was 32 kHz or lower are AI. "
                              "The first ResNet18 saw frequencies up to 16 kHz and learned that missing high frequencies mean AI. It was "
                              f"retrained with its spectrogram capped at {FMAX / 1000:g} kHz, below the lowest bandwidth of any file in the data, "
                              f"and {still} that shortcut. Every number below comes from the same rounds and test clips."
                              + ("" if not (cs and not shortcut) else
                                 f" Genuinely old recordings are still harder than modern ones: it wrongly calls {pct(cs['fpr_lim'])} of the "
                                 f"{cs['n_lim']} pre-2010 FMA tracks with almost no sound above 8 kHz AI (old or lo-fi recordings, {cs['n_ot']} of "
                                 f"them tagged as historic 78-rpm records), against {pct(cs['fpr_rest'])} of the other {cs['n_rest']}. Band-limiting a modern "
                                 f"recording at 8 kHz, above the {FMAX / 1000:g} kHz the detector sees, no longer changes the verdict, so the "
                                 "missing top octave is not the cause; whether a narrower bandwidth (most of these tracks also have little "
                                 "sound at 5-8 kHz) or another trait such as surface noise is the cause was not tested.")
                              + "</p>" + table(ba))

    fmt_limit = ("The main detector uses one of them, missing high frequencies (see the shortcut note under the headline); removing it would "
                 "mean retraining with the spectrogram capped lower or with random band-limiting of both classes."
                 if band_note.startswith("<p class='caution") else
                 "The main detector's spectrogram is capped below the lowest bandwidth in the data so the missing top octave of 16 and 32 kHz "
                 "files cannot be used; the interpretable feature models still see the format traces, which is what the 'no file-format "
                 "features' ablation and the codec-equalised variants measure.")
    harm_md, harm_csv = RESULTS / "harmony_probe.md", RESULTS / "harmony_probe.csv"  # optional step; the CSV is not published
    harm = harm_md.read_text(encoding="utf-8") if harm_md.exists() and harm_csv.exists() else ""
    harm_line = ""
    if harm:
        H = pd.read_csv(harm_csv)
        from sklearn.metrics import roc_auc_score
        a = roc_auc_score(H.label[H.harm_nondiatonic_rate.notna()], H.harm_nondiatonic_rate.dropna())
        k = roc_auc_score(H.label[H.harm_key_instability.notna()], H.harm_key_instability.dropna())
        harm_line = (f"<p class='col'>On {len(H):,} clips (100 per dataset), the non-diatonic-chord rate separates AI from human with AUC {a:.2f} "
                     f"(0.5 = no information; below 0.5 means AI tracks use <em>fewer</em> out-of-key chords). Key changes "
                     + ("lean very slightly the brief's way" if k > 0.5 else "do not lean the brief's way")
                     + f": the share of 2.5-second windows whose key differs from the excerpt's key gives AUC {k:.2f}, from only four "
                       "windows per 10-second clip. Neither feature is a usable AI tell.</p>")

    # how often the zero-shot genre tagger agrees with FMA's own genre labels (mapped to the same coarse genres)
    fma_map = {"Rock": "rock_metal", "Pop": "pop", "Hip-Hop": "hiphop", "Electronic": "electronic", "Soul-RnB": "rnb_soul_gospel",
               "Folk": "country_folk", "Country": "country_folk", "Jazz": "jazz_blues", "Blues": "jazz_blues",
               "Classical": "classical", "International": "world_latin_reggae"}
    man = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["clip_id", "source", "genre_meta"])
    gm = man[man.source.str.startswith("fma") & man.genre_meta.isin(fma_map)].merge(
        pd.read_csv(ROOT / "data" / "tags.csv", usecols=["clip_id", "genre"]), on="clip_id")
    g_agree, g_n, g_k = (gm.genre_meta.map(fma_map) == gm.genre).mean(), len(gm), gm.genre_meta.map(fma_map).nunique()
    g_truth = gm.genre_meta.map(fma_map)
    g_chance = float((g_truth.value_counts(normalize=True) * gm.genre.value_counts(normalize=True)).sum())
    g_major = float(g_truth.value_counts(normalize=True).max())
    g_major_name = g_truth.value_counts().idxmax().replace("_", "/")
    # ... and how often the AST vocal tagger agrees with the clips whose vocal label is known (its calibration set)
    tg = pd.read_csv(ROOT / "data" / "tags.csv")
    thr = tg.vocal_source.str.extract(r"AST p>=([\d.]+)")[0].dropna()
    cal = tg[(tg.vocal_source == "metadata/generator") & tg.p_vocal.notna()]
    v_agree = (np.mean([((cal.p_vocal >= float(thr.iloc[0])) == c)[cal.vocal == c].mean() for c in (0, 1)])
               if len(thr) and cal.vocal.nunique() == 2 else np.nan)
    v_tagged = float((tg.vocal_source != "metadata/generator").mean())

    n_h, n_ai = int((df.label == 0).sum()), int((df.label == 1).sum())
    gens = sorted(df[df.label == 1].model.unique())
    srcs = df.groupby(["label", "source"]).size().reset_index(name="clips")
    srcs["kind"] = srcs.label.map({0: "human", 1: "AI"})

    bh, bm, wh, bj = T.loc[best_h], T.loc[best_m], T.loc[worst_h], T.loc[best_j]
    # a partial detector may already have a higher matched score; name it so the tile and the table agree
    higher = [label(d) for d in ordered(T.index) if d not in PRETRAINED and T.loc[d, "partial"]
              and T.loc[d, "matched_balanced_accuracy"] > bm.matched_balanced_accuracy]
    hp_note = f" ({html.escape(', '.join(higher))} scores higher but still has conditions running)" if higher else ""
    # an off-the-shelf model may pass more unseen human tracks; name it so the tile and the table agree
    ph = [d for d in ordered(T.index) if d in PRETRAINED and pd.notna(T.loc[d, "unseen_human_acc_fma_pre2010"])
          and T.loc[d, "unseen_human_acc_fma_pre2010"] > bh.unseen_human_acc_fma_pre2010]
    ph_note = "".join(f" (the off-the-shelf {html.escape(label(d).replace('Off the shelf: ', '').replace(' (not retrained)', ''))}, not retrained, "
                      f"wrongly flags only {pct(1 - T.loc[d, 'unseen_human_acc_fma_pre2010'])} of those FMA tracks and "
                      f"{pct(1 - T.loc[d, 'unseen_human_acc_jamendo'])} of the Jamendo tracks, "
                      f"but it catches only {pct(T.loc[d, 'matched_recall'], 0)} of matched AI tracks: matched balanced accuracy "
                      f"{pct(T.loc[d, 'matched_balanced_accuracy'])}, unseen generators {pct(T.loc[d, 'unseen_gen_balanced_accuracy_mean'])}, "
                      f"vocal→instrumental {pct(T.loc[d, 'vocal_to_instr_balanced_accuracy'])}, instrumental→vocal "
                      f"{pct(T.loc[d, 'instr_to_vocal_balanced_accuracy'])})" for d in ph)

    # every headline number carries the other conditions beside it (the project's rule: never one condition alone)
    def human_txt(r) -> str:
        return f"unseen FMA tracks falsely called AI {pct(1 - r.unseen_human_acc_fma_pre2010)}, unseen Jamendo {pct(1 - r.unseen_human_acc_jamendo)}"

    def matched_txt(r) -> str:
        return f"matched split balanced accuracy {pct(r.matched_balanced_accuracy)} ({pct(r.matched_fpr)} of human tracks falsely called AI)"

    def unseen_gen_txt(r) -> str:
        return (f"unseen generators: balanced accuracy {pct(r.unseen_gen_balanced_accuracy_mean)} with one version held out (caught "
                f"{pct(r.unseen_gen_recall_mean, 0)} on average, worst {pct(r.unseen_gen_recall_worst, 0)}, "
                f"{html.escape(str(r.unseen_gen_worst_generator))}), {pct(r.unseen_family_balanced_accuracy_mean)} with a whole family "
                f"held out ({pct(r.unseen_family_recall_worst, 0)} of {html.escape(str(r.unseen_family_worst))} caught)")

    def vi_txt(r) -> str:
        return (f"balanced accuracy vocal→instrumental {pct(r.vocal_to_instr_balanced_accuracy)}, instrumental→vocal "
                f"{pct(r.instr_to_vocal_balanced_accuracy)}")

    mb = "MuseBench reported: there a fully trained ResNet18 scored 100% in-domain and passed only 28.4% of old FMA tracks"
    # which detector types are statistically tied in the lab (95% artist-bootstrap intervals overlapping the best one's)?
    Rm = R[(R.condition == "matched") & R.detector.isin([d for d in trained if "@" not in d])].set_index("detector")
    top = Rm.balanced_accuracy.idxmax() if len(Rm) else None
    tied = [] if top is None else [d for d in Rm.sort_values("balanced_accuracy", ascending=False).index
                                   if Rm.loc[d, "balanced_accuracy_lo"] <= Rm.loc[top, "balanced_accuracy_hi"]
                                   and Rm.loc[d, "balanced_accuracy_hi"] >= Rm.loc[top, "balanced_accuracy_lo"]]
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    if len(tied) > 1 and T.loc[tied, "unseen_human_acc_fma_pre2010"].notna().all():
        fma_t, ba_t = T.loc[tied, "unseen_human_acc_fma_pre2010"], Rm.loc[tied, "balanced_accuracy"]
        most = fma_t.idxmax()
        pattern = (f"The top {words.get(len(tied), len(tied))} detectors ({html.escape('; '.join(label(d) for d in tied))}) are "
                   f"statistically tied in the lab: {100 * ba_t.min():.0f}–{100 * ba_t.max():.0f}% matched balanced accuracy, with "
                   "overlapping 95% artist-bootstrap intervals. Yet on unseen pre-2010 FMA tracks they pass anywhere from "
                   f"{100 * fma_t.min():.0f}% to {100 * fma_t.max():.0f}%. The lab score does not predict who gets falsely accused, "
                   f"the same pattern {mb}. "
                   + ("Unlike MuseBench's ResNet18, the ResNet18 trained here passed the most of them, so which design suffers is "
                      "not fixed either." if most.split("@")[0] == "resnet18_mel"
                      else f"Here {html.escape(label(most))} passed the most of them.")
                   + " Their unseen-generator and vocal/instrumental scores are in the four-condition table below.")
    elif best_m != best_h:
        pattern = ("The model that wins in the lab is not the one that accuses the fewest unfamiliar human tracks, the same pattern "
                   f"{mb}.")
    elif best_m == best_j:
        pattern = f"Here the lab winner also accuses the fewest unseen human tracks on both FMA and Jamendo, unlike what {mb}."
    else:
        pattern = (f"Here the lab winner also accuses the fewest old FMA tracks among the detectors trained here, unlike what {mb}. "
                   f"It is not the most cautious everywhere: on unseen Jamendo music it wrongly flags {pct(1 - bh.unseen_human_acc_jamendo)} "
                   f"of human tracks, against {pct(1 - bj.unseen_human_acc_jamendo)} for {html.escape(label(best_j))}.")
    # single-version hold-outs (the table) vs whole families (harder): name the main detector's family results
    Fm = R[(R.condition == "unseen_generator") & R["round"].str.startswith("family") & (R.detector == primary)]
    fam_note = ""
    if len(Fm) and primary in T.index:
        w = Fm.loc[Fm.recall.idxmin()]
        fam_note = (f"<p class='note col'>The unseen-generator columns count single-version hold-outs only. Holding out a whole family is "
                    f"harder (the 'ALL ...' columns in the unseen-generator figure): there the main detector, {html.escape(label(primary))}, "
                    f"averages {pct(Fm.balanced_accuracy.mean())} balanced accuracy (against {pct(T.loc[primary, 'unseen_gen_balanced_accuracy_mean'])}) "
                    f"and catches only {pct(w.recall, 0)} of {html.escape(w['round'][len('family_'):])} tracks with every "
                    f"{html.escape(w['round'][len('family_'):])} version held out. For the two off-the-shelf detectors, Suno v3.5 and Udio v1 are "
                    "left out of these columns because those detectors were trained on them.</p>")
    n_train = int((base_split(df) == "train").sum())
    # vocal vs instrumental is partly a change of generator: open-weight models are mostly instrumental
    OPEN = {"musicgen", "magnet", "jasco", "audioldm2", "mustango", "stable_audio_open", "riffusion_v1", "acestep_1.5", "yue", "diffrhythm"}
    ai_t = df[(df.label == 1) & df.vocal.notna()]
    open_inst = float(ai_t[ai_t.vocal == 0].model.isin(OPEN).mean())
    comm_voc = float((~ai_t[ai_t.vocal == 1].model.isin(OPEN)).mean())
    vconf = vocal_generator_confound(df, primary)
    vi, iv = vconf.get("vocal_to_instr", {}), vconf.get("instr_to_vocal", {})
    vi_conf = "" if not vi.get("n_novel") else (
        f"<p class='col'>For the main detector the size of that effect can be measured. In the vocal → instrumental round, {vi['n_novel']} of the "
        f"{vi['n_ai']} AI test clips come from {len(vi['novel_generators'])} generators that never appear in vocal training "
        f"({html.escape(', '.join(vi['novel_generators']))}). Trained on vocal music, it catches {pct(vi['cross_recall_novel'])} "
        f"of those and {pct(vi['cross_recall_seen'])} of the rest; trained on instrumentals, {pct(vi['same_recall_novel'])} and "
        f"{pct(vi['same_recall_seen'])}. Counting only AI clips from generators present in vocal training, its vocal → instrumental "
        f"balanced accuracy is {pct(vi['cross_balanced_accuracy_seen_only'])} instead of {pct(vi['cross_balanced_accuracy'])}. "
        f"Separately, it wrongly calls {pct(vi['cross_fpr'])} of human instrumentals AI, against {pct(vi['same_fpr'])} when trained "
        "on instrumentals; generator novelty does not explain that, since it affects AI clips only."
        + (f" The reverse round has the same problem on a smaller scale: in instrumental → vocal, {iv['n_novel']} of the "
           f"{iv['n_ai']} AI test clips come from {len(iv['novel_generators'])} generators absent from instrumental training "
           f"({html.escape(', '.join(iv['novel_generators']))}); it catches {pct(iv['cross_recall_novel'])} of those against "
           f"{pct(iv['cross_recall_seen'])} of the rest, and counting only seen generators its balanced accuracy is "
           f"{pct(iv['cross_balanced_accuracy_seen_only'])} instead of {pct(iv['cross_balanced_accuracy'])}." if iv.get("n_novel") else "")
        + " So both vocal/instrumental columns measure an unseen-generator problem as well as a vocal/instrumental one.</p>")
    novel_note = " and ".join(f"{len(d['novel_generators'])} {k}" for k, d in (("one way", vi), ("the other", iv)) if d.get("novel_generators"))
    gen_scope = ("Generators are kept apart on purpose only in the unseen-generator rounds: the matched and both unseen-human rounds use "
                 "the same generators in training and test"
                 + (f", and both vocal/instrumental cross rounds include test generators absent from training because of the data "
                    f"({novel_note}), not by design." if novel_note else
                    ", as do the vocal/instrumental rounds."))
    # provenance facts the legal notes depend on
    lic = df.license.astype(str)
    is_hist = lic.str.contains(HIST_LICENCE, regex=False)
    n_hist = int(is_hist.sum())
    hist_where = ("all in FMA's 'Old-Time / Historic' genre" if (df[is_hist].genre_meta == "Old-Time / Historic").all()
                  else f"{int((df[is_hist].genre_meta == 'Old-Time / Historic').sum())} of them in FMA's 'Old-Time / Historic' genre")
    hist_lic = html.escape(", ".join(sorted(df[is_hist].license.unique())))
    n_scraped = int(lic.str.startswith("unclear").sum())
    # AI datasets with a different group key for every clip (no uploader or prompt recorded): split per track only
    per_track = sorted(s for s, g in df[df.label == 1].groupby("source") if g.artist_id_platform.nunique() == len(g))
    per_track_txt = "" if not per_track else (
        f" {words.get(len(per_track), str(len(per_track))).capitalize()} AI datasets ({html.escape(', '.join(per_track))}) "
        "have a different group key for every clip (no uploader or prompt is recorded"
        + ("; MusicGen's source publishes prompts, but the download step does not read them" if "musicgen" in per_track else "")
        + "), so they are split per track only and clips made from the same prompt can sit on both sides.")
    ai_df = df[df.label == 1]
    fams = sorted(f for f, g in ai_df.groupby("family") if g.model.nunique() > 1)
    vc = ai_df.model.value_counts()
    small = sorted(vc[vc < MIN_GEN_CLIPS].index)
    fam_txt = (f"Each AI model with at least {MIN_GEN_CLIPS} clips is removed from training in turn and the detector must still catch it. "
               f"Every family with more than one version ({', '.join(fams)}) is also held out as a whole, because siblings share decoders "
               "and make single-version hold-outs look easier than they are."
               + (f" {', '.join(small)} (too few clips) are held out only with their family." if small else ""))
    body = f"""
<header>
  <div class="kicker">AI-music detection · cross-domain evaluation</div>
  <h1>Who gets called AI?</h1>
  <p class="lede col">{n_det} detectors built from a pool of {n_h + n_ai:,} ten-second clips ({n_h:,} human, {n_ai:,} AI from {len(gens)} generator versions;
  no detector trains on more than {n_train:,} of them), then tested where detectors usually fail: on music from sources they never saw.
  The question is how often a real musician gets accused.</p>
</header>

<section>
  <h2>The headline</h2>
  <div class="verdict">
    <div class="stat good"><b>{pct(1 - bh.unseen_human_acc_fma_pre2010)}</b><span>of unseen pre-2010 FMA tracks falsely called AI by {html.escape(label(best_h))}{' (the main detector)' if best_h == primary else ''}, the lowest rate among the detectors trained here; on unseen Jamendo tracks it falsely calls {pct(1 - bh.unseen_human_acc_jamendo)} AI. Beside it: {matched_txt(bh)}; {unseen_gen_txt(bh)}; {vi_txt(bh)}{ph_note}</span></div>
    <div class="stat bad"><b>{pct(1 - wh.unseen_human_acc_fma_pre2010)}</b><span>of those same FMA tracks falsely called AI by {html.escape(label(worst_h))}; on unseen Jamendo tracks it falsely calls {pct(1 - wh.unseen_human_acc_jamendo)} AI. Beside it: {matched_txt(wh)}; {unseen_gen_txt(wh)}; {vi_txt(wh)}</span></div>
    <div class="stat"><b>{pct(bm.matched_balanced_accuracy)}</b><span>the best lab score (matched split) among detectors with all four conditions finished, from {html.escape(label(best_m))}. Beside it: {human_txt(bm)}; {unseen_gen_txt(bm)}; {vi_txt(bm)}{hp_note}</span></div>
  </div>
  <p class="col">{pattern}</p>
  {band_note}
</section>

<section>
  <h2>All four conditions, side by side</h2>
  <p class="col">No number on this page should be quoted without the others. Balanced accuracy averages the share of AI caught and the share of humans passed.
  Unseen human = accuracy on human-only music from a source absent from training. † the public lofcz model was itself trained on FMA.</p>
  {table(four, "wide")}
  {fam_note}
  {img("four_conditions.png", "Every detector under the four conditions.")}
</section>

<section>
  <h2>Unseen generators</h2>
  <p class="col">{fam_txt}</p>
  {img("unseen_generator.png", "Share of each held-out generator's tracks caught. Blank cells: the off-the-shelf detectors were trained on Suno v3.5 and Udio v1, so those are not unseen for them.")}
</section>

<section>
  <h2>Unseen human music</h2>
  <p class="col">Trained with human music from one platform only, tested on another. FMA uploads from before 2010 predate AI song generators by more than a decade,
  so every one of them is human. GuitarSet is solo acoustic guitar, a deliberately hard case.</p>
  {img("unseen_human.png", "Accuracy on human music from sources never seen in training.")}
</section>

<section>
  <h2>Vocal and instrumental</h2>
  <p class="col">Trained on one, tested on the other, in both directions. Caveat: {pct(open_inst, 0)} of instrumental AI clips come from open-weight models
  (MusicGen, MAGNeT, Stable Audio Open, AudioLDM2...) while {pct(comm_voc, 0)} of vocal ones come from commercial services (Suno, Udio, Mureka...),
  so this condition partly measures a change of generator too.</p>
  {vi_conf}
  {img("vocal_instr.png", "In-domain versus cross-domain balanced accuracy.")}
</section>

<section>
  <h2>The original hypothesis: does a detector fail on a genre it never saw?</h2>
  <p class="col">For each genre with enough test clips, the same detector is trained with and without that genre and tested on it. A large drop would support "detectors only work per genre".
  Genres come from a zero-shot CLAP tagger. It agrees with FMA's own genre labels on {pct(g_agree)} of {g_n:,} FMA clips across {g_k} coarse genres
  (agreement expected by chance, given how often each genre occurs, is about {pct(g_chance, 0)}; always answering {html.escape(g_major_name)} would score
  {pct(g_major, 0)}), so genre groups are noisy: treat this as a coarse test of the hypothesis. Both training sets in a pair are the same size,
  and a dataset that is almost all one genre (GuitarSet, solo acoustic guitar, for country/folk) is not scored in that genre's rounds: removing the genre would
  remove the whole dataset, which is the unseen-human question, not the genre question.</p>
  {gen_txt}
  {gen_tab}
  {img("genre.png", "AUC when the genre was in training versus held out (retrained detectors only).")}
</section>

<section>
  <h2>Everyday edits</h2>
  <p class="col">800 matched test clips (400 human, 400 AI), human and AI alike, after re-encoding, pitch and speed changes and loudness normalisation
  (main detector shown; the retrained detectors and their MP3-trained variants in the figure; the two off-the-shelf models were not re-tested after edits).
  'clean' is the same 800 clips unedited, so it differs slightly from the full matched score. The two 'bandwidth' rows are not in the brief's list: they
  imitate old or low-sample-rate recordings, which this data ties to AI (see the shortcut note under the headline). All of these are matched-split
  clips: the same detectors' unseen-generator, unseen-human and vocal/instrumental results are in the four-condition table above.</p>
  {rob_tab}
  {img("robustness.png", "Robustness of the retrained detectors to everyday edits (off-the-shelf models not tested).")}
</section>

<section>
  <h2>Reverse engineering, done the legitimate way</h2>
  <p class="col">Suno and Udio do not publish their architectures. What their outputs show is the decoder's upsampling grid: narrow spectral peaks at every multiple of the
  decoder's internal frame rate (Afchar et al., ISMIR 2025). Averaging up to 80 clips per generator (generators with fewer than 20 clips are not scanned)
  and testing every spacing from 12 to 400 Hz in 0.05 Hz steps shows which grids are present. Lossy file formats leave grids of their own (MP3 cuts the
  spectrum into 32 equal bands), so the same human clips were re-encoded in the lossy formats the AI files came in (MP3 at 48 and 44.1 kHz, AAC, Opus)
  and scanned the same way, and each generator's clips were split in two halves that must agree.</p>
  {fp_text}
  {fp_rows}
  {img("fingerprint_comb_scan.png", "Comb contrast per generator across tested spacings.")}
</section>

<section>
  <h2>Chords and keys</h2>
  <p class="col">The brief suggested AI tracks use non-diatonic chords and abrupt key changes. That idea comes from a qualitative look at 30 AI rock tracks
  in MuseBench's appendix. Corpus studies do not support it for chords: Suno and Udio score about the same as human music on a diatonic-strength measure
  (Cros Vila et al., TISMIR 2025, doi:10.5334/tismir.254), and a large chord-analysis study found similar chord types in all three, with Suno (up to v3.5)
  leaning on stock pop progressions such as I-V-vi-IV and Udio statistically close to human music (Dalmazzo et al., TISMIR 2026, doi:10.5334/tismir.348).
  Neither study measured key changes. The optional harmony features test both on a sample.</p>
  {harm_line}
  <p class="col">"Physical impossibility" checks (more than ten simultaneous piano notes, guitar notes outside E2-E6) were not implemented: they need reliable transcription and
  instrument identification inside full mixes, which no CPU-friendly open model provides, and no published detector uses them.</p>
</section>

<section>
  <h2>Data and provenance</h2>
  <p class="col">Every clip is cut and decoded the same way (ffmpeg, soxr to 48 kHz, 16-bit, one 10-second excerpt per track). Splits never share an artist,
  an uploader or a recorded prompt/caption group, a held-out generator or a held-out human platform; the program stops if one does.{per_track_txt}
  {gen_scope}
  No present-day commercial releases were downloaded for training or testing (the public website does fetch 30-second store previews of them for each
  check; see Limits): the human clips are Creative Commons or academic (FMA, Jamendo, GuitarSet). Two exceptions need a lawyer's view before publishing:
  {n_hist} FMA clips tagged '{hist_lic}' ({hist_where}) are digitised early-20th-century commercial 78-rpm records, and
  {n_scraped:,} AI clips (Suno, Udio, Mureka, Riffusion, Sonauto) come from a scraped collection with no clear licence. No audio is redistributed.</p>
  {table(srcs[["kind", "source", "clips"]].rename(columns={"kind": "Kind", "source": "Dataset", "clips": "Clips"}))}
</section>

<section class="col">
  <h2>What the brief got right, and what it got wrong</h2>
  <ul>
    <li><b>MuseBench is real</b> (arXiv 2610.20310, October 2026) and its numbers check out, including the 28.4% figure.</li>
    <li><b>Deezer's detector</b> beats every neural model in MuseBench on unseen human music (94.4% vs at most 79.8%). In-domain it trails the five
    spectrogram and transformer models (97.6% vs 99.5-100%) but beats the two raw-waveform models (M5 94.3%, RawNet2 82.3%). MuseBench warns that its 94.4%
    may partly reflect a boundary that leans towards "human", since it misses many AI tracks from held-out generators.</li>
    <li><b>The 5-8 kHz / 12-18 kHz bands</b> come from one model's Grad-CAM, not from detectors in general.</li>
    <li><b>The Velvet Sundown's</b> bio called it "an ongoing artistic provocation"; the "spokesperson" was an impostor and the Suno claim is unconfirmed.</li>
    <li><b>Breaking Rust's</b> generator is unknown. No reporting says it was not Suno or Udio, so it cannot be called a confirmed unseen generator.</li>
    <li><b>Non-diatonic chords</b> as an AI tell is not supported by corpus studies of Suno and Udio (Cros Vila et al., TISMIR 2025; Dalmazzo et al.,
    TISMIR 2026) or by this project's probe (AI uses fewer out-of-key chords); the only evidence for it is qualitative (MuseBench, 30 tracks). Key changes are
    untested in the literature and at most weakly present here. A "second-order FFT" detector is not in the literature: MuseBench uses it only as an analysis
    tool, and the peaks are visible in the ordinary spectrum.</li>
  </ul>
</section>

<section class="col">
  <h2>Limits</h2>
  <ul>
    <li>Ten-second excerpts, CPU-only training. ResNet18 trained on at most 3,000 clips per round for 4 epochs, far less than published baselines. A longer
    run was not tested and could score better or worse on unseen music, including the false-positive rate on human tracks.</li>
    <li>The datasets differ in sample rate, channels and bitrate. Every clip is re-decoded identically, but traces of the original format remain (see the
    leakage report). {fmt_limit}</li>
    <li>Vocal labels mix metadata with an AudioSet tagger: {pct(v_tagged, 0)} of clips were labelled by the tagger, which agrees with the known
    labels {pct(v_agree)} of the time (balanced), so some clips sit on the wrong side of the vocal/instrumental line. Genre labels are coarse.</li>
    <li>No Suno v6 or Udio v1.5 Allegro data, and no tracks generated by this project (tier A). Suno's flagship v6 models need a paid plan, and Udio has
    disabled audio downloads on every plan since late October 2025, so new Udio audio cannot currently be collected at all.</li>
    <li>The main detector{' and the redesign that capped its spectrogram at ' + f'{FMAX / 1000:g}' + ' kHz (it first saw up to 16 kHz) were' if primary == 'resnet18_mel' else ' was'}
    chosen after looking at the same test rounds that are reported here. There is no untouched final test set, so its numbers may be slightly optimistic.</li>
  </ul>
  <p class="caution">A detector score is evidence, not proof. Before publishing the paper, the trained models or derived data, or verdicts about
  named artists, get legal advice. The public website (<a href="{WEBSITE}">{WEBSITE}</a>) went live before that review: for each check it fetches the
  song's 30-second store preview from Deezer or Apple to a temporary file, which their API terms may not allow, reads titles from Spotify and YouTube
  pages, and shows the artist's name next to a verdict (details in the model card's legal notes). Other points for the review: Deezer's method is
  patent-pending and licensed non-commercially; the datasets mix licences, most of them non-commercial, and the Jamendo clips' per-track licences were
  not recorded; {n_scraped:,} AI clips were scraped with no clear licence; AI generators' own terms (Suno's, for example) restrict using their outputs to
  train other models; {n_hist} FMA clips are historic commercial records; the ResNet18 starts from ImageNet-pretrained weights; and calling a real
  musician "AI" can harm their career.</p>
</section>

<footer>Generated by <code>scripts/make_report.py</code> from the files in <code>results/</code>. Re-run it after any change to the pipeline.</footer>
"""
    page = f"""<title>Who Gets Called AI</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=JetBrains+Mono:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>{CSS}</style>
<main>{body}</main>
"""
    out = RESULTS / "report.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
