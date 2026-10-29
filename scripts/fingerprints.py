"""Step 11 - generator fingerprints: forensic "reverse engineering" from public outputs only.

    python scripts/fingerprints.py

Suno and Udio do not publish their architectures, and nothing here tries to access
their systems. What can be measured legitimately is what their decoders leave
behind: transposed-convolution upsampling puts narrow spectral peaks at every
multiple of the decoder's latent frame rate (Afchar et al., ISMIR 2025). Averaging
the spectrum over many clips from one generator makes that comb visible, and its
spacing says something about the decoder:

    spacing (Hz) = output sample rate / total upsampling factor

The method is checked on open generators whose decoders are published before it is
applied to closed ones: MusicGen, MAGNeT and JASCO decode with EnCodec-32k (predicted
grid: multiples of 50 Hz); Stable Audio Open's Oobleck VAE predicts 21.5 Hz. A
prediction that fails is reported, not hidden.

Two controls keep a comb from being over-read:
  * codec controls: lossy formats leave grids of their own (MP3 splits the spectrum into
    32 equal subbands, 750 Hz wide at 48 kHz, so a comb at 375 or 250 Hz can be the codec).
    Human clips are re-encoded in every delivery format the AI datasets use and scanned
    the same way; the strongest comb in any human or codec-control set is the ceiling.
  * split-half replication: each generator's clips are split in two and scanned
    separately; a real decoder comb shows up at the same spacing in both halves, while
    the best spacing of noise wanders.
A generator "has a comb" only if it clears the ceiling AND replicates.

Output: results/fingerprints.csv, results/figures/fingerprints.png, results/figures/fingerprint_combs.png
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA, REPO, RESULTS  # noqa: E402
from aimd.fakeprint import KNOWN_DECODERS, mean_db_spectrum  # noqa: E402

SR = 48000
FMIN, FMAX = 500.0, 16000.0
PER_GEN = 80


def residual(mean_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average spectrum minus its running lower envelope (~100 Hz window): only narrow peaks remain."""
    import scipy.ndimage as ndi

    f = np.linspace(0, SR / 2, len(mean_db))
    m = (f >= FMIN) & (f <= FMAX)
    x = mean_db[m]
    k = int(round(100 / (f[1] - f[0]))) | 1
    floor = ndi.uniform_filter1d(ndi.minimum_filter1d(x, k), k)
    return f[m], np.clip(x - floor, 0, None)


SPACINGS = np.round(np.arange(12.0, 400.0 + 0.025, 0.05), 2)  # 12-400 Hz inclusive: 400 Hz is the EnCodec-32k grid (8 x 50 Hz)


def comb_contrast(f: np.ndarray, r: np.ndarray, spacing: float) -> float:
    """Mean residual ON a grid of evenly spaced frequencies minus mean residual HALFWAY between them (dB).

    A decoder comb at spacing S gives a large positive value at S and ~0 at S/2, S/3 (half of
    those grid lines fall between peaks) and at 2S, 3S (their midpoints land on peaks).
    """
    n = np.arange(np.ceil(f[0] / spacing), np.floor(f[-1] / spacing))
    on = np.interp(n * spacing, f, r)
    off = np.interp((n + 0.5) * spacing, f, r)
    return float(on.mean() - off.mean())


_GRIDS: dict = {}


def comb_scan(f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """comb_contrast() at every spacing in SPACINGS at once (the grid points are built once per frequency axis)."""
    key = (len(f), float(f[0]), float(f[-1]))
    if key not in _GRIDS:
        on, off, seg = [], [], []
        for i, s in enumerate(SPACINGS):
            n = np.arange(np.ceil(f[0] / s), np.floor(f[-1] / s))
            on.append(n * s)
            off.append((n + 0.5) * s)
            seg.append(np.full(len(n), i))
        seg = np.concatenate(seg)
        _GRIDS[key] = (np.concatenate(on), np.concatenate(off), seg, np.bincount(seg, minlength=len(SPACINGS)))
    on, off, seg, cnt = _GRIDS[key]
    k = len(SPACINGS)
    return (np.bincount(seg, np.interp(on, f, r), k) - np.bincount(seg, np.interp(off, f, r), k)) / cnt


def comb(f: np.ndarray, r: np.ndarray) -> dict:
    """Best-matching comb spacing and its contrast, plus the contrast on each published decoder grid."""
    c = comb_scan(f, r)
    i = int(np.argmax(c))
    out = {"comb_spacing_hz": float(SPACINGS[i]), "comb_contrast_db": float(c[i])}
    for name, (sr, strides) in KNOWN_DECODERS.items():
        out[f"contrast@{sr / np.prod(strides):.2f}Hz"] = comb_contrast(f, r, sr / np.prod(strides))
    out["_curve"] = c
    return out


def _at(sr_codec: int, codec, **kw):
    """Run a 48 kHz stereo clip [2, n] through a codec at sr_codec, back to 48 kHz (as prepare.py would decode it)."""
    from aimd.audio import _resample

    def run(y):
        z = _resample(y.T, SR, sr_codec).T if sr_codec != SR else y
        z = codec(np.ascontiguousarray(z), sr_codec, **kw)
        return _resample(z.T, sr_codec, SR).T if sr_codec != SR else z
    return run


def _controls() -> dict:
    from aimd import augment as A

    return {"MP3 48 kHz 128 kbps": _at(48000, A.mp3, kbps=128), "MP3 48 kHz 192 kbps": _at(48000, A.mp3, kbps=192),
            "MP3 48 kHz 320 kbps": _at(48000, A.mp3, kbps=320), "MP3 44.1 kHz 128 kbps": _at(44100, A.mp3, kbps=128),
            "MP3 44.1 kHz 192 kbps": _at(44100, A.mp3, kbps=192),
            "MP3 44.1 kHz 320 kbps": _at(44100, A.mp3, kbps=320), "AAC 44.1 kHz 112 kbps": _at(44100, A.aac, kbps=112),
            "Opus 48 kHz 96 kbps": _at(48000, A.opus, kbps=96)}


CODEC_CONTROLS = _controls()


def matched_control(codec: str, sr: int, kbps: float) -> str:
    """The codec control in the same delivery format as a generator's files ('' for uncompressed audio)."""
    if codec == "mp3":
        rates = (128, 192, 320)
        k = min(rates, key=lambda r: abs(r - kbps))
        return f"MP3 {'48' if sr == 48000 else '44.1'} kHz {k} kbps"
    return {"aac": "AAC 44.1 kHz 112 kbps", "opus": "Opus 48 kHz 96 kbps"}.get(codec, "")


def nearest_decoder(spacing: float) -> str:
    best, err = "", 1e9
    for name, (sr, strides) in KNOWN_DECODERS.items():
        f0 = sr / np.prod(strides)
        for mult in (1, 2, 4, 5, 8):  # the comb can also show up at a coarser layer's grid
            e = abs(spacing - f0 * mult) / (f0 * mult)
            if e < err:
                best, err = f"{name}: {f0:.2f} Hz x{mult}", e
    return best if err < 0.01 else "no published decoder matches"


def main() -> None:
    import pandas as pd
    import soundfile as sf

    from aimd import plotstyle as S
    from aimd.splits import load_table

    S.apply()
    import matplotlib.pyplot as plt

    df = load_table(REPO)
    df["who"] = np.where(df.label == 0, "human: " + df.human_group, df.model)
    groups = [g for g in df.who.value_counts().index if (df.who == g).sum() >= 20]
    groups = sorted(groups, key=lambda g: (g.startswith(("human", "control")), g))
    rng = np.random.default_rng(0)
    spectra, fmt = {}, {}
    for g in groups:
        sub = df[df.who == g]
        sub = sub.iloc[rng.permutation(len(sub))[:PER_GEN]]
        spectra[g] = np.stack([mean_db_spectrum(sf.read(str(DATA / p), dtype="float32", always_2d=True)[0].T) for p in sub.path])
        fmt[g] = (sub.codec.mode().iloc[0], int(sub.orig_sr.mode().iloc[0]), float(sub.kbps.median()))
        print(f"  spectra: {g} ({len(sub)} clips)", flush=True)

    # codec controls: the same human clips re-encoded in each delivery format used by the AI datasets
    pool = pd.concat([df[(df.label == 0) & (df.human_group == h)].sample(PER_GEN // 2, random_state=0) for h in ("fma", "jamendo")])
    raw = [sf.read(str(DATA / p), dtype="float32", always_2d=True)[0].T for p in pool.path]
    for name, fn in CODEC_CONTROLS.items():
        with ThreadPoolExecutor(4) as ex:  # ffmpeg does the work in its own processes
            spectra[f"control: {name}"] = np.stack(list(ex.map(lambda y: mean_db_spectrum(fn(y)), raw)))
        print(f"  spectra: control {name}", flush=True)
    groups += [f"control: {n}" for n in CODEC_CONTROLS]

    rows, curves, contrast_curves = [], {}, {}
    for g in groups:
        S_ = spectra[g]
        f, r = residual(S_.mean(axis=0))
        curves[g] = (f, r)
        c = comb(f, r)
        contrast_curves[g] = c.pop("_curve")
        idx = np.random.default_rng(1).permutation(len(S_))  # split-half replication
        halves = [float(SPACINGS[np.argmax(comb_scan(*residual(S_[h].mean(axis=0))))]) for h in np.array_split(idx, 2)]
        ref = not g.startswith(("human", "control"))
        rows.append({"group": g, "n_clips": len(S_), **c, "peak_height_db_p99": float(np.percentile(r, 99)),
                     "half_a_spacing_hz": halves[0], "half_b_spacing_hz": halves[1],
                     "replicates": abs(halves[0] - halves[1]) <= 0.01 * max(halves),
                     "codec": fmt.get(g, ("", 0, 0))[0], "orig_sr": fmt.get(g, ("", 0, 0))[1], "kbps": fmt.get(g, ("", 0, 0))[2],
                     "closest_published_decoder": nearest_decoder(c["comb_spacing_hz"]) if ref else ""})
        print(f"{g:28s} comb {c['comb_spacing_hz']:8.2f} Hz  contrast {c['comb_contrast_db']:5.2f} dB  halves {halves[0]:.2f}/{halves[1]:.2f}  "
              f"{rows[-1]['closest_published_decoder']}", flush=True)
    T = pd.DataFrame(rows)
    is_ref = T.group.str.startswith(("human", "control"))
    null = T[is_ref].comb_contrast_db.max()
    # the human audio in the SAME delivery format, measured at this generator's own spacing
    for i, row in T[~is_ref].iterrows():
        ctl = matched_control(row.codec, row.orig_sr, row.kbps)
        T.loc[i, "format_control"] = ctl
        if ctl:
            T.loc[i, "format_control_db_at_spacing"] = comb_contrast(*curves[f"control: {ctl}"], row.comb_spacing_hz)
    T["ceiling_db"] = null
    T["comb_found"] = ~is_ref & (T.comb_contrast_db > null) & T.replicates
    T.to_csv(RESULTS / "fingerprints.csv", index=False)
    print(f"human and codec-control sets reach at most {null:.2f} dB of comb contrast; a generator 'has a comb' only above that "
          "AND at the same spacing in both halves of its clips")
    print("comb found: " + ", ".join(T[T.comb_found].group))

    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 0.62 * len(groups) + 1), sharex=True)
    for ax, g in zip(axes, groups):
        ax.plot(SPACINGS, contrast_curves[g], lw=0.8, color=S.MUTED if g.startswith(("human", "control")) else S.SERIES[0])
        ax.axhline(null, color=S.AXIS, lw=0.8)
        ax.set_yticks([])
        ax.grid(False)
        row = T[T.group == g].iloc[0]
        ax.text(0.995, 0.6, f"{g}: {row.comb_spacing_hz:.2f} Hz", transform=ax.transAxes, ha="right", fontsize=7.5, color=S.INK)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xscale("log")
    axes[-1].set_xticks([12.5, 16.67, 21.53, 25, 50, 75, 86.13, 150, 300], ["12.5", "16.7", "21.5", "25", "50", "75", "86", "150", "300"])
    axes[-1].set_xlabel("comb spacing tested (Hz, log scale); horizontal line = strongest comb in any human or codec-control set")
    fig.suptitle("Comb contrast: how strongly each generator's spectrum peaks on an evenly spaced grid", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(h_pad=0.1)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "fingerprint_comb_scan.png")
    plt.close(fig)

    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(11, 0.75 * n + 1), sharex=True)
    ymax = max(float(np.percentile(r, 99.9)) for _, r in curves.values())
    for ax, g in zip(axes, groups):
        f, r = curves[g]
        human = g.startswith(("human", "control"))
        ax.plot(f / 1000, r, lw=0.6, color=S.MUTED if human else S.SERIES[0])
        ax.set_ylim(0, ymax)
        ax.set_yticks([])
        ax.grid(False)
        ax.text(0.003, 0.72, g, transform=ax.transAxes, fontsize=8, color=S.INK)
        ax.spines["left"].set_visible(False)
    axes[-1].set_xlabel("frequency (kHz)")
    fig.suptitle("Average spectral residual per generator: narrow evenly spaced peaks = decoder upsampling artifacts "
                 "(gray = human and codec-control references)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(h_pad=0.1)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / "figures" / "fingerprints.png")
    plt.close(fig)

    # zoom: 4-6 kHz, where combs at 21.5-150 Hz spacing are easy to see
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(13, 1.6 * np.ceil(n / 3) + 0.6), sharex=True)
    for ax, g in zip(axes.ravel(), groups):
        f, r = curves[g]
        m = (f >= 4000) & (f <= 6000)
        ax.plot(f[m] / 1000, r[m], lw=0.8, color=S.MUTED if g.startswith(("human", "control")) else S.SERIES[0])
        row = T[T.group == g].iloc[0]
        ax.set_title(f"{g}  (comb {row.comb_spacing_hz:.2f} Hz, {row.comb_contrast_db:.1f} dB)", fontsize=8, loc="left")
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    fig.suptitle("Zoom on 4-6 kHz", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "fingerprint_combs.png")
    print(f"wrote results/fingerprints.csv and two figures")


if __name__ == "__main__":
    main()
