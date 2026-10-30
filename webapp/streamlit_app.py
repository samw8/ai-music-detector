"""Web page: paste a Spotify / Apple Music / YouTube / Deezer link (or type "Artist - Song", or upload a file)
and get the main detector's P(AI) for that song, with the same caveats predict.py prints.

    streamlit run webapp/streamlit_app.py      # then open http://localhost:8501

Online it runs on Streamlit Community Cloud straight from the GitHub repository (RESEARCH.md: "The
website"). The trained weights are not in the repository: they are downloaded at start-up from a private
Hugging Face model repository (webapp/upload_model.py) with the token stored as the app secret HF_TOKEN.

Scores audio with predict.py's decoder, excerpt positions and ResNet18; each 10-second excerpt is decoded on
its own, as the training clips were, so a long upload never sits in memory whole. For links, what is
analysed is the song's official 30-second preview (see links.py); nothing is stored. When the song found
for a link is not a confident match, the visitor confirms it before it is scored.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the repository: its aimd package, predict.py and models/
sys.path.insert(0, str(HERE))

import streamlit as st  # noqa: E402

import links  # noqa: E402
from aimd import MODELS  # noqa: E402
from aimd.audio import decode_stereo, probe, to_pcm16  # noqa: E402
from aimd.cnn import MelResNet, logmel  # noqa: E402
from predict import POSITIONS, SEG, SR, excerpts  # noqa: E402

REPO_URL = "https://github.com/samw8/ai-music-detector"
MODEL_REPO = "norows/ai-music-detector-model"  # private; override with the MODEL_REPO secret
MAX_SECONDS = 600       # uploads: excerpts are taken from the first 10 minutes
FALLBACK_SECONDS = 240  # when a file's length is unknown, its first 4 minutes are decoded instead
MAX_NATIVE_SR = 192000
MAX_UPLOAD_MB = 100
CFG = json.loads((MODELS / "config.json").read_text(encoding="utf-8"))
LO, HI = CFG.get("uncertain_band", [0.2, 0.8])


class Problem(Exception):
    """Shown to the visitor as is."""


def _secret(name: str, default: str | None = None) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        return st.secrets.get(name, default)
    except Exception:  # noqa: BLE001 - no secrets file (running locally)
        return default


@st.cache_resource(show_spinner="Loading the detector (first visit after a restart takes a minute)...")
def _detector() -> MelResNet:
    """models/resnet18_mel.pt, downloaded from the private model repository when it is not already here.
    The settings file comes first and the weights last, so a half-finished download is retried next time."""
    pt, settings = MODELS / "resnet18_mel.pt", MODELS / "resnet18_mel.settings.txt"
    if not (pt.exists() and settings.exists()):
        from huggingface_hub import hf_hub_download

        for f in (settings, pt):
            hf_hub_download(_secret("MODEL_REPO", MODEL_REPO), f.name, local_dir=MODELS, token=_secret("HF_TOKEN"))
    return MelResNet().load(pt)


def _decode(path: str, offset: float, dur: float, sr_native: int) -> np.ndarray:
    try:
        y = decode_stereo(path, offset, dur, sr_native, SR, max_frames=int((dur + 0.5) * sr_native))
    except ValueError:
        raise Problem("That file's audio does not match its own header, so it cannot be checked. "
                      "Try saving it again as MP3 or WAV.") from None
    if not np.isfinite(y).all():
        raise Problem("That file contains invalid audio samples, so it cannot be checked.")
    return to_pcm16(y)


def _excerpts(path: str) -> list[np.ndarray]:
    """predict.py's three 10-second excerpts (at 25%, 50% and 75% of the first 10 minutes), each decoded alone."""
    info = probe(path)
    sr_native, total = info["orig_sr"], min(info["duration"] or 0.0, MAX_SECONDS)
    if not sr_native:
        raise ValueError("no audio stream")
    if sr_native > MAX_NATIVE_SR:
        raise Problem("That file's sample rate is unusually high (over 192 kHz). Try an MP3, M4A or WAV version.")
    n = int(SR * SEG)
    if total > SEG + 1:
        segs = [_decode(path, pos * (total - SEG), SEG, sr_native) for pos in POSITIONS]
        if all(s.shape[1] >= n - SR // 10 for s in segs):  # the length ffmpeg reported was right
            segs = [np.pad(s[:, :n], ((0, 0), (0, max(0, n - s.shape[1])))) for s in segs]  # resampling can end a few samples short
            kept = [s for s in segs if np.sqrt(np.mean(s ** 2)) > 10 ** (-50 / 20)]
            if kept:
                return kept
            first = _decode(path, 0.0, SEG, sr_native)[:, :n]  # all three silent: the first 10 seconds, as predict.py
            return [np.pad(first, ((0, 0), (0, n - first.shape[1])))]
    # Short file, or its length was unknown: its first 4 minutes (less at high sample rates, to bound memory).
    y = _decode(path, 0.0, FALLBACK_SECONDS * min(1.0, SR / sr_native), sr_native)
    if y.shape[1] < SR:
        raise Problem("That audio is shorter than one second.")
    return excerpts(y)


def _score(path: str) -> tuple[float, list[float]]:
    mels = np.stack([logmel(to_pcm16(c).mean(axis=0)) for c in _excerpts(path)])
    p = np.ravel(_detector().predict_proba(mels)).astype(float)
    if not np.isfinite(p).all():
        raise Problem("The detector could not score that audio. Try another file.")
    return float(p.mean()), [round(float(x), 2) for x in p]


def _song_html(m: links.Match) -> str:
    cover = f'<img src="{html.escape(m.cover)}" alt="">' if m.cover.startswith("https://") else ""
    page = (f' · <a href="{html.escape(m.page)}" target="_blank" rel="noopener">open on {m.preview_from}</a>'
            if m.page.startswith("https://") else "")
    return (f'<div class="aimd-song">{cover}<div><div class="aimd-title">{html.escape(m.title)}</div>'
            f'<div class="aimd-sub">{html.escape(m.artist)}{" · " + html.escape(m.album) if m.album else ""}</div>'
            f'<div class="aimd-small aimd-sub">Found from your {html.escape(m.asked_via)}{page}</div></div></div>')


def _card(p: float, parts: list[float], what: str, song_html: str = "") -> str:
    if p >= HI:
        verdict, cls, line = "Likely AI-generated", "ai", "The detector finds strong signs of AI generation."
    elif p <= LO:
        verdict, cls, line = "Likely human-made", "human", "The detector finds few signs of AI generation."
    else:
        verdict, cls, line = "Uncertain", "unsure", "The detector cannot tell either way."
    return f"""
<div class="aimd-card">
  {song_html}
  <div class="aimd-score {cls}">
    <div class="aimd-num">{p:.2f}</div>
    <div><div class="aimd-verdict">{verdict}</div><div class="aimd-sub">{line}</div></div>
  </div>
  <div class="aimd-bar"><div class="aimd-fill {cls}" style="width:{100 * p:.0f}%"></div>
    <span class="aimd-tick" style="left:{100 * LO:.0f}%"></span><span class="aimd-tick" style="left:{100 * HI:.0f}%"></span></div>
  <div class="aimd-scale aimd-sub"><span>0 = human</span><span>P(AI)</span><span>1 = AI</span></div>
  <p class="aimd-small aimd-sub">Analysed: {html.escape(what)}. Scores of the 10-second slices: {", ".join(f"{x:.2f}" for x in parts)}.</p>
  <p class="aimd-small"><b>A score is evidence, not proof.</b> Never use it on its own to say an artist used AI.</p>
  <details><summary>How often is it wrong?</summary><p class="aimd-small aimd-sub">{html.escape(CFG.get("caveat", ""))}</p></details>
</div>"""


def find_song(text: str) -> links.Match:
    try:
        return links.resolve(text)
    except links.LookupError_ as e:
        raise Problem(str(e)) from None
    except Exception:  # noqa: BLE001 - network trouble with a music service
        raise Problem("Could not reach the music service just now. Try again, or type 'Artist - Song'.") from None


def check_song(m: links.Match) -> str:
    try:
        audio = links.fetch_preview(m)
    except links.LookupError_ as e:
        raise Problem(str(e)) from None
    except Exception:  # noqa: BLE001
        raise Problem("Could not download that song's preview clip just now. Press Check to try again.") from None
    suffix = ".m4a" if m.preview_from == "Apple" else ".mp3"
    with tempfile.TemporaryDirectory() as tmp:  # the preview is deleted as soon as it has been scored
        f = Path(tmp) / f"preview{suffix}"
        f.write_bytes(audio)
        try:
            p, parts = _score(str(f))
        except Problem:
            raise
        except Exception:  # noqa: BLE001
            raise Problem("Could not read that song's preview clip. Try another link, or upload the audio file.") from None
    return _card(p, parts, f"the official 30-second preview from {m.preview_from}", _song_html(m))


def check_file(up) -> str:
    if up.size > MAX_UPLOAD_MB * 1_000_000:
        raise Problem(f"That file is larger than {MAX_UPLOAD_MB} MB. Try an MP3 or M4A version.")
    suffix = Path(up.name).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,5}", suffix):  # ffmpeg only needs a plausible extension
        suffix = ""
    with tempfile.TemporaryDirectory() as tmp:  # deleted as soon as it has been scored
        f = Path(tmp) / f"upload{suffix}"
        f.write_bytes(up.getvalue())
        try:
            p, parts = _score(str(f))
        except Problem:
            raise
        except Exception:  # noqa: BLE001
            raise Problem("Could not read that file as audio. Try an MP3, WAV, FLAC, M4A or OGG file.") from None
    song = f'<div class="aimd-song"><div><div class="aimd-title">{html.escape(up.name)}</div></div></div>'
    return _card(p, parts, "your uploaded file (excerpts from its first 10 minutes)", song)


def _plain(text: str) -> str:
    """st.error renders Markdown: escape it so song titles in messages are shown as typed."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|$<>~:])", r"\\\1", text)


# Colours that read in both Streamlit's light and dark themes (text inherits the theme's colour).
STYLE = """<style>
.aimd-card { border: 1px solid rgba(128,128,128,.35); border-radius: 12px; padding: 18px; display: grid; gap: 12px; }
.aimd-card p { margin: 0; }
.aimd-song { display: flex; gap: 14px; align-items: center; }
.aimd-song img { width: 84px; height: 84px; border-radius: 8px; object-fit: cover; }
.aimd-title { font-size: 1.15rem; font-weight: 700; }
.aimd-sub { opacity: .75; }
.aimd-small { font-size: .88rem; }
.aimd-score { display: flex; gap: 16px; align-items: center; }
.aimd-num { font-size: 2.6rem; font-weight: 800; font-variant-numeric: tabular-nums; min-width: 3.2ch; line-height: 1.1; }
.aimd-verdict { font-size: 1.25rem; font-weight: 700; }
.aimd-score.ai .aimd-num, .aimd-score.ai .aimd-verdict { color: #d4500f; }
.aimd-score.human .aimd-num, .aimd-score.human .aimd-verdict { color: #1f8a5b; }
.aimd-score.unsure .aimd-num, .aimd-score.unsure .aimd-verdict { color: #b7860b; }
.aimd-bar { position: relative; height: 12px; border-radius: 6px; background: rgba(128,128,128,.15); border: 1px solid rgba(128,128,128,.35); overflow: hidden; }
.aimd-fill { height: 100%; }
.aimd-fill.ai { background: #d4500f; } .aimd-fill.human { background: #1f8a5b; } .aimd-fill.unsure { background: #b7860b; }
.aimd-tick { position: absolute; top: 0; bottom: 0; width: 2px; background: currentColor; opacity: .45; }
.aimd-scale { display: flex; justify-content: space-between; font-size: .8rem; }
.aimd-note { padding: 8px 10px; border-radius: 8px; background: rgba(128,128,128,.14); }
.aimd-card summary { cursor: pointer; }
</style>"""

INTRO = """
Paste a **Spotify, Apple Music, YouTube or Deezer** link, or type **Artist - Song**. The detector finds the song's
official 30-second preview and checks it for signs of AI generation. You can also upload an audio file.
"""

ABOUT = f"""
**How it works.** A ResNet18 network looks at the clip's spectrogram up to 7 kHz. It was built and tested on a pool
of 12,265 ten-second clips (5,553 human; 6,712 from 23 AI generator versions such as Suno, Udio, Mureka and MusicGen),
always tested on artists it never saw, and also on whole music sources and AI generators left out of training,
because that is where detectors fail and real musicians get wrongly accused. Full results, methods and limits:
[GitHub]({REPO_URL}).

**What it is not.** It cannot prove anything about an artist. It is weakest on very old or lo-fi recordings and on
songs from AI generators it has never seen. Links are matched to the song's 30-second preview, so it does not hear the
whole song. Nothing you paste or upload is stored.
"""

st.set_page_config(page_title="Who Gets Called AI", page_icon="🎵", layout="centered")
st.title("Is this song AI-generated?")
st.markdown(INTRO)
try:
    _detector()
except (Exception, SystemExit):  # noqa: BLE001 - MelResNet.load exits on a settings mismatch
    traceback.print_exc()  # shown in the app's logs, not to visitors
    st.error("The detector could not be loaded, so the page cannot check songs right now. Please try again later.")
    st.stop()

with st.form("song", border=False):
    c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
    text = c1.text_input("Song link or 'Artist - Song'", placeholder="https://open.spotify.com/track/...  or  Radiohead - Karma Police",
                         max_chars=links.MAX_TEXT)
    asked = c2.form_submit_button("Check", type="primary", width="stretch")
slot = st.container()  # the result goes here, above the upload box
with st.expander("...or upload your own audio file"):
    up = st.file_uploader("Audio file (MP3, WAV, FLAC, M4A, OGG)", type=["mp3", "wav", "flac", "m4a", "aac", "ogg", "opus"],
                          max_upload_size=MAX_UPLOAD_MB)
    asked_file = st.button("Check this file", disabled=up is None)

state = st.session_state
try:
    if asked:
        with slot, st.spinner("Finding the song..."):
            m = find_song(text)
        if m.similarity >= links.CONFIDENT:
            with slot, st.spinner("Listening to its 30-second preview..."):
                state.result = ("ok", check_song(m))
        else:
            state.result = ("confirm", m)
    elif asked_file and up is not None:
        with slot, st.spinner("Listening to your file..."):
            state.result = ("ok", check_file(up))
except Problem as e:
    state.result = ("error", str(e))

kind, shown = state.get("result", ("", ""))
with slot:
    if kind == "error":
        st.error(_plain(shown))
    elif kind == "ok":
        st.html(STYLE + shown)
    elif kind == "confirm":
        st.html(STYLE + '<div class="aimd-card">' + _song_html(shown) + '<p class="aimd-note">This is the closest match the '
                "music services returned, so it may not be the song you meant. Check it before it is scored, or type "
                "<b>Artist - Song</b> to be more specific.</p></div>")
        if st.button("Yes, check this song", type="primary"):
            try:
                with st.spinner("Listening to its 30-second preview..."):
                    state.result = ("ok", check_song(shown))
            except Problem as e:
                state.result = ("error", str(e))
            st.rerun()
st.markdown(ABOUT)
