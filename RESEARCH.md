# Running the research

How to check songs on your own computer, re-run every experiment, and redeploy the website.
For what the project is and its results, see `README.md`.

## Checking a song on your own computer

This needs the trained detector in `models\`, which is not on GitHub. You get it by running the steps
below (or from the project's authors).

Easiest: drag a song file (or a whole folder of songs) onto **`Check songs.bat`** in this folder.
A window opens and shows the result for each song.

Or open a Command Prompt in this folder (in File Explorer, click the address bar, type `cmd`, press Enter) and type:

```
run check "C:\path\to\song.mp3"
```

You get the probability that the song is AI-generated from the main detector (a ResNet18 network
that looks at the song's spectrogram) and a caveat with that detector's measured false-positive
rate on music unlike its training data. Below that is a short list of measurable audio features
from a separate, simpler model (interpretable features + logistic regression). The main detector
cannot explain its own score, so that list is **not** its reasons, and the simpler model's score
(printed beside it) can disagree. Point it at a folder to check every song inside.

The first version of the main detector mistook missing high frequencies for AI. It was retrained
with its spectrogram capped at 7 kHz, and band-limiting a modern recording to 8 kHz no longer changes
its verdict. Genuinely old or lo-fi recordings (almost no sound above 8 kHz) are still wrongly called
AI about twice as often as other human music, and phone-quality audio (below 8 kHz) was not tested;
see `MODEL_CARD.md`. If a retrain brings the shortcut back, `run evaluate` detects it and the
checker warns on low-sample-rate, low-bitrate or band-limited files again (the website does not show
that warning).

## The website

https://who-gets-called-ai.streamlit.app runs `webapp/streamlit_app.py`. For a link, it finds the song's
official 30-second preview on Deezer or Apple (`webapp/links.py`); for Spotify and YouTube links it reads
only the title and artist (from the public Spotify track page, or YouTube's oEmbed service, falling back
to the YouTube watch page when oEmbed refuses), then finds that preview. The preview is written to a
temporary file for scoring and deleted straight after; nothing is kept. When the song it finds is not a
confident match, it shows the match and asks before scoring it. It scores audio with the same decoder,
excerpt positions, detector and 0.2 / 0.8 verdict cut-offs as `run check`, but hears only those 30 seconds.
Any file longer than 11 seconds, including every preview, is decoded one 10-second excerpt at a time (as
the training clips were cut) to save memory, and long uploads use only their first 10 minutes, so scores
can differ from `run check`: typically by about 0.01, in tests by up to 0.07 on 30-second files and more
on files over 10 minutes. The published error rates are per 10-second excerpt of each source file (a
30-second clip for FMA, the full track for Jamendo); the site averages three excerpts of a 30-second
preview, and its error rates on previews have not been measured. Like `run check`, it picks the verdict
from the unrounded score and never shows a rounded number on the wrong side of a cut-off (0.797 shows as
0.79).

Redeploying (free, on Streamlit Community Cloud):

1. The trained detector is not on GitHub. It sits in a private Hugging Face model repository
   (`norows/ai-music-detector-model`). Upload it from a Command Prompt in this folder with
   `%USERPROFILE%\.aimd\venv\Scripts\hf auth login` (once, with a Hugging Face token that can write) and then
   `%USERPROFILE%\.aimd\venv\Scripts\python webapp\upload_model.py`. After every retrain, in this order:
   `run evaluate`, upload the weights, commit and push the regenerated `models\config.json` (with the other
   results), then reboot the app (step 3). The page refuses to start, showing "The detector could not be
   loaded", if the weights it downloads are not the file whose fingerprint `run evaluate` wrote into
   `models\config.json`, so the weights and the pushed `config.json` must always match.
2. At https://share.streamlit.io sign in with GitHub, click **Create app**, then "Deploy a public app from
   GitHub": repository `samw8/ai-music-detector`, branch `main`, main file path `webapp/streamlit_app.py`.
   Under **Advanced settings** choose Python 3.12 and paste into **Secrets** one line,
   `HF_TOKEN = "hf_..."`, with a fine-grained Hugging Face token from https://huggingface.co/settings/tokens
   that can only read `norows/ai-music-detector-model`.
3. After a retrain and upload, open the app's menu on Streamlit and choose **Reboot app** so it fetches
   the new detector.

The site goes to sleep after 12 hours without visitors; the first visit after that wakes it in about
a minute. Open it a few minutes before showing it to anyone.

To run it on your own computer instead: `run web` (the first time it installs the packages in
`webapp\requirements.txt` into `%USERPROFILE%\.aimd\webvenv`), then open http://localhost:8501. It needs
`models\resnet18_mel.pt`, or a Hugging Face login made with the `hf auth login` command above (or an
`HF_TOKEN` environment variable) with access to the private model repository.

## Re-running the research, step by step

Every command is typed in a Command Prompt opened in this folder. Each step can be stopped
(Ctrl+C) and started again; it picks up where it left off. Times are for this laptop
(8-core Ryzen AI 7, no graphics card).

| Step | Command | What it does | Time |
|---|---|---|---|
| 0 | `run setup` | Installs a private copy of Python and the libraries into `%USERPROFILE%\.aimd` (first time only). | 10 min |
| 1 | `run download` | Shows how many GB each dataset needs and the free disk space, asks before starting, then downloads (resumable). It stops if the disk has too little room. | ~1-3 h; ~39 GB download, ~68 GB free needed (download + excerpts + 5 GB margin) |
| 2 | `run prepare` | Cuts one 10-second excerpt per track, every file through the same decoder, at 48 kHz (clips that are silent once mixed to mono are rejected). Builds `data/manifest.csv` with each track's provenance tier. | ~40 min |
| 3 | `run embed` | Pretrained CLAP audio embeddings (`laion/clap-htsat-unfused`), for model 4 and for tagging. | ~30-60 min |
| 4 | `run tag` | Labels each clip vocal/instrumental and with a coarse genre, the same way for every dataset. Genre: zero-shot CLAP on the step-3 embeddings. Vocals: the dataset's own label or what the generator can do where known, otherwise an AudioSet tagger (AST) with a threshold calibrated on the known labels. | ~1.5-2 h |
| 5 | `run features` | The interpretable features and Deezer's fakeprint for every clip. | ~15 min |
| 6 | `run melspec` | Spectrograms (20 Hz-7 kHz) for the ResNet18 model; rebuilt automatically if the settings in `aimd/cnn.py` change. | ~25 min |
| 7 | `run leakage` | Duplicate / near-duplicate check, shuffled-label control, format-shortcut probe. Also lists test clips that have an exact copy in training, which step 13 leaves out. | 5 min |
| 8 | `run train` | Trains every model under every test condition; saves predictions. Fails loudly if any split leaks or any input is incomplete. | 6-8 hours (ResNet18 is most of it) |
| 8b | `run features-mp3` then `run train-mp3` | The codec-equalised variant: every clip through the same MP3 encoder, feature models retrained. | ~45 min |
| 9 | `run baselines` | Scores two published, already-trained detectors (SONICS SpecTTTra, a public Deezer-method model) on the same test clips. | ~1.5 h |
| 10 | `run robustness` | Re-scores test clips after MP3/OGG/AAC, pitch +/-2, speed 0.9x/1.1x, tempo 0.9x/1.1x (pitch kept), loudness -14 LUFS, plus two band-limit edits that imitate old or low-sample-rate recordings. | ~3 h |
| 11 | `run fingerprints` | Per-generator decoder fingerprints (the "reverse engineering" figure). | 10 min |
| 12 | `run harmony` | Optional: tests the chord/key features on 100 clips per dataset. | ~30 min |
| 13 | `run evaluate` | Builds the results table (CSV + Markdown), figures and confusion matrices; picks the main detector by its worst condition. Any model with unfinished rounds is marked "partial" and cannot be picked. | 5 min |
| 14 | `run modelcard` | Rewrites `MODEL_CARD.md` from the current results. | 1 min |
| 15 | `run report` | Writes `results\report.html`, a plain-language summary page with every figure. | 1 min |
| 16 | `run demo` | Case study on named artists, using your own copies placed in `demo\audio\<artist>\`. Use only the releases listed in the `which_releases` column of `demo\artists.csv` (not in the public repository; see below); for its two "track-level only" rows only the named track counts as AI. | depends |

`run all` does steps 1-15 in order. `run test` checks that the leakage guards still catch leaks.

**Retraining a detector.** Re-running a step never retrains a finished model: finished rounds and
finished robustness rows are skipped. To retrain one (for example after changing `aimd/cnn.py`), move
`results\predictions\<model>\` aside, delete that detector's rows from `results\robustness.csv`, then
run `run melspec` (ResNet18 only), `run train`, `run robustness` and `run evaluate`. The ResNet18's
cache, weights and predictions each record their spectrogram settings, and the program stops with an
error if they do not match `aimd/cnn.py`.

**Keep the laptop plugged in with the lid open** while long steps run: closing the lid puts Windows to sleep and pauses
everything (each step resumes where it stopped, but nothing happens while it sleeps).

Maintenance scripts (only needed when the rules change): `%USERPROFILE%\.aimd\venv\Scripts\python
scripts\refresh_meta.py` re-applies the label/grouping rules to downloaded metadata without
re-downloading; `%USERPROFILE%\.aimd\venv\Scripts\python scripts\check_mono.py` removes clips whose mono
mix is silent.

## What is in here

| Path | Contents |
|---|---|
| `aimd/sources.py` | Every dataset: source, licence, provenance tier, how a subset is fetched |
| `aimd/features.py` | The interpretable features, each with a one-sentence explanation |
| `aimd/fakeprint.py` | Deezer's Fourier fakeprint (ISMIR 2025 best paper), re-implemented |
| `aimd/splits.py` | The evaluation conditions and the leakage checks that stop the program |
| `aimd/models.py`, `aimd/cnn.py` | The detectors |
| `predict.py` | Check one song or a folder |
| `webapp/` | The website (`streamlit_app.py`), turning song links into official preview clips (`links.py`), uploading the detector to its private repository (`upload_model.py`) |
| `demo/artists.csv` | 21 publicly reported AI acts, 2 single AI-assisted tracks by otherwise human artists (track-level only; one is tier D, assumed), and 30 genre-matched human artists, with evidence and sources. It names real people, so it is kept out of the public repository until it has been reviewed. |
| `results/` | Tables, figures and reports (per-clip predictions are rebuilt by `run train` and not published) |
| `MODEL_CARD.md` | Training data, failure modes, the out-of-domain false-positive rate |

## Data and the rules that keep it honest

* **No present-day commercial releases were downloaded for training or testing, and no audio is
  redistributed.** (The website fetches a song's official 30-second preview to score it and deletes it
  straight away; see "The website" above. FMA does
  include 133 early-20th-century commercial 78-rpm records that it tags "Sound Recording Common Law
  Protection", and 1,845 AI clips come from a scraped collection with no clear licence: see the legal
  notes in `MODEL_CARD.md`.) Training uses public research
  datasets (FMA and Jamendo for human music, plus GuitarSet solo acoustic guitar: 4 of its 6 players are in the
  matched training split that the models in `models\` come from; Suno, Udio, Mureka, Riffusion, Sonauto, Lyria,
  ACE-Step, YuE, DiffRhythm, MusicGen and the MuseBench open-model outputs for AI). The named-artist
  case study runs only on files you supply, and only scores are kept.
* **Provenance tier per track** (A generated by us, B dataset with verified metadata, C public
  reporting, D assumed). Tier D never enters training.
* **Artist-disjoint splits everywhere; generator-disjoint only in the unseen-generator condition and
  source-disjoint only in the unseen-human condition.** The matched and unseen-human conditions use the
  same AI generators in training and test by design. The vocal/instrumental conditions are not
  generator-disjoint either, but because of the data some test generators are missing from training in
  both cross rounds (vocal to instrumental: 7 generators, 437 of 723 AI test clips; instrumental to vocal:
  DiffRhythm and YuE, 162 of 653; see `MODEL_CARD.md`). The program stops with an error if a split leaks.
  For AI clips, "artist" means the creator account or text prompt where the dataset records one;
  MusicGen's prompts are not read yet, and Suno v5.5, Udio v1.5, Mureka, YuE and DiffRhythm record
  neither, so those clips are kept apart per track only.
* **Do not publish the audio.** Before publishing anything else from this folder, have a lawyer or the
  university's copyright office look at it: the trained models in `models\`, `data\features*.csv`,
  `data\emb_clap.npy`, `data\fakeprints*.npy` and `data\manifest.csv` are all derived from audio under
  NonCommercial, NoDerivatives, ShareAlike and "unclear (scraped; research only)" terms, and the manifest
  holds Suno/Udio/Riffusion/Sonauto user IDs (drop or hash the `artist_id` column before any release).
  The raw sidecar `.json` files under `%USERPROFILE%\.aimd\data\raw` also hold platform users' display
  names, handles and avatar links: strip them before sharing. Until then, share these files only with
  research collaborators.

## Not done yet

* **Test tracks generated by this project (tier A).** Suno v6 needs a paid plan, and Udio
  has disabled downloads on every plan since late October 2025, so a Udio subscription would not produce files.
* **Commercial detector APIs** (Ircam Amplify, Resemble AI) need paid keys, so they were not called.
* **Legal review** before publishing the paper, the trained models, the derived data files or
  `demo\artists.csv`, before publishing per-artist verdicts, and for the public website, which went live
  before that review - see `MODEL_CARD.md`. The website downloads Deezer's and Apple's preview clips to a
  temporary file to analyse them (Deezer's API guidelines forbid storing audio, and its API Terms of Use
  limit use to non-commercial purposes; Apple's Search API terms say previews are for promoting the store, streamed rather than
  downloaded, shown near a store badge and credited "provided courtesy of iTunes"), and reads song titles
  from Spotify and YouTube pages; whether that fits those services' terms is part of the same review. So
  are the AI generators' own terms (Suno's, for example, forbid using its output to train other AI models)
  as they apply to the third-party datasets used here, and a licence for this repository's code, which
  has none yet (`aimd/fakeprint.py` follows Deezer's CC BY-NC, patent-pending code).
* **Per-track licences of the 2,000 Jamendo clips** were not recorded (FMA's were), so how many are
  NonCommercial or NoDerivatives is unknown. MTG-Jamendo publishes them for its 1,000 clips
  (`audio_licenses.txt`); the 1,000 MuseBench / JamendoMaxCaps clips would need looking up track by track.
* **MusicGen prompt groups.** MusicGen clips that share a text prompt can sit on both sides of a split
  (`aimd/sources.py` does not read MusicGen's prompts yet). Fixing it means re-running prepare, train and
  evaluate.
* **An untouched final test.** The main detector and its 7 kHz redesign were chosen on the same test
  rounds that are reported, so confirming its numbers on a fresh held-out generator or human source
  would make them unbiased.
