# Cross-domain AI-music detection

A research pipeline that tells human-made music from AI-generated music and is
judged on the hard question: does it stay honest on music it has never seen?

Detecting AI music under lab conditions is essentially solved (MuseBench, 2026: 99.5-100%
balanced accuracy). This project measures the failures that matter instead:

1. **Unseen human music** - the headline. How often real musicians from a source the
   detector never trained on are wrongly called AI (false-positive rate).
2. **Unseen generators** - every AI model with at least 60 clips is held out in turn (and every multi-version
   family as a whole) and the detector must still catch it.
3. **Vocal vs instrumental** - trained on one kind, tested on the other.
4. **Matched** - the ordinary lab number, reported only next to the three above.

Plus the professor's original hypothesis (does a detector fail on a *genre* it never saw?),
a robustness test against everyday edits, and a forensic "fingerprint" of each generator's decoder.

**Results:** `results/results_table.md` (and `results/figures/`). **Limitations and false-positive
rates:** `MODEL_CARD.md`. **Proof the splits are clean:** `results/leakage_report.md`.

---

## Checking a song (no programming needed)

Open a Command Prompt in this folder (in File Explorer, click the address bar, type `cmd`, press Enter) and type:

```
run check "C:\path\to\song.mp3"
```

You get the probability that the song is AI-generated from the main detector (a ResNet18 network
that looks at the song's spectrogram) and a caveat with that detector's measured false-positive
rate on music unlike its training data. Below that is a short list of measurable audio features
from a separate, simpler model (interpretable features + logistic regression). The main detector
cannot explain its own score, so that list is **not** its reasons, and the simpler model's score
(printed beside it) can disagree. Point it at a folder to check every song inside.

(The first version of the main detector mistook missing high frequencies for AI. It was retrained
with its spectrogram capped at 7 kHz, and band-limiting a modern recording to 8 kHz no longer changes
its verdict. Genuinely old or lo-fi recordings (almost no sound above 8 kHz) are still wrongly called
AI about twice as often as other human music, and phone-quality audio (below 8 kHz) was not tested;
see `MODEL_CARD.md`. If a future retrain brings the shortcut back, `run evaluate` detects it and the
checker warns on low-sample-rate, low-bitrate or band-limited files again.)

**A score is evidence, not proof. Never use it alone to accuse an artist.**

---

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
| 10 | `run robustness` | Re-scores test clips after MP3/OGG/AAC, pitch +/-2, speed 0.9x/1.1x, loudness -14 LUFS, plus two band-limit edits that imitate old or low-sample-rate recordings. | ~3 h |
| 11 | `run fingerprints` | Per-generator decoder fingerprints (the "reverse engineering" figure). | 10 min |
| 12 | `run harmony` | Optional: tests the chord/key features on 100 clips per dataset. | ~30 min |
| 13 | `run evaluate` | Builds the results table (CSV + Markdown), figures and confusion matrices; picks the main detector by its worst condition. Any model with unfinished rounds is marked "partial" and cannot be picked. | 5 min |
| 14 | `run modelcard` | Rewrites `MODEL_CARD.md` from the current results. | 1 min |
| 15 | `run report` | Writes `results\report.html`, a plain-language summary page with every figure. | 1 min |
| 16 | `run demo` | Case study on named artists, using your own copies placed in `demo\audio\<artist>\`. Use only the releases listed in the `which_releases` column of `demo\artists.csv`; for the two "track-level only" rows (Josh Fawaz, mikeeysmind) only the named track counts as AI. | depends |

`run all` does steps 1-15 in order. `run test` checks that the leakage guards still catch leaks.

**Retraining a detector.** Re-running a step never retrains a finished model: finished rounds and
finished robustness rows are skipped. To retrain one (for example after changing `aimd/cnn.py`), move
`results\predictions\<model>\` aside, delete that detector's rows from `results\robustness.csv`, then
run `run melspec` (ResNet18 only), `run train`, `run robustness` and `run evaluate`. The ResNet18's
cache, weights and predictions each record their spectrogram settings, and the program stops with an
error if they do not match `aimd/cnn.py`.

**Keep the laptop plugged in with the lid open** while long steps run: closing the lid puts Windows to sleep and pauses
everything (each step resumes where it stopped, but nothing happens while it sleeps).

Maintenance scripts (only needed when the rules change): `scripts\refresh_meta.py` re-applies the
label/grouping rules to downloaded metadata without re-downloading; `scripts\check_mono.py` removes
clips whose mono mix is silent.

---

## What is in here

| Path | Contents |
|---|---|
| `aimd/sources.py` | Every dataset: source, licence, provenance tier, how a subset is fetched |
| `aimd/features.py` | The interpretable features, each with a one-sentence explanation |
| `aimd/fakeprint.py` | Deezer's Fourier fakeprint (ISMIR 2025 best paper), re-implemented |
| `aimd/splits.py` | The evaluation conditions and the leakage checks that stop the program |
| `aimd/models.py`, `aimd/cnn.py` | The detectors |
| `predict.py` | Check one song or a folder |
| `demo/artists.csv` | 21 publicly reported AI acts, 2 single AI-assisted tracks by otherwise human artists (track-level only; one is tier D, assumed), and 30 genre-matched human artists, with evidence and sources. It names real people, so it is kept out of the public repository until it has been reviewed. |
| `results/` | Tables, figures and reports (per-clip predictions are rebuilt by `run train` and not published) |
| `MODEL_CARD.md` | Training data, failure modes, the out-of-domain false-positive rate |

## Data and the rules that keep it honest

* **No present-day commercial releases are downloaded, and no audio is redistributed.** (FMA does
  include 133 early-20th-century commercial 78-rpm records that it tags "Sound Recording Common Law
  Protection", and 1,845 AI clips come from a scraped collection with no clear licence: see the legal
  notes in `MODEL_CARD.md`.) Training uses public research
  datasets (FMA and Jamendo for human music, plus GuitarSet solo acoustic guitar: 4 of its 6 players are in the
  matched training split that the models in `models\` come from; Suno, Udio, Mureka, Riffusion, Sonauto, Lyria,
  ACE-Step, YuE, DiffRhythm, MusicGen and the MuseBench open-model outputs for AI). The named-artist
  case study runs only on files you supply, and only scores are kept.
* **Provenance tier per track** (A generated by us, B dataset with verified metadata, C public
  reporting, D assumed). Tier D never enters training.
* **Artist-disjoint splits everywhere; generator-disjoint and source-disjoint where the
  condition requires it.** The program stops with an error if a split leaks.
* **Do not publish the audio.** Before publishing anything else from this folder, have a lawyer or the
  university's copyright office look at it: the trained models in `models\`, `data\features*.csv`,
  `data\emb_clap.npy`, `data\fakeprints*.npy` and `data\manifest.csv` are all derived from audio under
  NonCommercial, NoDerivatives, ShareAlike and "unclear (scraped; research only)" terms, and the manifest
  holds Suno/Udio/Riffusion/Sonauto user IDs (drop or hash the `artist_id` column before any release).
  Until then, share these files only with research collaborators.

## Things that need a human decision (not done automatically)

* **Tier-A test tracks (generated by this project)** - not done. Suno v6 needs a paid plan, and Udio
  has disabled downloads on every plan since late October 2025, so a Udio subscription would not produce files.
* **Commercial detector APIs** (Ircam Amplify, Resemble AI) need paid keys - not called.
* **Legal review** before publishing the paper, the trained models, the derived data files or
  `demo\artists.csv`, before publishing per-artist verdicts, and before deploying a public tool - see `MODEL_CARD.md`.
