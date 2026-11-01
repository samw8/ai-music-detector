# Who Gets Called AI

A detector that estimates whether a song was made by an AI music generator, and a study of how often
detectors like it wrongly flag real musicians.

**Try it: https://who-gets-called-ai.streamlit.app**

Paste a Spotify, Apple Music, YouTube or Deezer link, type "Artist - Song", or upload an audio file.
For a link, the site finds the song's official 30-second preview, because full songs on those services
are copy-protected. You get a score from 0 (human) to 1 (AI). At 0.2 or below it says "likely
human-made", at 0.8 or above "likely AI-generated", and anything in between "uncertain". A score is
evidence, not proof that an artist used AI.

The site goes to sleep when nobody has used it for 12 hours. If you see a "Zzzz" page, click the button
to wake it; it is ready again in about a minute.

## Why

Tools like Suno and Udio now make songs that are hard to tell apart from human recordings. Published
detectors score close to 100% when they are tested on the same kind of music they were trained on.
They do much worse on music they haven't seen, like a new AI generator or human music from a different
source. When a detector fails on human music, a real musician gets accused of faking it. This project
measures those failures.

## How it works

1. **Data.** 12,265 ten-second clips: 5,553 human (from the FMA and Jamendo free-music archives and
   the GuitarSet recordings) and 6,712 from 23 versions of AI generators, including Suno, Udio, Mureka,
   Riffusion, Lyria and MusicGen. Every clip goes through the same decoder at 48 kHz. Traces of the
   original file format still survive in the audio (see the leakage report), which is why the
   website's detector ignores everything above 7 kHz and was tested against re-encoding.
2. **Detectors.** Five were trained and compared (plus an ablation without file-format features): a set of explainable audio measurements with
   logistic regression, the same measurements with boosted trees, a ResNet18 image network that reads
   the clip's spectrogram, pretrained CLAP audio embeddings with a linear classifier, and Deezer's
   published "fakeprint" method retrained on this data. Two published, already-trained detectors were
   also scored on the same clips for comparison.
3. **Tests.** No artist appears in both training and testing. Each detector was tested four ways:
   * **Matched:** new artists, but the same kinds of music it trained on.
   * **Unseen human music:** a whole human music source left out of training.
   * **Unseen generator:** one AI generator, or a whole family such as every Udio version, left out of training.
   * **Vocal vs instrumental:** trained on one, tested on the other.
4. **Pick.** The website uses the detector with the best worst-case result across those tests: the
   ResNet18. Its spectrogram stops at 7 kHz, because an earlier version learned to treat missing high
   frequencies as a sign of AI. That choice, and the 7 kHz redesign, were made with the same test
   results reported below, so its numbers may be slightly optimistic.

## Results for the detector on the website

| Test | Result |
|---|---|
| Matched | 89% balanced accuracy; 8% of human tracks wrongly called AI |
| Unseen human music | FMA left out: 14% of its tracks uploaded before 2010 wrongly called AI. Jamendo left out: 28% |
| Unseen generator | Caught 69% of the left-out generator's tracks on average (79% balanced accuracy). Worst: Riffusion v1, 9% caught. With every Udio version left out: 42% caught |
| Vocal vs instrumental | Trained on vocal music: 69% balanced accuracy on instrumentals, with 23% of human instrumentals wrongly called AI. Much of that drop is a change of generator, not of vocals: 60% of the AI instrumentals in that test come from generators missing from vocal training, and counting only the others it scores 82.5%. Trained on instrumentals: 85% on vocal music |

These are per 10-second clip, counting a score of 0.5 or more as an AI call. The website only says
"likely AI-generated" at 0.8 or more; at that cut-off, 4% (FMA) and 14% (Jamendo) of the unseen human
tracks were still flagged. Old or lo-fi human recordings are flagged more often: 24% of the 123 FMA
tracks from before 2010 with almost no sound above 8 kHz, against 12% of the other 676.

All detectors and conditions: [`results/results_table.md`](results/results_table.md). Where it fails and
why: [`MODEL_CARD.md`](MODEL_CARD.md). Checks for test songs leaking into training (exact copies
removed; 14 near-copies reported and left in): [`results/leakage_report.md`](results/leakage_report.md). The genre test, robustness to edits such as
MP3 compression and pitch shifts, and per-generator fingerprints are in `results/`.

## What's in this repository

| Path | Contents |
|---|---|
| `webapp/` | The website |
| `aimd/`, `scripts/` | Data preparation, the detectors and the experiments |
| `predict.py`, `Check songs.bat` | Checking songs on your own computer |
| `results/` | Tables and figures |
| `MODEL_CARD.md` | What the detector was trained on and where it fails |
| [`RESEARCH.md`](RESEARCH.md) | How to re-run everything, check songs locally, and redeploy the website |

No audio is stored here. The trained detector is kept private until a lawyer has reviewed its
licences, because some of the training audio is under non-commercial or no-derivatives terms.
Details are in `RESEARCH.md` and `MODEL_CARD.md`.
