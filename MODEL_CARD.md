# Model card - 3. ResNet18 on log-mel

> **Read this first.** Retrained with FMA left out of its training data, this detector called **14.0% of FMA's pre-2010 tracks** AI-generated; retrained without Jamendo, it called **27.8% of Jamendo tracks** AI-generated. Every one of those is a false accusation of a real musician. Expect rates like these on music unlike its training data (the trained copy, which is kept private and is the one the website uses, was trained on FMA, Jamendo and four of the six GuitarSet players too). Rates are per 10-second excerpt of each source file (a 30-second clip for FMA, the full track for Jamendo, median 3.6 minutes). These rates count a P(AI) of 0.5 or above as an AI call. predict.py and the website say LIKELY AI-GENERATED only at 0.8 or above, LIKELY HUMAN-MADE at 0.2 or below and UNCERTAIN in between; at 0.8 or above it still flagged 3.9% of those FMA tracks and 14.1% of those Jamendo tracks, and put 29.8% and 34.6% of them in UNCERTAIN. In the other conditions: on the matched test split (new artists, otherwise familiar music) 89.3% balanced accuracy, with 8.4% of human tracks called AI; with one AI generator held out, 79.2% balanced accuracy on average (it caught 69.2% of the held-out generator's tracks; worst riffusion_v1, 8.7%); trained on vocal music only, 69.4% on instrumentals, and trained on instrumentals only, 84.7% on vocal music. A score is evidence, not proof.

## What it is

- Binary classifier: P(AI-generated) for a 10-second excerpt at 48 kHz (predict.py averages up to three excerpts per song).
- ResNet18 initialised from torchvision's ImageNet-1k weights (IMAGENET1K_V1), first layer summed to one input channel; input = mono 128-band log-mel, 20 Hz-7 kHz (capped there so a missing top octave cannot separate the classes; see Known failure modes). It has no per-feature explanation: the 'why' lines predict.py prints come from a separate, simpler model (interpretable features + logistic regression), are not this detector's reasons, and can disagree with it.
- Chosen as the main detector because it has the best *worst-case* score across the four evaluation conditions (0.694); see results/results_table.md for every model. That choice, and the redesign that capped its spectrogram at 7 kHz (an earlier version saw up to 16 kHz), were made after looking at the same test rounds reported here. There is no untouched final test set, so this detector's numbers may be slightly optimistic.
- Intended use: research on AI-music detection; triage that a human then checks. **Not** for accusing, demonetising or removing an artist's music.

## Results in all four conditions (never quote one without the others)

| Model | Matched: bal. acc / FPR | Unseen generator: bal. acc (mean) / worst caught | **Unseen human: accuracy on FMA pre-2010** | Unseen human: Jamendo | Vocal→Instr bal. acc | Instr→Vocal bal. acc |
|---|---|---|---|---|---|---|
| Deezer fakeprint + LR (published method, retrained here) | 88.8% / 11.2% | 77.0% / 0.7% (riffusion_v1) | **80.6%** (78.0%-83.2%) | 81.8% | 67.7% (in-domain 89.8%) | 79.7% (in-domain 91.7%) |
| Deezer fakeprint + LR (published method, retrained here) [codec-equalised: all audio via MP3 192k] | 87.9% / 12.2% | 77.2% / 0.7% (riffusion_v1) | **78.6%** (75.7%-81.4%) | 82.0% | 67.4% (in-domain 89.6%) | 79.4% (in-domain 89.8%) |
| 1. Interpretable features + LR | 77.4% / 23.7% | 74.7% / 4.3% (stable_audio_open) | **52.3%** (48.6%-56.0%) | 59.7% | 59.8% (in-domain 77.3%) | 70.3% (in-domain 82.8%) |
| 1. Interpretable features + LR [codec-equalised: all audio via MP3 192k] | 77.4% / 22.6% | 75.2% / 21.3% (stable_audio_open) | **48.3%** (44.6%-52.1%) | 65.7% | 60.5% (in-domain 77.8%) | 68.8% (in-domain 79.0%) |
| 2. Interpretable features + boosted trees | 89.8% / 8.7% | 81.1% / 1.0% (stable_audio_open) | **40.4%** (36.7%-43.9%) | 65.1% | 64.0% (in-domain 88.7%) | 76.4% (in-domain 90.9%) |
| 2. Interpretable features + boosted trees [codec-equalised: all audio via MP3 192k] | 87.5% / 11.2% | 81.8% / 10.0% (stable_audio_open) | **50.4%** (46.6%-54.0%) | 71.9% | 66.7% (in-domain 87.0%) | 74.7% (in-domain 87.7%) |
| 3. ResNet18 on log-mel | 89.3% / 8.4% | 79.2% / 8.7% (riffusion_v1) | **86.0%** (83.4%-88.4%) | 72.2% | 69.4% (in-domain 90.6%) | 84.7% (in-domain 93.2%) |
| 4. CLAP embedding + linear head | 88.0% / 11.4% | 79.7% / 4.3% (riffusion_v1) | **68.7%** (65.2%-72.0%) | 61.3% | 63.8% (in-domain 88.0%) | 77.1% (in-domain 92.5%) |
| Ablation: features + LR, no file-format features | 69.2% / 37.4% | 66.5% / 7.0% (riffusion_v1) | **38.3%** (34.7%-42.0%) | 47.6% | 59.3% (in-domain 68.5%) | 64.5% (in-domain 76.6%) |
| Ablation: features + LR, no file-format features [codec-equalised: all audio via MP3 192k] | 69.5% / 36.4% | 66.6% / 5.7% (riffusion_v1) | **40.1%** (36.4%-43.8%) | 47.0% | 59.1% (in-domain 68.5%) | 64.4% (in-domain 76.0%) |
| Off the shelf: SONICS SpecTTTra-α 5s (not retrained) | 60.3% / 8.1% | 58.6% / 0.3% (audioldm2) | **92.4%** (90.4%-94.1%) | 94.0% | 51.1% (in-domain: n/a, not retrained) | 70.5% (in-domain: n/a, not retrained) |
| Off the shelf: public Deezer-method LR, lofcz (not retrained) | 53.5% / 0.0% | 53.6% / 0.0% (acestep_1.5) | -† | 99.9% | 52.8% (in-domain: n/a, not retrained) | 54.3% (in-domain: n/a, not retrained) |

† The public lofcz model was trained on FMA, so FMA is not unseen for it: its FMA cells are left blank.

The off-the-shelf detectors are scored as published and never retrained, so they have no in-domain vocal/instrumental reference (n/a).

Unseen generator: mean over leave-one-version-out rounds; 'worst caught' is the lowest detection rate in any of them. Whole-family hold-outs are in results/unseen_generator_table.csv. For the off-the-shelf detectors, Suno v3.5 and Udio v1 are left out of that mean because those detectors were trained on them.

Balanced accuracy = average of the share of AI tracks caught and the share of human tracks correctly passed. Unseen human = accuracy on human-only music from a source absent from training (= 1 - false-positive rate).

## How it was tested

- **Matched:** 70% train / 10% validation / 20% test, split by artist (a cross-platform artist key, so the same act on two sites stays on one side) and, for generated music, by uploader account or prompt/caption group where the dataset records one (some do not; see below). The brief asked for 70/20/10; the larger share went to test so the confidence intervals are tighter. Validation is used only to keep the ResNet18's best epoch; every detector calls a clip AI at P(AI) >= 0.5 (no threshold tuning).
- **Unseen generator:** each model version with at least 60 clips is removed from training in turn and tested alone; every family with more than one version is also removed as a whole (audiocraft, lyria, mureka, riffusion, suno, udio). mureka_other, suno_other (under 60 clips) get no round of their own and are held out only with their family.
- **Unseen human:** train on one human source (FMA or Jamendo) and test on the other, plus GuitarSet solo acoustic recordings, which are also left out of training in these rounds.
- **Vocal / instrumental:** train on one, test on both, in both directions.
- **Generator-disjointness applies only to the unseen-generator rounds.** The matched round and both unseen-human rounds train and test on clips from the same generators by design (in each of these 3 rounds, all 23 generators in the test set also appear in training). Both vocal/instrumental cross rounds include test generators that are absent from training because of the data, not by design: 7 in vocal → instrumental (audioldm2, jasco, lyria_realtime, magnet, musicgen, mustango, stable_audio_open) and 2 in instrumental → vocal (diffrhythm, yue); see Known failure modes. Only the unseen-generator rounds hold out generators on purpose.
- Every round re-checks that no artist group, held-out generator or held-out source appears on both sides, and stops with an error if one does. For generated music the artist group is the uploader account, or the text prompt or caption where the dataset's metadata records one. Six AI datasets (diffrhythm, mureka, musicgen, suno_v55, udio_v15, yue; 2,386 clips) have a different group key for every clip (no uploader or prompt is recorded for them), so they are split per track only and clips made from the same prompt can sit on both sides of a round. MusicGen's source (the HAIM dataset) does publish its prompts, but the download step does not read them for MusicGen. 53 of the 81 matched-test MusicGen clips share one of HAIM's 40 prompts with a training clip (matched by file order and md5 in a separate check, not reproduced by this pipeline). MusicGen is 81 of the 1,376 AI clips in the matched test set. 95% intervals come from resampling whole artists; a Wilson interval over artists is used when a set has no errors (or only errors), where resampling would give a zero-width interval. No interval is reported for a test set with fewer than 20 artists (e.g. GuitarSet's 6 players): treat those accuracies as point estimates.
- **Genre** (the original hypothesis): for each genre with at least 30 AI and 30 human test clips, each detector except the ResNet18 is trained with and without it (training sets of equal size) and tested on it; the ResNet18 was skipped in these rounds to save CPU time. A dataset that is almost all one genre (GuitarSet: country/folk) is not scored in that genre's rounds, since removing the genre would remove the dataset too.
- Exact duplicate audio that ended up on both sides of a round is removed from that round's scoring (results/excluded_test_clips.csv); see results/leakage_report.md for the duplicate, near-duplicate, shuffled-label and format-shortcut checks.
- One 10-second excerpt per track, every file through the same decoder to 48 kHz; fixed random seed 0. The ResNet18 was trained on CPU with a budget (3-second crops, at most 3,000 clips per round, 4 epochs), far less than published ResNet18 baselines. No longer run was tested: more training may raise its matched score, but it could also learn more dataset-specific format traces and flag more unseen human music, so do not read these numbers as a lower or upper bound.

## Unseen human music, by source

One condition broken down by test source; read it next to the other three conditions in the table above.

| round                        | test_set                    |    n |   n_artists |   accuracy |   95% CI low |   95% CI high |
|:-----------------------------|:----------------------------|-----:|------------:|-----------:|-------------:|--------------:|
| train_human=fma_test=jamendo | Jamendo (MTG + MuseBench)   | 2000 |        1084 |      0.722 |        0.697 |         0.747 |
| train_human=fma_test=jamendo | GuitarSet (solo acoustic)   |  360 |           6 |      0.994 |      nan     |       nan     |
| train_human=fma_test=jamendo | AI tracks (seen generators) | 1376 |         998 |      0.895 |        0.874 |         0.915 |
| train_human=jamendo_test=fma | FMA uploaded before 2010    |  799 |         610 |      0.860 |        0.834 |         0.884 |
| train_human=jamendo_test=fma | FMA, all                    | 3193 |        2377 |      0.866 |        0.853 |         0.877 |
| train_human=jamendo_test=fma | GuitarSet (solo acoustic)   |  360 |           6 |      0.847 |      nan     |       nan     |
| train_human=jamendo_test=fma | AI tracks (seen generators) | 1376 |         998 |      0.827 |        0.803 |         0.849 |

## Datasets (clip counts cover all splits)

| Dataset | Human/AI | Generator(s) | Clips | Tier | License |
|---|---|---|---|---|---|
| acestep | AI | acestep_1.5 | 400 | B | CC BY-NC 4.0 |
| diffrhythm | AI | diffrhythm | 400 | B | CC BY-NC 4.0 |
| fma_pre2010 | human | human | 799 | B | per track: mostly CC NonCommercial, many NoDerivatives; some FMA 'Old-Time / Historic' 78-rpm records tagged 'Sound Recording Common Law Protection' |
| fma_train | human | human | 2394 | B | per track: mostly CC NonCommercial, many NoDerivatives; some FMA 'Old-Time / Historic' 78-rpm records tagged 'Sound Recording Common Law Protection' |
| guitarset | human | human | 360 | B | CC BY 4.0 |
| lyria3 | AI | lyria_pro3 | 400 | B | CC BY-NC 4.0 |
| mb_audioldm2 | AI | audioldm2 | 300 | B | CC BY-NC-SA 4.0 (model) |
| mb_jamendo | human | human | 1000 | B | CC (per track, Jamendo); collection CC BY-SA 3.0 |
| mb_jasco | AI | jasco | 281 | B | CC BY-NC 4.0 (model) |
| mb_lyria_rt | AI | lyria_realtime | 300 | B | Apache 2.0 (per MuseBench card) |
| mb_magnet | AI | magnet | 300 | B | CC BY-NC 4.0 (model) |
| mb_mustango | AI | mustango | 300 | B | CC BY-NC-SA 4.0 (model) |
| mb_riffusion_v1 | AI | riffusion_v1 | 300 | B | CreativeML OpenRAIL-M (model) |
| mb_stable_audio | AI | stable_audio_open | 300 | B | Stability AI Community License (model) |
| mtg_jamendo | human | human | 1000 | B | CC (per track); dataset CC BY-NC-SA |
| mureka | AI | mureka_V6, mureka_other | 400 | B | unclear (scraped; research only) |
| musicgen | AI | musicgen | 400 | B | CC BY-NC 4.0 |
| riffusion | AI | riffusion_fuzz1.1 | 300 | B | unclear (scraped; research only) |
| sonauto | AI | sonauto_v2 | 296 | B | unclear (scraped; research only) |
| suno_dedup | AI | suno_other, suno_v3.5, suno_v4, suno_v4.5 | 449 | B | unclear (scraped; research only) |
| suno_v55 | AI | suno_v5.5 | 386 | B | CC BY 4.0 (claimed) |
| udio_dedup | AI | udio_v1, udio_v1.5 | 400 | B | unclear (scraped; research only) |
| udio_v15 | AI | udio_v1.5 | 400 | B | CC BY-NC 4.0 |
| yue | AI | yue | 400 | B | CC BY-NC 4.0 |

Clip counts above cover every split (12,265 clips). The matched split puts 8,494 clips in training, 1,230 in validation and 2,541 in test; the trained models (kept private: the public repository has only models/config.json) were trained on the 8,494 training clips only. The trained ResNet18, the copy the website uses, saw a class-balanced random 3,000 of them (1,500 human: 684 fma_train, 269 mb_jamendo, 265 mtg_jamendo, 194 fma_pre2010, 88 guitarset; 1,500 AI from 23 generators); a balanced 600 of the 1,230 validation clips picked its best epoch.

Provenance tiers: A = generated by this project, B = dataset with verified metadata, C = public reporting, D = assumed. Tier D never enters training. No present-day commercial releases were downloaded for training or testing (the public website does fetch store previews of them for each check; see Ethics and legal notes). Exception: FMA contributed 133 clips (98 fma_train, 35 fma_pre2010) tagged 'Public Domain / Sound Recording Common Law Protection' (all in its 'Old-Time / Historic' genre) that are digitised early-20th-century commercial 78-rpm records; **have a lawyer confirm them (or drop them) before publishing.** The named-artist case study uses the researcher's own copies and only scores are kept.

## Known failure modes

Each item below is one condition or stress test; read it next to the four-condition table above.

- **Unseen generators:** with one model version held out it caught 69.2% of that version's tracks on average (worst: riffusion_v1, 8.7%). With a whole family held out it caught as few as 42.0% (all udio versions; every family is in results/unseen_generator_table.csv); that is the better guide to music from a new company or architecture. Fingerprints can differ even between versions of one service: the fakeprint detector of Afchar et al. (ISMIR 2025), trained with Suno v3.5 and Udio-130, caught only 39.8% of tracks from Udio-32, an older Udio model it had not seen.
- **Vocal vs instrumental:** trained on vocal music it scores 69.4% balanced accuracy on instrumentals (vs 90.6% when trained on instrumentals). Much of that gap is a change of generator, not of vocals: 437 of the 723 AI clips in that instrumental test set come from 7 generators that never appear in vocal training (audioldm2, jasco, lyria_realtime, magnet, musicgen, mustango, stable_audio_open). The vocal-trained model catches 44.4% of those against 87.8% of the rest (instrumental-trained: 93.8% and 86.4%). Counting only AI clips from generators present in vocal training, its balanced accuracy is 82.5% instead of 69.4%. Separately, it wrongly calls 22.7% of human instrumentals AI against 9.7% for the instrumental-trained model; generator novelty does not explain that, since it affects AI clips only. So this round mixes two shifts (vocal to instrumental, and seen to unseen generators) and does not isolate the vocal/instrumental effect. The reverse round has the same problem on a smaller scale: trained on instrumentals it scores 84.7% on vocal music (vs 93.2% when trained on vocal music), but 162 of its 653 AI test clips come from 2 generators absent from its training (diffrhythm, yue). It catches 78.4% of those against 82.9% of the rest, and counting only generators present in its training its balanced accuracy is 85.3%.
- **Everyday edits:** worst cases pitch +2 semitones: balanced accuracy 81.5% (missed AI 31.0%, accused humans 6.0%); pitch -2 semitones: balanced accuracy 83.7% (missed AI 26.5%, accused humans 6.0%); tempo 1.1x (pitch kept): balanced accuracy 85.2% (missed AI 23.8%, accused humans 5.8%) vs clean 89.2%.
- **Format shortcut (checked):** almost every clip whose original file was 32 kHz or lower is AI (1278 of 1295: audioldm2, jasco, magnet, musicgen, mustango). An earlier version of this detector, which saw up to 16 kHz, had learned the cue: it wrongly called 79.5% of human clips AI after bandwidth 8 kHz (via 16 kHz); 35.8% of human clips AI after bandwidth 16 kHz (via 32 kHz) (unedited: 8.0%). It was retrained with the spectrogram capped at 7 kHz, below the lowest bandwidth of any file in the data (results/resnet18_16k_before*.csv). This detector sees up to 7 kHz. On the robustness test clips it wrongly called 8.2% of human clips AI after bandwidth 8 kHz (via 16 kHz); 8.2% of human clips AI after bandwidth 16 kHz (via 32 kHz) (unedited: 8.2%). Unseen FMA tracks stored below 44.1 kHz were called AI in 3 of 17 cases. So band-limiting at 8 or 16 kHz, above the 7 kHz this detector sees, no longer changes its verdict; narrower band limits inside its range (phone quality, for example) were not tested. Genuinely old recordings are still harder: it calls 24.4% of the 123 unseen pre-2010 FMA tracks with almost no sound above 8 kHz AI (old or lo-fi recordings, 32 of them tagged as historic 78-rpm records), against 12.1% of the other 676.
- **Genre:** not measured for this detector (it was skipped in the genre rounds to save CPU time); results/genre_table.csv covers the other detectors only.
- **Hybrid tracks** (AI stems mixed with human performance) were not evaluated (out of scope).

## Features the interpretable models use

| Feature | Group | What it measures and why it should differ |
|---|---|---|
| `sofft_peak_ratio_db` | fourier | Height of the tallest spike in the second-order FFT over its median, in dB: evenly spaced spectral peaks left by transposed-convolution upsampling in neural decoders make one tall spike, while human recordings give diffuse energy. |
| `sofft_top_spacing_hz` | fourier | Spacing in Hz of the most periodic spectral-peak pattern; neural decoders repeat at their latent frame rate (e.g. 75 Hz for EnCodec-24k, 86 Hz for DAC-44k), human audio has no preferred spacing. |
| `sofft_comb_db` | fourier | Average height of the second-order FFT at multiples of its tallest spike, in dB over the median: a true upsampling grid repeats at every multiple, an accidental spike does not. |
| `sofft_kurtosis` | fourier | Kurtosis of the second-order FFT: a few sharp spikes (synthetic periodic peaks) give a heavy-tailed shape, natural spectra give a light-tailed one. |
| `fine_peak_rate` | fourier | Narrow peaks per kHz that stick out at least 6 dB from the local spectral trend, a direct count of the tonal 'spikes' that decoder artifacts add to the long-term spectrum. |
| `residual_std_db` | fourier | Roughness of the long-term spectrum around its trend, in dB: decoder artifacts add fine ripple that a recorded acoustic mix averages away. |
| `band_0_150_db` | bands | Share of energy below 150 Hz (dB re total); generators often under- or over-produce sub-bass compared with mastered human mixes. |
| `band_150_500_db` | bands | Share of energy in 150-500 Hz (dB re total), the low-mid body of instruments and voice, included as a spectral-balance reference. |
| `band_500_2k_db` | bands | Share of energy in 500 Hz-2 kHz (dB re total), where vocal fundamentals and most melody live, included as a spectral-balance reference. |
| `band_2k_5k_db` | bands | Share of energy in 2-5 kHz (dB re total), the presence region, included as a spectral-balance reference. |
| `band_5k_8k_db` | bands | Share of energy in 5-8 kHz (dB re total), the band where vocoder limits distort vocal overtones and sibilance in AI vocals. |
| `band_8k_12k_db` | bands | Share of energy in 8-12 kHz (dB re total), the brilliance region, included as a spectral-balance reference. |
| `band_12k_18k_db` | bands | Share of energy in 12-18 kHz (dB re total), where real instruments put little energy and generators leave odd persistent energy. |
| `band_18k_24k_db` | bands | Share of energy from 18 kHz up to the band limit (dB re total; missing when the file stops below ~21 kHz): studio masters often carry air up here, many generators little. |
| `band_5k_8k_var_db` | bands | Frame-to-frame variability of the 5-8 kHz share in dB: real sibilants and breaths come and go, generated vocal bands are often smoother. |
| `band_12k_18k_var_db` | bands | Frame-to-frame variability of the 12-18 kHz share in dB: cymbals and hi-hats make it spiky in human mixes, a constant synthetic noise floor makes it flat. |
| `band_12k_18k_persistence_db` | bands | Gap between the quietest and loudest moments of the 12-18 kHz band (10th minus 90th percentile, dB): a small gap means energy that never goes away, typical of generator noise. |
| `band_5k_8k_flatness` | bands | Spectral flatness inside 5-8 kHz: smeared, noise-like overtones from a vocoder are flatter than the harmonic overtones of a real voice. |
| `band_12k_18k_flatness` | bands | Spectral flatness inside 12-18 kHz, below the band limit only (missing when the file stops below ~15 kHz): synthetic hiss is flat and stationary, real cymbals and air are uneven. |
| `rolloff85_hz` | spectral | Frequency below which 85% of the energy sits (mean over frames), a summary of overall brightness that shifts when a generator renders highs differently. |
| `rolloff99_hz` | spectral | Frequency below which 99% of the energy sits (mean over frames), which exposes the effective bandwidth a generator or codec actually produces. |
| `rolloff99_std_hz` | spectral | Frame-to-frame spread of the 99% rolloff: human mixes vary their bandwidth with the music, a fixed decoder bandwidth barely moves. |
| `flatness_mean` | spectral | Average spectral flatness over 20 Hz-7 kHz (0 = pure tones, 1 = white noise); generated audio often carries a higher diffuse noise floor. |
| `flatness_std` | spectral | Frame-to-frame spread of flatness over 20 Hz-7 kHz: a real performance alternates tonal and noisy moments more than a smoothed generation. |
| `bandlimit_edge_hz` | bandlimit | Frequency of the steepest fall-off above 8 kHz, which reveals a brick-wall low-pass from a neural codec's output sample rate or from MP3/AAC encoding. |
| `bandlimit_drop_db` | bandlimit | How many dB the spectrum falls across that edge within 1 kHz: a cliff points to a digital band limit, a gentle slope to natural roll-off. |
| `above_edge_db` | bandlimit | Energy above the band-limit edge relative to the total (dB): near-silence above a cliff is a codec or decoder signature. |
| `stereo_width_db` | stereo | Side-to-mid energy ratio in dB (-60 = mono): generators often produce narrow or artificially decorrelated stereo images unlike mixed multitrack recordings. |
| `stereo_correlation` | stereo | Correlation between left and right channels (1 = mono): a real stereo mix sits in a typical range set by panning and room sound. |
| `stereo_hf_coherence` | stereo | Average left-right coherence between 4 and 16 kHz: decoders that generate channels jointly or upmix from mono leave unnaturally high or random high-frequency coherence. |
| `stereo_width_hf_minus_lf_db` | stereo | Stereo width above 4 kHz minus width below 500 Hz (dB): mixing engineers keep bass centred and spread highs, a pattern generators reproduce imperfectly. |

The optional harmony features (harm_nondiatonic_rate, harm_chord_change_rate, harm_key_instability) are not used by any trained model; they were tested separately on a sample (results/harmony_probe.md).

## Ethics and legal notes (not legal advice)

- Deezer's fakeprint method is published under CC BY-NC 4.0 and covered by Deezer patent applications: fine for non-commercial research, **ask a lawyer before any commercial or public deployment.**
- The AI-music datasets carry mixed terms (see the Datasets table): CC BY-NC 4.0 (2,981 clips; for 581 of them, the generating model's licence); CC BY-NC-SA 4.0 (600 clips; the generating model's licence); CC BY 4.0 (386 clips; as claimed by the uploader); Apache 2.0 (300 clips; per the MuseBench card); CreativeML OpenRAIL-M (300 clips; the generating model's licence); Stability AI Community License (300 clips; the generating model's licence). A model's licence is not necessarily the licence of the audio it made, and a claimed licence is the uploader's own statement. A further 1,845 AI clips (mureka, riffusion, sonauto, suno_dedup, udio_dedup) come from a scraped collection with no clear licence: research only, do not redistribute the audio.
- FMA human tracks are licensed per track: 2,746 of 3,193 are NonCommercial and 1,462 are NoDerivatives; 133 are historic commercial records (see Datasets). Features, embeddings, fakeprints, scores, the manifest (which also holds scraped platform user IDs) and the trained models are all derived from this audio: **ask a lawyer before releasing any of them publicly.**
- The per-track licences of the 2,000 Jamendo clips (mb_jamendo, mtg_jamendo) were not recorded: the manifest holds only the generic notes 'CC (per track, Jamendo)' / 'CC (per track, Jamendo); collection CC BY-SA 3.0'. So how many of them are NonCommercial or NoDerivatives is unknown. MTG-Jamendo publishes its per-track licences (audio_licenses.txt); they have not yet been joined to the manifest.
- The trained models (kept private; the public repository has only models/config.json) are not cleared for release. resnet18_mel.pt starts from torchvision's ImageNet-pretrained ResNet18 (torchvision notes such weights may carry the terms of their training data; ImageNet's terms limit the images to non-commercial research and education). Whether the audio's licence terms carry over to trained weights is unsettled: **ask a lawyer before publishing the trained weights or using any of them commercially.**
- **The public website** (https://who-gets-called-ai.streamlit.app) went live before this legal review. For each check it fetches the song's official 30-second preview from the Deezer API or Apple's iTunes Search API to a temporary file on the server and deletes it after scoring. Apple's Search API terms say previews may be used only to promote store content, must be streamed and not downloaded, saved or cached, must be shown near a store badge and credited 'provided courtesy of iTunes'. Deezer's API guidelines (developers.deezer.com/guidelines) forbid storing audio locally or offline, and its API Terms of Use (developers.deezer.com/termsofuse, section IV) limit use to non-commercial purposes. The site also reads song titles from public Spotify track pages and from YouTube's oEmbed service (falling back to the watch page), and it shows the artist's name next to a verdict. **A lawyer should look at all of it.**
- AI generators' own terms of service may bind the people who built the third-party AI datasets used here. For example, Suno's terms (effective 3 September 2026) forbid using Outputs to train other AI or machine-learning models and forbid scraping, and free-tier outputs are for personal, non-commercial use. Whether that affects training or publishing a detector on those clips is a question for the legal review.
- A public tool that labels named artists as AI can harm careers and could raise defamation questions: **get legal advice before publishing per-artist verdicts**, use a precision-tuned threshold and offer an appeals path.
- US law has no text-and-data-mining exception; research use of copyrighted audio is a fair-use question (US Copyright Office, Part 3 report, 2025). The EU/UK research exceptions require lawful access.