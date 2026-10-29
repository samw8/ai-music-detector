# Leakage and confound report

1. **Disjointness:** all 48 rounds pass (no shared artist, track, held-out generator or held-out human source between training and test).

2a. **Identical audio:** 10 clips share an exact duplicate somewhere in the data; 0 duplicate(s) span the matched train/test split.

    Duplicate groups (first 10): fma_train-85ffa8b61d, fma_train-cac55b28f4; udio_dedup-53e4d6264c, udio_dedup-e6a9bdc417; udio_dedup-e6fb8ee383, udio_dedup-fe242ef5aa; udio_dedup-9138d1a32f, udio_v15-39d86e2bdf; udio_dedup-3239119115, udio_dedup-7cf252c461

2b. **Near-duplicates (CLAP cosine >= 0.98):** 14 of 2541 test clips have a near-copy in training (median nearest-neighbour similarity 0.921).

2c. **Duplicates straddling a split in any round:** 2. unseen_generator/udio_v1: udio_dedup-9138d1a32f, udio_v15-39d86e2bdf; unseen_generator/udio_v1.5: udio_dedup-9138d1a32f, udio_v15-39d86e2bdf. The test-side copies (2 clip-round pairs) are listed in results/excluded_test_clips.csv and left out of scoring by evaluate.py.

    Duplicates carrying different generator labels (the same audio filed under two versions): udio_dedup-9138d1a32f (udio_v1), udio_v15-39d86e2bdf (udio_v1.5). When one of those versions is held out, its copy under the other label stays in training: one track per round, too small to move a result, but it means one of the two labels is wrong.

3. **Shuffled-label control (10 permutations):** balanced accuracy 0.472 ± 0.048, AUC 0.462 ± 0.065; it should centre on 0.5. A model that cannot learn from the labels scores at chance, so the real models' scores come from the labels, not from the split.

4. **Format shortcut probe:** file format alone (sample rate, channels, bitrate, codec) separates human from AI with balanced accuracy 0.753 (AUC 0.850). The models never see these metadata, and every clip is re-decoded to 48 kHz, but traces of the original format (band limits, mono) survive in the audio - which is why the 'no file-format features' ablation exists.

    Format by class:

| label   |   orig_sr |   orig_ch |   kbps |
|:--------|----------:|----------:|-------:|
| human   |     44100 |         2 |    256 |
| AI      |     44100 |         2 |    192 |

| codec     |   AI |   human |
|:----------|-----:|--------:|
| aac       |  300 |       0 |
| mp3       | 3938 |    5193 |
| opus      |  296 |       0 |
| pcm_f32le |  397 |       0 |
| pcm_s16le | 1781 |     360 |

5. **Dataset-identity probe:** the interpretable features identify which of the 24 datasets a test clip came from with accuracy 0.584 (chance = 0.042). Datasets have recognisable recording/format signatures; this is why unseen-human and unseen-generator conditions, not the matched split, are the real test.
