# Harmony probe: do AI tracks break harmonic conventions?

2400 ten-second clips (100 per dataset). AUC > 0.5: AI scores higher; < 0.5: AI scores lower.

| Feature | Human median | AI median | AUC (AI vs human) | What it measures |
|---|---|---|---|---|
| `harm_nondiatonic_rate` | 0.370 | 0.315 | 0.448 | Fraction of time the estimated chord is outside the estimated key: a qualitative MuseBench analysis claims generated tracks slip in non-diatonic chords, though corpus studies of Suno and Udio do not support it. |
| `harm_chord_change_rate` | 2.300 | 2.300 | 0.475 | Chord changes per second: implausibly jittery or static harmony is a possible sign of generation, though it varies a lot by genre. |
| `harm_key_instability` | 0.500 | 0.500 | 0.535 | Fraction of 2.5-second windows whose best-fitting key differs from the excerpt's key, a proxy for the abrupt key shifts claimed (qualitatively, untested in corpus studies) for generated music. |

Per generator (median non-diatonic rate):

| model             |   harm_nondiatonic_rate |
|:------------------|------------------------:|
| acestep_1.5       |                   0.231 |
| audioldm2         |                   0.495 |
| diffrhythm        |                   0.255 |
| human             |                   0.37  |
| jasco             |                   0.366 |
| lyria_pro3        |                   0.245 |
| lyria_realtime    |                   0.292 |
| magnet            |                   0.5   |
| mureka_V6         |                   0.306 |
| mureka_other      |                   0.204 |
| musicgen          |                   0.417 |
| mustango          |                   0.5   |
| riffusion_fuzz1.1 |                   0.292 |
| riffusion_v1      |                   0.051 |
| sonauto_v2        |                   0.31  |
| stable_audio_open |                   0.199 |
| suno_other        |                   0.356 |
| suno_v3.5         |                   0.255 |
| suno_v4           |                   0.324 |
| suno_v4.5         |                   0.315 |
| suno_v5.5         |                   0.333 |
| udio_v1           |                   0.329 |
| udio_v1.5         |                   0.356 |
| yue               |                   0.306 |