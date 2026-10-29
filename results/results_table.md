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
| Off the shelf: SONICS SpecTTTra-α 5s (not retrained) | 60.3% / 8.1% | 58.6% / 0.3% (audioldm2) | **92.4%** (90.4%-94.1%) | 94.0% | 51.1% (in-domain 51.1%) | 70.5% (in-domain 70.5%) |
| Off the shelf: public Deezer-method LR, lofcz (not retrained) | 53.5% / 0.0% | 53.6% / 0.0% (acestep_1.5) | **-**† (---) | 99.9% | 52.8% (in-domain 52.8%) | 54.3% (in-domain 54.3%) |

† The public lofcz model was trained on FMA, so FMA is not unseen for it: its FMA cells are left blank.

Unseen generator: mean over leave-one-version-out rounds; 'worst caught' is the lowest detection rate in any of them. Whole-family hold-outs are in results/unseen_generator_table.csv. For the off-the-shelf detectors, Suno v3.5 and Udio v1 are left out of that mean because those detectors were trained on them.