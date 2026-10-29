"""Model 3: ResNet18 on log-mel spectrograms (the strong in-domain baseline from FakeMusicCaps / MuseBench-style work).

Input: 10 s at 48 kHz mono -> 128-band log-mel, 20 ms hop (501 frames), 20 Hz-7 kHz.
The 7 kHz ceiling is below the lowest native bandwidth in the data (8 kHz, from the
16 kHz AudioLDM2 / Mustango files) and inside the passband of a 16 kHz resample. An
earlier version used 16 kHz (after Deezer, Afchar et al. 2025): files that were
natively 16 or 32 kHz, almost all AI here, then showed their band edge and the model
learned "missing high frequencies = AI" (results/resnet18_16k_before*.csv). With
7 kHz both classes fill the whole input, so that cue is gone. N_FFT is 4096 so the
lowest of the 128 bands still covers at least one FFT bin.

CPU budget (no GPU on the project laptop): ImageNet-initialised (torchvision
IMAGENET1K_V1, first layer summed to one channel), trained on random 3 s crops of at
most 3,000 class-balanced clips per round for 4 epochs, then scored on the full 10 s.
That is far less training than published ResNet18 baselines get, so its numbers
describe this budget only: a longer run may score differently in either direction on
unseen sources.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SR = 48000
N_MELS, N_FFT, HOP, FMAX = 128, 4096, 960, 7000
SETTINGS = f"n_mels={N_MELS} n_fft={N_FFT} fmax={FMAX}"  # recorded next to the cache, the weights and the predictions
CROP = 151  # 3 s random crop during training; full 10 s at test time


def seed_everything(seed: int = 0) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


_mel = None


def logmel(mono: np.ndarray) -> np.ndarray:
    """[n] float32 at 48 kHz -> [128, frames] float16 log-mel in dB, standardised per clip."""
    global _mel
    import torchaudio

    if _mel is None:
        _mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, f_min=20, f_max=FMAX, power=2.0)
    with torch.inference_mode():
        m = _mel(torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32)))
        m = 10 * torch.log10(m + 1e-10)
        m = torch.maximum(m, m.max() - 80)
        m = (m - m.mean()) / (m.std() + 1e-5)
    return m.numpy().astype(np.float16)


def _balanced_subset(X, y, n: int, seed: int):
    """At most n clips, half human and half AI where possible (random, seeded)."""
    y = np.asarray(y)
    if len(y) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = []
    for c in (0, 1):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, size=min(len(pool), n // 2), replace=False))
    idx = np.sort(np.concatenate(idx))
    return X[idx], y[idx]


def resnet18_1ch(pretrained: bool = True) -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18

    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    w = net.conv1.weight.data.sum(dim=1, keepdim=True)
    net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.conv1.weight.data = w
    net.fc = nn.Linear(512, 1)
    return net


class MelResNet:
    """sklearn-style wrapper: fit(X[N,128,T], y, X_val, y_val) / predict_proba(X)."""

    def __init__(self, epochs: int = 4, batch: int = 32, lr: float = 3e-4, seed: int = 0, threads: int = 8, verbose: bool = True,
                 max_train: int = 3000, max_val: int = 600):
        self.epochs, self.batch, self.lr, self.seed, self.verbose = epochs, batch, lr, seed, verbose
        self.max_train, self.max_val = max_train, max_val
        torch.set_num_threads(threads)
        self.net = None

    def _batches(self, X, idx, train: bool):
        for i in range(0, len(idx), self.batch):
            b = idx[i:i + self.batch]
            x = X[b].astype(np.float32)
            if train:
                starts = np.random.randint(0, x.shape[-1] - CROP + 1, size=len(b))
                x = np.stack([xi[:, s:s + CROP] for xi, s in zip(x, starts)])
            yield b, torch.from_numpy(x[:, None])

    def predict_proba(self, X) -> np.ndarray:
        self.net.eval()
        out = np.zeros(len(X), dtype=np.float32)
        with torch.inference_mode():
            for b, x in self._batches(X, np.arange(len(X)), train=False):
                out[b] = torch.sigmoid(self.net(x)[:, 0]).numpy()
        return out

    def fit(self, X, y, X_val=None, y_val=None):
        from .metrics import summarize

        seed_everything(self.seed)
        X, y = _balanced_subset(X, y, self.max_train, self.seed)
        if X_val is not None and len(X_val):
            X_val, y_val = _balanced_subset(X_val, y_val, self.max_val, self.seed + 1)
        self.net = resnet18_1ch()
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * int(np.ceil(len(X) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.15)
        y = np.asarray(y).astype(np.float32)
        pos_w = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        best, best_state = -1.0, None
        for ep in range(self.epochs):
            self.net.train()
            perm = np.random.permutation(len(X))
            tot = 0.0
            for b, x in self._batches(X, perm, train=True):
                opt.zero_grad()
                loss = loss_fn(self.net(x)[:, 0], torch.from_numpy(y[b]))
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss) * len(b)
            if X_val is not None and len(X_val):
                s = summarize(y_val, self.predict_proba(X_val))
                score = s["balanced_accuracy"] if not np.isnan(s["balanced_accuracy"]) else -s["fpr"]
                if self.verbose:
                    print(f"      epoch {ep + 1}/{self.epochs} loss {tot / len(X):.4f} val bal.acc {score:.4f}", flush=True)
                if score >= best:
                    best, best_state = score, copy.deepcopy(self.net.state_dict())
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def save(self, path) -> None:
        torch.save(self.net.state_dict(), path)
        Path(path).with_suffix(".settings.txt").write_text(SETTINGS)  # which spectrogram settings these weights expect

    def load(self, path):
        s = Path(path).with_suffix(".settings.txt")
        if not s.exists() or s.read_text() != SETTINGS:
            raise SystemExit(f"{path} was trained with other spectrogram settings than aimd/cnn.py ({SETTINGS}): "
                             "retrain it (RESEARCH.md: 'Retraining a detector')")
        self.net = resnet18_1ch(pretrained=False)
        self.net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.net.eval()
        return self
