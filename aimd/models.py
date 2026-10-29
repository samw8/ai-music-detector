"""The detectors, trained in the order the brief asks for.

    fakeprint_lr     Deezer's published method (logistic regression on the Fourier fakeprint) - the baseline to beat
    feat_lr          1. logistic regression on the interpretable features
    feat_gbt         2. gradient-boosted trees on the same features
    resnet18_mel     3. ResNet18 on log-mel spectrograms
    clap_lr          4. pretrained audio embedding (CLAP) + linear head
    feat_lr_noformat ablation: feat_lr without features that mostly describe the file format

All random seeds are fixed. Each model is a factory returning an object with
``fit(X, y, X_val, y_val)`` and ``predict_proba(X) -> P(AI)``; ``INPUT`` says which
representation it reads (features / fakeprint / clap / mel).
"""
from __future__ import annotations

import numpy as np

from . import features as F

SEED = 0


class SkModel:
    def __init__(self, est):
        self.est = est

    def fit(self, X, y, X_val=None, y_val=None):
        self.est.fit(X, np.asarray(y).astype(int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.est.predict_proba(X)[:, 1]


def _feat_lr():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return SkModel(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                 LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED)))


def _feat_gbt():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return SkModel(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
                                                  class_weight="balanced", random_state=SEED))


def _fakeprint_lr():
    from sklearn.linear_model import LogisticRegression

    # exactly Deezer's train_test_regressor.py settings (defaults + balanced classes), more iterations to converge
    return SkModel(LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))


def _clap_lr():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return SkModel(make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=SEED)))


def _resnet():
    from .cnn import MelResNet

    return MelResNet(epochs=4, seed=SEED)


# Everything whose value (or whether it exists at all) depends on the file's bandwidth or channel layout.
# Bands above 8 kHz are missing for 16 kHz-native files and partly missing for 32 kHz-native ones, so they
# encode the format too; the ablation drops them to see what is left when the format is invisible.
FORMAT_SENSITIVE = [n for n, (_, g) in F.REGISTRY.items() if g in F.FORMAT_GROUPS] + [
    "band_18k_24k_db", "rolloff99_hz", "rolloff99_std_hz", "band_8k_12k_db", "band_12k_18k_db",
    "band_12k_18k_var_db", "band_12k_18k_persistence_db", "band_12k_18k_flatness"]

MODELS = {
    "fakeprint_lr": (_fakeprint_lr, "fakeprint"),
    "feat_lr": (_feat_lr, "features"),
    "feat_gbt": (_feat_gbt, "features"),
    "feat_lr_noformat": (_feat_lr, "features_noformat"),
    "clap_lr": (_clap_lr, "clap"),
    "resnet18_mel": (_resnet, "mel"),
}
ORDER = ["fakeprint_lr", "feat_lr", "feat_gbt", "resnet18_mel", "clap_lr", "feat_lr_noformat", "spectttra_alpha5s", "lofcz_fakeprint"]
PRETRAINED = {"spectttra_alpha5s", "lofcz_fakeprint"}  # scored off the shelf by scripts/baseline_pretrained.py, never retrained
LABELS = {
    "spectttra_alpha5s": "Off the shelf: SONICS SpecTTTra-α 5s (not retrained)",
    "lofcz_fakeprint": "Off the shelf: public Deezer-method LR, lofcz (not retrained)",
    "fakeprint_lr": "Deezer fakeprint + LR (published method, retrained here)",
    "feat_lr": "1. Interpretable features + LR",
    "feat_gbt": "2. Interpretable features + boosted trees",
    "resnet18_mel": "3. ResNet18 on log-mel",
    "clap_lr": "4. CLAP embedding + linear head",
    "feat_lr_noformat": "Ablation: features + LR, no file-format features",
}
VARIANTS = {"mp3": "codec-equalised: all audio via MP3 192k"}


def label(name: str) -> str:
    """Display name, including variants such as 'feat_lr@mp3'."""
    base, _, var = name.partition("@")
    return LABELS.get(base, base) + (f" [{VARIANTS.get(var, var)}]" if var else "")


def ordered(names) -> list[str]:
    """Models in the brief's order, each followed by its variants."""
    names = set(names)
    out = []
    for base in ORDER:
        out += [n for n in sorted(names) if n == base or n.startswith(base + "@")]
    return out + sorted(names - set(out))
