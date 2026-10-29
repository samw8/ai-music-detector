"""Load the trained detectors (models/) and score raw audio. Used by predict.py and the robustness test."""
from __future__ import annotations

import numpy as np

from . import MODELS as MODEL_DIR
from . import features as F
from .audio import to_pcm16
from .fakeprint import fakeprint
from .models import FORMAT_SENSITIVE, MODELS, ORDER

SR = 48000
HAND = ("features", "features_noformat", "fakeprint")


def hand_reps(clip: np.ndarray, feature_names: list[str], tags=("", "@mp3")) -> dict:
    """One clip's hand-made inputs (feature row + fakeprint, as recorded and through the MP3 encoder).

    A plain function so robustness.py can run it in worker processes; Detectors.represent() calls it too.
    """
    c = to_pcm16(clip)  # idempotent: a clip that is already 16-bit comes back unchanged
    out = {}
    for tag in tags:
        src = c
        if tag == "@mp3":  # the codec-equalised models see audio through the same MP3 encoder they were trained with
            from .augment import mp3

            src = np.clip(mp3(c if c.ndim == 2 else np.stack([c, c]), SR, kbps=192), -1, 1).astype(np.float32)
            # identical to extract_features.codec_equalise (no re-rounding)
        f = F.extract(src, SR)
        out["features" + tag] = [f.get(k, np.nan) for k in feature_names]
        out["fakeprint" + tag] = fakeprint(src, SR)[1]
    return out


class Detectors:
    def __init__(self, names: list[str] | None = None):
        import joblib

        self.models, self.inputs = {}, {}
        if names is None:
            names = list(ORDER) + sorted(p.stem for p in MODEL_DIR.glob("*@*.joblib"))
        for n in names:
            if n == "resnet18_mel":
                p = MODEL_DIR / "resnet18_mel.pt"
                if p.exists():
                    from .cnn import MelResNet

                    self.models[n] = MelResNet().load(p)
                    self.inputs[n] = "mel"
                continue
            p = MODEL_DIR / f"{n}.joblib"
            if p.exists():
                d = joblib.load(p)
                self.models[n] = d["model"]
                self.inputs[n] = d["input"]
                if d.get("feature_names"):
                    self.feature_names = d["feature_names"]
        if not self.models:
            raise FileNotFoundError("No trained models in models/ - run scripts/run_experiments.py first.")
        self.feature_names = getattr(self, "feature_names", F.names())
        self.noformat_names = [c for c in self.feature_names if c not in FORMAT_SENSITIVE]

    def hand_tags(self) -> list[str]:
        """Which variants ('' and/or '@mp3') of the hand-made inputs the loaded models need."""
        need = set(self.inputs.values())
        return [t for t in ("", "@mp3") if need & {h + t for h in HAND}]

    def represent(self, clips: list[np.ndarray], hand: list[dict] | None = None) -> dict:
        """clips: list of [2, n] (or [n]) float32 at 48 kHz, 10 s each.
        hand: hand_reps() of each clip if already computed (robustness.py does that in parallel)."""
        need = set(self.inputs.values())
        reps: dict = {}
        # every training clip was stored as 16-bit PCM; score in the same precision so empty bands read alike
        clips = [to_pcm16(c) for c in clips]
        tags = self.hand_tags()
        if hand is None:
            hand = [hand_reps(c, self.feature_names, tags) for c in clips]
        for tag in tags:
            feats = [h["features" + tag] for h in hand]
            fps = [h["fakeprint" + tag] for h in hand]
            reps["features" + tag] = np.array(feats, dtype=np.float64)
            keep = [self.feature_names.index(k) for k in self.noformat_names]
            reps["features_noformat" + tag] = reps["features" + tag][:, keep]
            reps["fakeprint" + tag] = np.stack(fps).astype(np.float32)
            reps["feature_dicts" + tag] = [dict(zip(self.feature_names, row)) for row in feats]
        mono = [c.mean(axis=0) if c.ndim == 2 else c for c in clips]
        if "clap" in need:
            from .clap import embed_waves

            reps["clap"] = embed_waves(mono)
        if "mel" in need:
            from .cnn import logmel

            reps["mel"] = np.stack([logmel(m) for m in mono])
        return reps

    def predict(self, reps: dict) -> dict[str, np.ndarray]:
        return {n: np.asarray(m.predict_proba(reps[self.inputs[n]])) for n, m in self.models.items()}

    def explain(self, reps: dict, name: str = "feat_lr", top: int = 4) -> list[tuple[str, float, str]]:
        """Which features pushed the SONG (all its excerpts, averaged) toward AI (+) or human (-), for an
        interpretable logistic-regression detector (feat_lr or its @mp3 variant)."""
        m = self.models.get(name)
        tag = "@" + name.split("@")[1] if "@" in name else ""
        dicts = reps.get("feature_dicts" + tag)
        if m is None or not dicts or not hasattr(m, "est") or not hasattr(m.est, "steps"):
            return []
        imp, sc, lr = m.est.steps[0][1], m.est.steps[1][1], m.est.steps[2][1]
        X = np.array([[d.get(k, np.nan) for k in self.feature_names] for d in dicts])
        contrib = sc.transform(imp.transform(X)).mean(axis=0) * lr.coef_[0]  # LR is linear in z, so averaging is exact
        order = np.argsort(-np.abs(contrib))[:top]
        return [(self.feature_names[i], float(contrib[i]), F.doc(self.feature_names[i])) for i in order]
