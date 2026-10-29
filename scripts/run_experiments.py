"""Step 8 - train every model under every evaluation condition and save its test predictions.

    python scripts/run_experiments.py                                   # all models, all conditions
    python scripts/run_experiments.py --models feat_lr,clap_lr          # just some models
    python scripts/run_experiments.py --conditions matched,unseen_human

Resumable: a (model, round) whose predictions already exist is skipped.
Predictions go to results/predictions/<model>/<condition>__<round>.csv, and the
model trained on the "matched" training split is saved to models/ for predict.py.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import MODELS as MODEL_DIR  # noqa: E402
from aimd import REPO, RESULTS  # noqa: E402
from aimd import features as F  # noqa: E402
from aimd.models import FORMAT_SENSITIVE, MODELS, PRETRAINED  # noqa: E402
from aimd.models import ORDER as _ORDER  # noqa: E402

ORDER = [m for m in _ORDER if m not in PRETRAINED]
from aimd.splits import load_table, make_rounds  # noqa: E402

ALL_CONDITIONS = ["matched", "unseen_generator", "unseen_human", "vocal_instr", "genre"]


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)


def load_inputs(df, needed: set, variant: str = ""):
    """Each representation as an array aligned with df's rows (variant 'mp3' = codec-equalised features)."""
    import pandas as pd

    data = {}
    order = df.clip_id.to_numpy()
    sfx = f"_{variant}" if variant else ""
    if needed & {"features", "features_noformat"}:
        feats = pd.read_csv(REPO / "data" / f"features{sfx}.csv").set_index("clip_id")
        cols = [c for c in F.names() if c in feats.columns]
        X = feats.loc[order, cols]
        data["features"] = X.to_numpy(np.float64)
        data["features_noformat"] = X.drop(columns=[c for c in FORMAT_SENSITIVE if c in cols]).to_numpy(np.float64)
        data["feature_names"] = cols
    for name, stem in (("fakeprint", f"fakeprint{sfx}"), ("clap", "emb_clap"), ("mel", "logmel")):
        if name not in needed:
            continue
        arr_path = REPO / "data" / (f"fakeprints{sfx}.npy" if name == "fakeprint" else f"{stem}.npy")
        ids = (REPO / "data" / f"{stem}_ids.txt").read_text().split("\n")
        pos = {c: i for i, c in enumerate(ids)}
        arr = np.load(arr_path, mmap_mode="r")
        if arr.shape[0] != len(ids) or len(set(ids)) != len(ids):
            raise SystemExit(f"data/{arr_path.name} has {arr.shape[0]} rows but {len(ids)} ids: the files are out of step; re-run that step")
        if name == "mel":
            prog = REPO / "data" / "logmel_progress.txt"
            done = int(prog.read_text().strip() or 0) if prog.exists() else 0
            if done < len(ids):
                raise SystemExit(f"data/logmel.npy is incomplete ({done}/{len(ids)} clips): finish `run melspec` first")
            from aimd.cnn import SETTINGS

            st = REPO / "data" / "logmel_settings.txt"
            if not st.exists() or st.read_text() != SETTINGS:
                raise SystemExit(f"data/logmel.npy was built with other spectrogram settings than aimd/cnn.py ({SETTINGS}): re-run "
                                 "`run melspec`, then retrain the ResNet18 (README: 'Retraining a detector')")
        missing = [c for c in order if c not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} clips have no {name} representation yet (e.g. {missing[:2]}): re-run that step")
        idx = np.array([pos[c] for c in order])
        data[name] = np.asarray(arr[idx], dtype=np.float32) if name != "mel" else np.asarray(arr[idx])
        if name == "mel":
            empty = ~data["mel"].reshape(len(idx), -1).any(axis=1)
            if empty.any():
                raise SystemExit(f"{int(empty.sum())} log-mel rows are all zero (unfinished melspec): re-run `run melspec`")
    return data


def main() -> None:
    import joblib
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ORDER))
    ap.add_argument("--conditions", default=",".join(ALL_CONDITIONS))
    ap.add_argument("--variant", choices=["", "mp3"], default="", help="'mp3' = train/test on codec-equalised features")
    a = ap.parse_args()
    models = [m for m in a.models.split(",") if m]
    conds = [c for c in a.conditions.split(",") if c]
    if a.variant:
        models = [m for m in models if MODELS[m][1] in ("features", "features_noformat", "fakeprint")]
    tag = f"@{a.variant}" if a.variant else ""

    df = load_table(REPO)
    rounds = make_rounds(df, which=conds)
    print(f"{len(df)} clips; {len(rounds)} rounds pass the leakage checks: "
          + ", ".join(f"{c}={sum(r.condition == c for r in rounds)}" for c in conds), flush=True)
    data = load_inputs(df, {MODELS[m][1] for m in models}, a.variant)
    y = df.label.to_numpy()
    MODEL_DIR.mkdir(exist_ok=True)

    for m in models:
        make, rep = MODELS[m]
        X = data[rep]
        if rep == "mel":  # finished rounds are skipped, so predictions made with other spectrogram settings must not be reused
            from aimd.cnn import SETTINGS

            pdir = RESULTS / "predictions" / f"{m}{tag}"
            st = pdir / "_settings.txt"
            if st.exists() and st.read_text() != SETTINGS:
                raise SystemExit(f"{pdir} holds predictions made with other spectrogram settings ({st.read_text()}); "
                                 "move that folder aside to retrain (README: 'Retraining a detector')")
            pdir.mkdir(parents=True, exist_ok=True)
            st.write_text(SETTINGS)
        for r in rounds:
            if m == "resnet18_mel" and r.condition == "genre":
                continue  # too slow on CPU for 2 extra trainings per scored genre; the genre question is answered by the feature/embedding models
            out = RESULTS / "predictions" / f"{m}{tag}" / f"{r.condition}__{slug(r.name)}.csv"
            if out.exists():
                continue
            t0 = time.time()
            print(f"[{m}{tag}] {r.condition} / {r.name}: train {len(r.train)}, test {len(r.test)}", flush=True)
            model = make().fit(X[r.train], y[r.train], X[r.val], y[r.val])
            p = model.predict_proba(X[r.test])
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"clip_id": df.clip_id.to_numpy()[r.test], "label": y[r.test], "p_ai": p}).to_csv(out, index=False)
            if r.condition == "matched":
                if m == "resnet18_mel":
                    model.save(MODEL_DIR / f"{m}.pt")
                else:
                    joblib.dump({"model": model, "input": rep + tag, "feature_names": data.get("feature_names")},
                                MODEL_DIR / f"{m}{tag}.joblib")
            print(f"    done in {time.time() - t0:.0f}s", flush=True)
    print("All predictions saved. Next step: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
