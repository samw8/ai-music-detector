"""Model 4's front end: CLAP (laion/clap-htsat-unfused) audio embeddings, 48 kHz input, 512 dimensions.

laion/larger_clap_music was tried first, but under transformers 5.x it returns near-identical embeddings for
every clip (mean cosine 0.97 between unrelated songs; zero-shot labels all the same), so it was replaced.
"""
from __future__ import annotations

import numpy as np

MODEL = "laion/clap-htsat-unfused"
_model = None


def load_model():
    global _model
    if _model is None:
        import torch
        from transformers import ClapModel, ClapProcessor

        torch.set_num_threads(8)
        _model = (ClapModel.from_pretrained(MODEL).eval(), ClapProcessor.from_pretrained(MODEL))
    return _model


def embed_waves(waves: list[np.ndarray], sr: int = 48000, batch: int = 8) -> np.ndarray:
    """L2-normalised CLAP audio embeddings for mono 48 kHz signals."""
    import torch

    model, proc = load_model()
    out = []
    for i in range(0, len(waves), batch):
        inp = proc(audio=list(waves[i:i + batch]), sampling_rate=sr, return_tensors="pt")
        with torch.inference_mode():
            e = model.get_audio_features(**inp)
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        out.append(torch.nn.functional.normalize(e, dim=-1).numpy().astype(np.float32))
    return np.concatenate(out)
