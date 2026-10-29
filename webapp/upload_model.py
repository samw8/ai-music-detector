"""Upload the trained main detector to a PRIVATE Hugging Face model repository, where the online page
(webapp/streamlit_app.py on Streamlit Community Cloud) downloads it at start-up.

    hf auth login                      # once, with a Hugging Face access token (Write)
    python webapp/upload_model.py      # again after every retrain of the ResNet18, then reboot the online app

The weights are not cleared for public release (MODEL_CARD.md, legal notes), so the repository stays private:
visitors can use the detector but not download it. This script never handles tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    api = HfApi()
    repo = f"{api.whoami()['name']}/ai-music-detector-model"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    for name in ("resnet18_mel.pt", "resnet18_mel.settings.txt"):
        api.upload_file(path_or_fileobj=str(ROOT / "models" / name), path_in_repo=name, repo_id=repo,
                        commit_message=f"Upload {name}")
    print(f"private model repository: https://huggingface.co/{repo}")
    print("If that is not norows/ai-music-detector-model, add the secret MODEL_REPO = \"" + repo + "\" to the online app.")


if __name__ == "__main__":
    sys.exit(main())
