"""aimd: AI-generated music detection toolkit."""
import os
from pathlib import Path


def _winlong(p: Path) -> Path:
    """Windows without long-path support fails past 260 characters; the \\\\?\\ prefix lifts that limit."""
    s = str(p)
    if os.name == "nt" and len(s) > 120 and not s.startswith("\\\\?\\"):
        return Path("\\\\?\\" + s)
    return p


# Large files (datasets, model downloads) live outside the repo, in a short path:
# Windows without long-path support breaks on deep paths, and the Microsoft Store
# Python / packaged apps redirect writes under AppData.
HOME = Path(os.environ.get("AIMD_HOME", Path.home() / ".aimd"))
DATA = HOME / "data"
REPO = Path(__file__).resolve().parent.parent
MODELS = _winlong(REPO / "models")
RESULTS = _winlong(REPO / "results")

os.environ.setdefault("HF_HOME", str(HOME / "hf"))
