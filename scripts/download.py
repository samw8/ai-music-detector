"""Step 1 - download the datasets.

    python scripts/download.py              # show what will be downloaded (GB per dataset), check disk, ask to confirm
    python scripts/download.py --yes        # same, without the confirmation question
    python scripts/download.py --only fma_pre2010,suno_v55
    python scripts/download.py --scale 0.2  # a quick 20% subset to try things out

Safe to stop (Ctrl+C) and re-run: every download resumes where it stopped and
finished datasets are skipped. Everything is saved under %USERPROFILE%\\.aimd\\data.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aimd import DATA  # noqa: E402
from aimd.fetch import free_gb  # noqa: E402
from aimd.sources import SOURCES  # noqa: E402

CLIP_GB_PER_1000 = 1.9  # 10 s stereo 48 kHz FLAC excerpts made in step 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated dataset names (default: all)")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the default track counts")
    ap.add_argument("--yes", action="store_true", help="don't ask for confirmation")
    a = ap.parse_args()

    names = [n for n in a.only.split(",") if n] or list(SOURCES)
    unknown = [n for n in names if n not in SOURCES]
    if unknown:
        sys.exit(f"Unknown dataset(s): {unknown}. Choose from: {', '.join(SOURCES)}")

    print(f"\nDatasets to download into {DATA}\n")
    print(f"{'dataset':14s} {'kind':6s} {'tracks':>6s} {'GB':>6s}  {'status':8s} what it is")
    total = 0.0
    todo = []
    for n in names:
        s = SOURCES[n]
        done = (s.raw / "DONE").exists()
        gb = s.est_gb(a.scale)
        total += 0 if done else gb
        if not done:
            todo.append(s)
        print(f"{n:14s} {'human' if s.label == 0 else 'AI':6s} {int(s.n * a.scale):6d} {gb:6.1f}  {'done' if done else 'to do':8s} {s.note}")
    n_tracks = sum(int(SOURCES[n].n * a.scale) for n in names)
    clips_gb = n_tracks / 1000 * CLIP_GB_PER_1000
    need = total + clips_gb + 5
    have = free_gb(DATA)
    print(f"\nDownload still needed: {total:.1f} GB  (+{clips_gb:.1f} GB for the 10-second excerpts made in step 2)")
    print(f"Free disk space: {have:.1f} GB")
    if have < need:
        sys.exit(f"Not enough disk space: need about {need:.0f} GB including a 5 GB margin. Free some space or use --scale.")
    if not todo:
        print("Everything is already downloaded.")
        return
    if not a.yes and input("\nStart downloading? [y/N] ").strip().lower() not in ("y", "yes"):
        print("Cancelled.")
        return

    failed = []
    for s in todo:
        t0 = time.time()
        print(f"\n--> {s.name}: {s.note}", flush=True)
        try:
            s.fetch(s, a.scale)
            rows = sum(1 for _ in open(s.raw / "meta.jsonl", encoding="utf-8"))
            (s.raw / "DONE").write_text(f"{rows} tracks\n")
            print(f"    {rows} tracks in {(time.time() - t0) / 60:.1f} min", flush=True)
        except KeyboardInterrupt:
            print("\nStopped. Run the same command again to resume.")
            raise
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            failed.append(s.name)
            print(f"    FAILED: {s.name} (re-run to retry; other datasets continue)", flush=True)
    if failed:
        sys.exit(f"\nSome datasets failed: {failed}. Re-run the command to retry them.")
    print("\nAll downloads finished. Next step: python scripts/prepare.py")


if __name__ == "__main__":
    main()
