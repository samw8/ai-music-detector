"""Resumable download helpers.

Three ways of getting data, all of which survive being interrupted and re-run:

* ``download_file``  - whole file over HTTP with Range-resume (``.part`` file).
* ``RemoteZip``      - pull single members out of a huge remote .zip without
                       downloading the rest (reads the zip index with Range requests).
* ``stream_tar``     - read the first N samples of a huge remote .tar shard and stop;
                       the byte offset reached is saved so a re-run continues there.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests

UA = {"User-Agent": "aimd-research/0.1 (academic AI-music detection study)"}
_session = requests.Session()
_session.headers.update(UA)
if os.environ.get("HF_TOKEN"):
    _hf_auth = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
else:
    _hf_auth = {}


def _headers(url: str, extra: dict | None = None) -> dict:
    h = dict(extra or {})
    if "huggingface.co" in url:
        h.update(_hf_auth)
    return h


def _get(url: str, headers: dict | None = None, stream: bool = False, tries: int = 6):
    for attempt in range(tries):
        try:
            r = _session.get(url, headers=_headers(url, headers), stream=stream, timeout=60, allow_redirects=True)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            if attempt == tries - 1:
                raise
            time.sleep(2 * (attempt + 1))
            last = e  # noqa: F841


def remote_size(url: str) -> int:
    r = _get(url, headers={"Range": "bytes=0-0"}, stream=True)
    r.close()
    cr = r.headers.get("Content-Range")
    if cr:
        return int(cr.split("/")[-1])
    return int(r.headers.get("Content-Length", 0))


def download_file(url: str, dest: Path, expected_size: int | None = None, chunk: int = 1 << 20) -> Path:
    """Download ``url`` to ``dest``, resuming a previous partial download if one exists."""
    dest = Path(dest)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(6):
        have = part.stat().st_size if part.exists() else 0
        if expected_size and have >= expected_size:
            break
        try:
            r = _get(url, headers={"Range": f"bytes={have}-"} if have else None, stream=True)
            mode = "ab" if (have and r.status_code == 206) else "wb"
            with open(part, mode) as f:
                for block in r.iter_content(chunk):
                    f.write(block)
            break
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to download {url}")
    if expected_size and part.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {url}: {part.stat().st_size} != {expected_size}")
    os.replace(part, dest)
    return dest


class _HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.size = remote_size(url)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        data = _get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}).content
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


class RemoteZip:
    """Random access to members of a remote zip via HTTP Range requests."""

    def __init__(self, url: str, index_cache: Path | None = None):
        self.url = url
        self._raw = _HTTPRangeFile(url)
        self.zf = zipfile.ZipFile(io.BufferedReader(self._raw, buffer_size=1 << 20))
        if index_cache:
            index_cache.parent.mkdir(parents=True, exist_ok=True)
            index_cache.write_text("\n".join(self.zf.namelist()), encoding="utf-8")

    def names(self) -> list[str]:
        return self.zf.namelist()

    def info(self, name: str) -> zipfile.ZipInfo:
        return self.zf.getinfo(name)

    def extract(self, name: str, dest: Path) -> Path:
        dest = Path(dest)
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        for attempt in range(5):
            try:
                with self.zf.open(name) as src, open(part, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                break
            except (requests.ConnectionError, requests.Timeout, zipfile.BadZipFile, EOFError):
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"failed to extract {name} from {self.url}")
        os.replace(part, dest)
        return dest


def tar_find_offset(url: str, prefix: str, start: int = 0, max_steps: int = 20000) -> int:
    """Walk a remote tar's headers with tiny Range requests (no file data) to find the first member under ``prefix``."""
    off, longname = start, None
    for _ in range(max_steps):
        h = _get(url, headers={"Range": f"bytes={off}-{off + 511}"}).content
        if len(h) < 512 or h[:1] == b"\0":
            break
        size = int(h[124:136].rstrip(b"\0 ") or b"0", 8)
        typ = h[156:157]
        body_len = ((size + 511) // 512) * 512
        if typ == b"L":  # GNU long name: the body holds the real path of the next member
            longname = _get(url, headers={"Range": f"bytes={off + 512}-{off + 512 + size - 1}"}).content.rstrip(b"\0").decode("utf-8", "replace")
            hdr_start = off
            off += 512 + body_len
            continue
        name = longname or h[:100].rstrip(b"\0").decode("utf-8", "replace")
        member_start = hdr_start if longname else off
        longname = None
        if name.startswith(prefix):
            return member_start
        off += 512 + body_len
    raise RuntimeError(f"{prefix} not found in {url} within {max_steps} members")


def stream_tar(url: str, out_dir: Path, max_samples: int, keep_ext: set[str], state_file: Path,
               accept=None, max_seen: int | None = None, prefix: str | None = None, start_offset: int = 0,
               hash_names: bool = False) -> list[dict]:
    """Extract up to ``max_samples`` WebDataset samples (same-stem files) from a remote tar.

    ``accept(meta_json)`` can reject samples (e.g. wrong model version); rejected
    samples don't count, but we stop after ``max_seen`` samples either way. With
    ``prefix`` only members under that folder are kept, and reading stops once the
    folder has been passed. Progress (byte offset + kept stems) is saved after every
    sample, so an interrupted run resumes from where it stopped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_file.read_text()) if state_file.exists() else {"offset": start_offset, "kept": [], "seen": 0, "done": False}
    if state["done"] or len(state["kept"]) >= max_samples or (max_seen and state["seen"] >= max_seen):
        return state["kept"]
    start = state["offset"]
    if prefix is not None:
        return _stream_folder(url, out_dir, max_samples, keep_ext, state_file, state, prefix, hash_names)
    r = _get(url, headers={"Range": f"bytes={start}-"} if start else None, stream=True)
    raw = r.raw
    raw.decode_content = True

    def save():
        state_file.write_text(json.dumps(state), encoding="utf-8")

    pending: dict[str, dict] = {}
    cur_stem, cur_offset = None, start
    try:
        with tarfile.open(fileobj=raw, mode="r|") as tf:
            for m in tf:
                if not m.isfile():
                    continue
                stem, ext = os.path.splitext(os.path.basename(m.name))
                if stem != cur_stem:
                    if cur_stem is not None:
                        _finish_sample(cur_stem, pending.pop(cur_stem, {}), out_dir, keep_ext, accept, state)
                        state["offset"] = start + m.offset
                        save()
                        if len(state["kept"]) >= max_samples or (max_seen and state["seen"] >= max_seen):
                            return state["kept"]
                    cur_stem, cur_offset = stem, start + m.offset
                data = tf.extractfile(m).read()
                pending.setdefault(stem, {})[ext.lower()] = data
            if cur_stem is not None:
                _finish_sample(cur_stem, pending.pop(cur_stem, {}), out_dir, keep_ext, accept, state)
            state["done"] = True
            save()
    finally:
        r.close()
    return state["kept"]


def _stream_folder(url, out_dir, max_samples, keep_ext, state_file, state, prefix, hash_names) -> list[dict]:
    """Plain-tar variant of stream_tar: one file per member, kept only under ``prefix``."""
    import hashlib

    start = state["offset"]
    r = _get(url, headers={"Range": f"bytes={start}-"} if start else None, stream=True)
    raw = r.raw
    raw.decode_content = True
    entered = False
    try:
        with tarfile.open(fileobj=raw, mode="r|") as tf:
            for m in tf:
                state["offset"] = start + m.offset
                if not m.isfile():
                    continue
                inside = m.name.startswith(prefix)
                if entered and not inside:
                    break
                ext = os.path.splitext(m.name)[1].lower()
                if not inside or ext not in keep_ext:
                    continue
                entered = True
                data = tf.extractfile(m).read()
                stem = hashlib.md5(m.name.encode()).hexdigest()[:16] if hash_names else os.path.splitext(os.path.basename(m.name))[0]
                (out_dir / f"{stem}{ext}").write_bytes(data)
                state["kept"].append({"stem": stem, "file": f"{stem}{ext}", "member": m.name})
                state["seen"] += 1
                state["offset"] = start + m.offset_data + ((m.size + 511) // 512) * 512
                state_file.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
                if len(state["kept"]) >= max_samples:
                    return state["kept"]
        state["done"] = True
    finally:
        r.close()
        state_file.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    state["done"] = True
    state_file.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return state["kept"]


def _finish_sample(stem: str, files: dict, out_dir: Path, keep_ext: set[str], accept, state: dict) -> None:
    state["seen"] += 1
    meta = {}
    if ".json" in files:
        try:
            meta = json.loads(files[".json"].decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            meta = {}
    audio = [(e, b) for e, b in files.items() if e in keep_ext]
    if not audio or (accept is not None and not accept(meta)):
        return
    ext, blob = audio[0]
    path = out_dir / f"{stem}{ext}"
    path.write_bytes(blob)
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    state["kept"].append({"stem": stem, "file": path.name})


def free_gb(path: Path) -> float:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9
