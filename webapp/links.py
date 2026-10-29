"""Turn what someone types (a Spotify / Apple Music / YouTube / Deezer link, or "Artist - Song") into the
song's official 30-second preview clip, the one those services already play publicly.

No full song is ever downloaded: Spotify, Apple Music and YouTube audio is copy-protected and their terms
forbid downloading it. Deezer and Apple publish preview clips through free public APIs (no key needed);
for Spotify and YouTube links only the song's title and artist are read, and the same song's preview is
then looked up on Deezer (or Apple). The preview is analysed in memory and never stored.
"""
from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher

UA = {"User-Agent": "Mozilla/5.0 (compatible; ai-music-detector research demo)"}
TIMEOUT = 15
MAX_PREVIEW_BYTES = 5_000_000


@dataclass
class Match:
    title: str
    artist: str
    album: str
    cover: str          # artwork URL ('' if none)
    preview_url: str
    preview_from: str   # "Deezer" or "Apple"
    page: str           # the song's page on that service
    asked_via: str      # "Spotify link", "YouTube link", "search", ...
    similarity: float   # 1.0 = exact track lookup; lower = closest search result


class LookupError_(Exception):
    """Shown to the user as is."""


def _get(url: str, data_limit: int | None = None) -> tuple[str, bytes]:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        body = r.read(data_limit + 1 if data_limit else -1)
        if data_limit and len(body) > data_limit:
            raise LookupError_("The preview clip is unexpectedly large; try another song.")
        return r.geturl(), body


def _json(url: str) -> dict:
    return json.loads(_get(url)[1])


def _sim(a: str, b: str) -> float:
    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())  # noqa: E731
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


# "(Remastered 2016)", " - Single Version" and the like name the same recording; "(Live)", "(Remix)", covers and
# "(Extended Version)" are different recordings, picked only when the title asked for one.
_SAME = r"remaster|single version|album version|radio edit|original mix|taylor.s version|mono|stereo|deluxe"
_SAME_RECORDING = re.compile(rf"\s*[\(\[][^\)\]]*({_SAME})[^\)\]]*[\)\]]|\s+-\s+[^-]*({_SAME})[^-]*$", re.I)
_OTHER_RECORDING = re.compile(r"\b(live|remix|mix|mashup|cover|acoustic|extended|instrumental|karaoke|demo|sped up|slowed"
                              r"|tribute|originally|commentary|version|edit)\b", re.I)


def _title_sim(asked: str, found: str) -> float:
    asked, found = _SAME_RECORDING.sub("", asked), _SAME_RECORDING.sub("", found)
    s = _sim(asked, found)
    return s - 0.25 if _OTHER_RECORDING.search(found) and not _OTHER_RECORDING.search(asked) else s


def _artist_sim(asked: str, found: str) -> float:
    """1 for the same artist, also when guests are credited ("Daft Punk & Pharrell Williams"); tribute and cover
    acts that merely contain the name ("We Love Taylor Swift") score well below that."""
    parts = [found] + re.split(r"\s*(?:,|&|\bfeat\.?|\bft\.?|\bfeaturing\b|\bwith\b|\bx\b)\s*", found, flags=re.I)
    return 1.0 if max(_sim(asked, p) for p in parts if p) >= 0.9 else 0.7 * _sim(asked, found)


# ------------------------------------------------------------------ exact lookups

def _deezer_track(track_id: str, asked_via: str) -> Match:
    d = _json(f"https://api.deezer.com/track/{track_id}")
    if "error" in d:
        raise LookupError_("Deezer does not know that track.")
    if not d.get("preview"):
        raise LookupError_("Deezer has no preview clip for that track (it may be unavailable in this region).")
    return Match(d["title"], d["artist"]["name"], d.get("album", {}).get("title", ""), d.get("album", {}).get("cover_medium", ""),
                 d["preview"], "Deezer", d.get("link", ""), asked_via, 1.0)


def _apple_track(track_id: str, asked_via: str) -> Match:
    res = _json(f"https://itunes.apple.com/lookup?id={track_id}&entity=song").get("results", [])
    res = [r for r in res if r.get("wrapperType") == "track" and r.get("previewUrl")]
    if not res:
        raise LookupError_("Apple has no preview clip for that song.")
    r = res[0]
    return Match(r["trackName"], r["artistName"], r.get("collectionName", ""), r.get("artworkUrl100", "").replace("100x100", "300x300"),
                 r["previewUrl"], "Apple", r.get("trackViewUrl", ""), asked_via, 1.0)


# ------------------------------------------------------------------ search by artist + title

def search(artist: str, title: str, asked_via: str) -> Match:
    q = f"{artist} {title}".strip()
    best, best_s = None, -1.0
    try:
        for d in _json("https://api.deezer.com/search?" + urllib.parse.urlencode({"q": q, "limit": 10})).get("data", []):
            if not d.get("preview"):
                continue
            s = 0.6 * _title_sim(title, d["title"]) + 0.4 * (_artist_sim(artist, d["artist"]["name"]) if artist else 0.6)
            if s > best_s:
                best_s, best = s, Match(d["title"], d["artist"]["name"], d.get("album", {}).get("title", ""),
                                        d.get("album", {}).get("cover_medium", ""), d["preview"], "Deezer", d.get("link", ""), asked_via, 0)
    except Exception:  # noqa: BLE001 - fall back to Apple
        pass
    if best is None or best_s < 0.55:
        try:
            for r in _json("https://itunes.apple.com/search?" + urllib.parse.urlencode({"term": q, "entity": "song", "limit": 10})).get("results", []):
                if not r.get("previewUrl"):
                    continue
                s = 0.6 * _title_sim(title, r["trackName"]) + 0.4 * (_artist_sim(artist, r["artistName"]) if artist else 0.6)
                if s > best_s:
                    best_s, best = s, Match(r["trackName"], r["artistName"], r.get("collectionName", ""),
                                            r.get("artworkUrl100", "").replace("100x100", "300x300"), r["previewUrl"], "Apple",
                                            r.get("trackViewUrl", ""), asked_via, 0)
        except Exception:  # noqa: BLE001
            pass
    if best is None:
        raise LookupError_(f"Could not find a preview clip for \"{q}\". Try 'Artist - Song', or upload the audio file.")
    best.similarity = round(best_s, 2)
    return best


# ------------------------------------------------------------------ title and artist from Spotify / YouTube

_JUNK = re.compile(r"\s*[\(\[][^\)\]]*(official|video|audio|lyric|visuali[sz]er|hd|4k|remaster|mv|clip|live)[^\)\]]*[\)\]]", re.I)


def _spotify(track_id: str) -> tuple[str, str]:
    _, body = _get(f"https://open.spotify.com/track/{track_id}")
    page = body.decode("utf-8", "replace")
    title = re.search(r'<meta property="og:title" content="([^"]*)"', page)
    desc = re.search(r'<meta property="og:description" content="([^"]*)"', page)  # "Artist · Album · Song · 1997"
    if not title:
        raise LookupError_("Could not read that Spotify link. Try 'Artist - Song' instead.")
    artist = html.unescape(desc.group(1)).split(" · ")[0] if desc else ""
    return artist, html.unescape(title.group(1))


def _youtube(url: str) -> tuple[str, str]:
    d = _json("https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe=""))
    title, author = _JUNK.sub("", d.get("title", "")).strip(), d.get("author_name", "")
    author = re.sub(r"\s*-\s*Topic$|VEVO$|\s*Official$", "", author).strip()
    if " - " in title:
        artist, song = title.split(" - ", 1)
    elif " – " in title:
        artist, song = title.split(" – ", 1)
    else:
        artist, song = author, title
    song = re.sub(r"\s+(ft\.|feat\.|featuring)\s.*$", "", song, flags=re.I).strip()
    return artist.strip(), song


# ------------------------------------------------------------------ entry point

def resolve(text: str) -> Match:
    text = (text or "").strip()
    if not text:
        raise LookupError_("Paste a song link, type 'Artist - Song', or upload an audio file.")
    if re.match(r"https?://", text):
        url = text
        if re.search(r"(link\.deezer\.com|deezer\.page\.link|spotify\.link|apple\.co)/", url):
            url = _get(url, 2_000_000)[0]  # short share links: follow the redirect to the real page
        if m := re.search(r"deezer\.com/(?:[a-z]{2}/)?track/(\d+)", url):
            return _deezer_track(m.group(1), "Deezer link")
        if "music.apple.com" in url or "itunes.apple.com" in url:
            m = re.search(r"[?&]i=(\d+)", url) or re.search(r"/song/[^/]+/(\d+)", url)
            if not m:
                raise LookupError_("That looks like an Apple Music album or playlist link. Share a single song instead.")
            return _apple_track(m.group(1), "Apple Music link")
        if m := re.search(r"open\.spotify\.com/(?:intl-[a-z-]+/)?track/([A-Za-z0-9]+)", url):
            artist, title = _spotify(m.group(1))
            return search(artist, title, "Spotify link")
        if "spotify.com" in url:
            raise LookupError_("That looks like a Spotify album, artist or playlist link. Share a single song instead.")
        if re.search(r"(youtube\.com/(watch|shorts/)|youtu\.be/|music\.youtube\.com/watch)", url):
            artist, title = _youtube(url)
            return search(artist, title, "YouTube link")
        raise LookupError_("Only Spotify, Apple Music, YouTube and Deezer links work. You can also type 'Artist - Song'.")
    for sep in (" - ", " – ", " by "):
        if sep in text:
            a, b = text.split(sep, 1)
            artist, title = (b, a) if sep == " by " else (a, b)
            return search(artist.strip(), title.strip(), "search")
    return search("", text, "search")


def fetch_preview(m: Match) -> bytes:
    """The 30-second preview clip as MP3/AAC bytes (kept in memory only)."""
    return _get(m.preview_url, MAX_PREVIEW_BYTES)[1]
