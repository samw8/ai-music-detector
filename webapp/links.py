"""Turn what someone types (a Spotify / Apple Music / YouTube / Deezer link, or "Artist - Song") into the
song's official 30-second preview clip, the one those services already play publicly.

No full song is ever downloaded: Spotify, Apple Music and YouTube audio is copy-protected and their terms
forbid downloading it. Deezer and Apple publish preview clips through free public APIs (no key needed);
for Spotify and YouTube links only the song's title and artist are read, and the same song's preview is
then looked up on Deezer (or Apple). The preview is analysed in memory and never stored.

A pasted URL is never fetched as is: links are reduced to an ID and looked up through the services' public
APIs. The one exception is a share link on a music service's own short-link host (SHORT_HOSTS), which is
opened to see where it leads; redirects are followed only to the music services' own https hosts.
"""
from __future__ import annotations

import html
import http.client
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from difflib import SequenceMatcher

UA = {"User-Agent": "Mozilla/5.0 (compatible; ai-music-detector research demo)"}
TIMEOUT = 15
MAX_PREVIEW_BYTES = 5_000_000
MAX_PAGE_BYTES = 600_000  # share-link and song pages: only the start of the HTML is needed
SHORT_HOSTS = {"link.deezer.com", "deezer.page.link", "dzr.page.link", "spotify.link", "spotify.app.link", "spoti.fi", "apple.co"}
APPLE_MUSIC_HOSTS = {"music.apple.com", "geo.music.apple.com", "embed.music.apple.com", "itunes.apple.com", "geo.itunes.apple.com"}
CONFIDENT = 0.8  # below this the page asks the visitor to confirm the match before scoring it
UNSURE = 0.75    # a match that cannot be checked by name (the title is in another script): offered for confirmation
FLOOR = 0.45     # below this the closest candidate is not offered at all
_BELOW = round(CONFIDENT - 0.01, 2)  # the score of a match that must be confirmed
_ONLY = "Only Spotify, Apple Music, YouTube and Deezer links work. You can also type 'Artist - Song'."


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


# ------------------------------------------------------------------ HTTP

_PREVIEW_HOSTS = ("dzcdn.net", "mzstatic.com")  # where Deezer and Apple serve preview clips and artwork


def _music_host(host: str | None) -> bool:
    host = (host or "").lower()
    return bool(_service(host)) or any(host == s or host.endswith("." + s) for s in _PREVIEW_HOSTS)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to the music services' own https hosts, so no link can make the server fetch an
    internal address (a name like 'localtest.me' looks public but points back at the server)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urllib.parse.urlsplit(newurl)
        if p.scheme != "https" or not _music_host(p.hostname):
            raise LookupError_("That link points somewhere unexpected, so it was not opened.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_SafeRedirect)


def _open(url: str):
    return _OPENER.open(urllib.request.Request(url, headers=UA), timeout=TIMEOUT)


def _get(url: str, data_limit: int | None = None) -> tuple[str, bytes]:
    with _open(url) as r:
        body = r.read(data_limit + 1 if data_limit else -1)
        if data_limit and len(body) > data_limit:
            raise LookupError_("The preview clip is unexpectedly large; try another song.")
        return r.geturl(), body


def _page(url: str) -> tuple[str, str]:
    """A web page's final URL (after redirects) and the start of its HTML."""
    with _open(url) as r:
        return r.geturl(), r.read(MAX_PAGE_BYTES).decode("utf-8", "replace")


def _json(url: str) -> dict:
    return json.loads(_get(url, 3_000_000)[1])


def _meta(page: str, prop: str) -> str:
    m = re.search(rf'<meta (?:property|name)="{prop}" content="([^"]*)"', page)
    return html.unescape(m.group(1)) if m else ""


# ------------------------------------------------------------------ comparing names and titles

_TRANSLIT = str.maketrans({"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ł": "l", "Ł": "L",
                           "đ": "d", "Đ": "D", "ð": "d", "Ð": "D", "þ": "th", "Þ": "TH", "ı": "i"})


def _norm(s: str) -> str:
    """Lowercase letters and digits of every script, Latin accents folded ('Björk', 'MØ' -> 'bjork', 'mo'),
    punctuation gone."""
    s = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", s).translate(_TRANSLIT))
    s = unicodedata.normalize("NFC", re.sub(r"[̀-ͯ]", "", s))  # keeps Japanese dakuten (U+3099/309A)
    return " ".join(re.sub(r"[^\w\s]|_", "", s.casefold()).split())


def _sim(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:  # nothing comparable left (e.g. only emoji): equal only if literally equal
        return float(bool(na == nb and a.strip() and a.strip() == b.strip()))
    if not set(na.replace(" ", "")) & set(nb.replace(" ", "")):
        return 0.0  # not one letter in common, e.g. '좋은 날' and 'Bad Day'
    return SequenceMatcher(None, na, nb).ratio()


def _non_latin(s: str) -> bool:
    return any(c.isalpha() and not unicodedata.name(c, "LATIN").startswith("LATIN") for c in s)


def _japanese_or_chinese(s: str) -> bool:
    return any(unicodedata.name(c, "").startswith(("HIRAGANA", "KATAKANA", "CJK UNIFIED")) for c in s)


# A title is split into its core ("Get Lucky") and qualifiers taken from brackets and " - " suffixes. Qualifiers
# that name the same recording ("Remastered 2016", "Radio Edit", "Taylor's Version", "From 'Frozen'") and
# guest credits ("feat. X") are dropped; the rest mark a different recording ("Live", "Remix", "Pianoforte"),
# or translate the title into another script ("뚜두뚜두 (DDU-DU DDU-DU)").
# One level of nesting: '(Live at Wembley (2011 Remaster))', '[The Official 2010 FIFA World Cup (TM) Song]'.
_BRACKET = re.compile(r"\s*[\(\[（［【]((?:[^\(\)\[\]（）［］【】]|[\(\[（［【][^\(\)\[\]（）［］【】]*[\)\]）］】])*)[\)\]）］】]?")
_PART_SPLIT = re.compile(r"\s*/\s*|\s+[-–—]\s+|\s*[,;]\s+")
_CREDIT = re.compile(r"(?:feat\.?|ft\.|featuring|with|prod\.?)\s", re.I)
_TAIL_CREDIT = re.compile(r"\s+(?:feat\.?|ft\.|featuring)\s.*$", re.I)
_NEUTRAL = set("remaster remastered remasterd digital digitally version edition edit mix single album radio original "
               "originale mono stereo explicit clean bonus track deluxe taylors 7 inch lp".split())
_OTHER = re.compile(r"\b(?:live|remix|rmx|mix|mashup|medley|cover|acoustic|unplugged|stripped|extended|instrumental|inst|"
                    r"karaoke|a ?cappella|acapella|demo|sped ?up|slowed|nightcore|8d|reverb|tribute|parody|commentary|"
                    r"interview|rehearsal|session|piano|pianoforte|orchestral|symphonic|drumless|dub|club|vip|bootleg|"
                    r"flip|reimagined|re-?recorded|rendition|edition|version|edit)\b|カラオケ|原曲|伴奏|노래방", re.I)
_SOFT = {"version", "edition", "edit", "mix"}  # also in same-recording tags ('Radio Edit', "Taylor's Version")
_KARAOKE = re.compile(r"\b(?:karaoke|originally performed|made popular|made famous|in the style of|tribute to|salute to)\b"
                      r"|カラオケ|原曲歌手|노래방", re.I)
_TRIBUTE = re.compile(r"\b(?:tribute|salute|sounds? of|karaoke|made famous|in the style of|originally performed|cover band)\b", re.I)
_LIVE_ALBUM = re.compile(r"\blive\s*$|[\(\[]live[\)\]]|\blive (?:at|from|in|on)\b|\bin concert\b|\bunplugged\b|\bkaraoke\b", re.I)
_QUOTES = re.compile(r"[‘’“”\"「」『』]")


def _is_same(part: str) -> bool:
    n = _norm(part)
    if not n or re.match(r"from\b|.*\bsoundtrack\b|ost\b", n):
        return True
    return all(t in _NEUTRAL or re.fullmatch(r"\d{2,4}", t) for t in n.split())


def _split_title(title: str) -> tuple[str, list[str]]:
    quals: list[str] = []

    def parts(text: str) -> None:
        for p in _PART_SPLIT.split(text):
            p = p.strip(" \"'")
            if not p:
                continue
            if _CREDIT.match(p + " "):  # a credit runs to the end of its bracket
                break
            if not _is_same(p):
                quals.append(p)

    def bracket(m: re.Match) -> str:
        if m.start() == 0:  # "(I Can't Get No) Satisfaction": a leading bracket is part of the title
            return m.group(0)
        parts(m.group(1))
        return " "

    core = _BRACKET.sub(bracket, title)
    if m := re.match(r"(.+?)\s+[-–—]\s+(.+)$", core):
        core = m.group(1)
        parts(m.group(2))
    return " ".join(_TAIL_CREDIT.sub("", core).split()), quals


def _translations(core: str, quals: list[str]) -> list[str]:
    return [q for q in quals if _non_latin(q) != _non_latin(core) and not _OTHER.search(q)]


def _alts(title: str) -> list[str]:
    """The core title and any translation of it in brackets."""
    core, quals = _split_title(title)
    return [core] + _translations(core, quals)


def _q_title(title: str, core: str | None = None) -> str:
    """The title as a search query: credits and same-recording tags dropped, other-recording words kept."""
    c, quals = _split_title(title)
    return " ".join(_QUOTES.sub(" ", " ".join([core or c] + [q for q in quals if _OTHER.search(q)])).split())


def _markers(quals: list[str]) -> set[str]:
    return {m.group(0).lower() for q in quals for m in _OTHER.finditer(q)}


def _penalty(asked: str, found: str, album: str = "") -> float:
    """How far `found` (on album `album`) is from the recording `asked` names, beyond the title words themselves."""
    (ca, qa), (cf, qf) = _split_title(asked), _split_title(found)
    qa = [q for q in qa if q not in _translations(ca, qa)]
    qf = [q for q in qf if q not in _translations(cf, qf)]
    ma, mf = _markers(qa), _markers(qf)
    ma |= _markers([ca]) - _markers([cf]) - _SOFT  # typed without brackets: 'Bohemian Rhapsody live' (not 'Live Forever')
    pen = 0.0
    if mf and not (mf & ma):
        pen += 0.35  # a live version, remix, cover... that was not asked for: confirm it first
    elif ma and not mf:
        pen += 0.35  # a live version, remix... was asked for and this is the plain recording: confirm it first
    elif (ma - mf) - _SOFT:
        pen += 0.35  # asked for 'Remix (Slowed + Reverb)', found only the remix: it sounds different, so confirm
    unasked = [q for q in qf if not any(_sim(q, a) >= 0.8 for a in qa)]
    if any(not _OTHER.search(q) for q in unasked):
        subtitles = [a for a in qa if not _OTHER.search(a)]
        # Another subtitle than the one asked for ('Esto Es Africa' for 'This Time for Africa') is usually another
        # language edition or song; one nobody asked about ('(Rockshow)') matters less.
        pen += 0.35 if subtitles and not any(_sim(q, a) >= 0.8 for q in qf for a in subtitles) else 0.1
    if ma and mf & ma and (marked := [q for q in unasked if _OTHER.search(q)]):
        rest_a = [r for a in qa if (r := _OTHER.sub(" ", a)).strip()]
        other = all((r := _OTHER.sub(" ", q)).strip() and rest_a and max(_sim(r, x) for x in rest_a) < 0.5 for q in marked)
        pen += 0.35 if other else 0.1  # another concert or remixer than the one asked for, or just worded differently
    if (_KARAOKE.search(found) or _KARAOKE.search(album)) and not _KARAOKE.search(asked):
        pen += 0.4
    if album and _LIVE_ALBUM.search(album) and "live" not in ma | mf:
        pen += 0.35  # a live album that does not say "live" in its track titles
    return pen


def _title_sim(asked: str, found: str, album: str = "") -> float:
    s = max(_sim(x, y) for x in _alts(asked) for y in _alts(found))
    return max(0.0, s - _penalty(asked, found, album))


_GUEST = re.compile(r"\s*(?:,|&|\+|\bfeat\.?|\bft\.?|\bfeaturing\b|\bwith\b|\bx\b|\band\b|\bvs\.?)\s*", re.I)
_TAIL_GUEST = re.compile(r"\s*[\(\[]?\s*\b(?:feat\.?|ft\.|featuring|with)\s.*$", re.I)


def _artist_key(s: str) -> str:
    words = _norm(s).split()
    # 'Simon & Garfunkel' == 'Simon and Garfunkel'; 'Ali Sethi x Shae Gill' == 'Ali Sethi & Shae Gill' (not 'Malcolm X')
    words = [w for i, w in enumerate(words) if w not in ("and", "n") and not (w == "x" and 0 < i < len(words) - 1)]
    return " ".join(words[1:] if words[:1] == ["the"] and len(words) > 1 else words)


def _artist_sim1(asked: str, found: str) -> float:
    a, f = _artist_key(asked), _artist_key(found)
    if not a or not f:
        return _sim(asked, found)
    if a == f:
        return 1.0
    parts = [k for p in _GUEST.split(found) if (k := _artist_key(p))]
    if a in parts:
        return 1.0
    if _TRIBUTE.search(found) and not _TRIBUTE.search(asked):  # 'The Sounds Of Garth Brooks', 'Tribute to Ed Sheeran'
        return 0.3
    near = max([SequenceMatcher(None, a, f).ratio()] + [SequenceMatcher(None, a, p).ratio() for p in parts])
    if near >= 0.9:
        return 0.9
    if len(set(a.split())) >= 2 and set(a.split()) <= set(f.split()):
        return 0.8
    return 0.7 * SequenceMatcher(None, a, f).ratio()


def _artist_sim(asked: str, found: str) -> float:
    """1 for the same artist, also when guests are credited on either side ("Daft Punk & Pharrell Williams",
    "Mark Ronson feat. Bruno Mars"); 0.9 for a near spelling; 0.8 for a duo typed short ("Hall and Oates" ~
    "Daryl Hall & John Oates"); tribute and karaoke acts that merely contain the name 0.3."""
    base = _TAIL_GUEST.sub("", asked).strip()
    return max(_artist_sim1(asked, found), _artist_sim1(base, found) if base and base != asked else 0.0)


# ------------------------------------------------------------------ search by artist + title

@dataclass
class _Reading:
    """One way of reading what the visitor asked for: candidate artist names (with how much each is trusted)
    and a title. query_artist is the name put into the search query ('' = search by title)."""
    artists: list[tuple[str, float]]
    title: str
    query_artist: str = field(default="")

    def __post_init__(self):
        if not self.query_artist:
            self.query_artist = next((n for n, w in self.artists if w >= 1.0), "")


def _combine(t: float, a: float) -> float:
    """Title and artist similarity in one score; confident only if the artist matches at least as a near spelling
    ('Jimi Hendrix' ~ 'The Jimi Hendrix Experience' is offered for confirmation)."""
    s = 0.6 * t + 0.4 * a
    return s if a >= 0.9 else min(s, _BELOW)


def _reading_score(r: _Reading, artist: str, title: str, album: str) -> tuple[float, float]:
    t = _title_sim(r.title, title, album)
    if not r.artists:
        return min(0.6 * t + 0.24, _BELOW), t  # artist unknown: never confident from the title alone
    return _combine(t, max(w * _artist_sim(n, artist) for n, w in r.artists)), t


def _free_score(text: str, artist: str, title: str, album: str) -> tuple[float, float]:
    """Typed text without a clear artist/title split ('adele hello'): the whole text as a title, or any split of
    its words into artist + title (either order). Only a split whose artist really matches can be confident."""
    t = _title_sim(text, title, album)
    best = (min(0.6 * t + 0.24, _BELOW), t)
    words = text.split()[:12]
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        for who, what in ((a, b), (b, a)):
            ts = _title_sim(what, title, album)
            best = max(best, (_combine(ts, _artist_sim(who, artist)), ts))
    return best


def _deezer_alias(name: str) -> str:
    """Deezer's own spelling of an artist written in a non-Latin script ('米津玄師' -> 'Kenshi Yonezu')."""
    try:
        data = _json("https://api.deezer.com/search/artist?" + urllib.parse.urlencode({"q": name, "limit": 1})).get("data", [])
    except Exception:  # noqa: BLE001
        return ""
    return data[0]["name"] if data and data[0].get("name") and data[0]["name"] != name else ""


@dataclass
class _Cand:
    score: float
    title_sim: float
    artist_sim: float
    rank: int
    match: Match


def _find(readings: list[_Reading], asked_via: str, free_text: str = "", country: str | None = None,
          named: bool | None = None) -> Match:
    """The best preview for `readings`. `named`: the visitor clearly named an artist, so a song by someone
    else entirely is an error rather than a candidate to confirm (default: any reading trusts an artist fully)."""
    # The visitor's own artist, when written only in a non-Latin script, also by Deezer's spelling of it.
    aliases: dict[str, str] = {}
    if readings and not any(w >= 1.0 and not _non_latin(n) for n, w in readings[0].artists):
        for n, w in readings[0].artists[:2]:
            if w >= 1.0 and _non_latin(n) and (alias := _deezer_alias(n)):
                aliases[n] = alias
    for r in readings:
        r.artists += [(aliases[n], w) for n, w in r.artists if n in aliases]

    queries, keys = [], set()

    def add(q: str) -> None:
        q = " ".join(q.split())
        if q and (k := tuple(sorted(_norm(q).split()))) not in keys:
            keys.add(k)
            queries.append(q)

    for r in readings:
        for lead in [r.query_artist] + ([aliases[r.query_artist]] if r.query_artist in aliases else []):
            for alt in _alts(r.title):
                add(f"{lead} {_q_title(r.title, alt)}")
                add(f"{lead} {_QUOTES.sub(' ', alt)}")  # the plain title too: a remix or cover may not be listed
    if free_text:
        add(_q_title(free_text))
    queries = queries[:4]
    if named is None:
        named = any(w >= 1.0 for r in readings for _, w in r.artists)

    def best_of(cands: list[_Cand]) -> _Cand | None:
        return max(cands, key=lambda c: (c.score, c.rank), default=None)

    def cand(artist: str, title: str, album: str, rank: int, m: Match) -> _Cand:
        best = max((_reading_score(r, artist, title, album) for r in readings), default=(0.0, 0.0))
        if free_text:
            best = max(best, _free_score(free_text, artist, title, album))
        a = max((w * _artist_sim(n, artist) for r in readings for n, w in r.artists), default=0.0)
        return _Cand(best[0], best[1], a, rank, m)

    def exact(c: _Cand | None) -> bool:
        return bool(c and c.score >= 0.95 and c.title_sim >= 0.97)

    pool: list[_Cand] = []  # every candidate, in the order the services ranked them
    seen = set()
    for q in queries:
        if exact(best_of(pool)):
            break
        try:
            data = _json("https://api.deezer.com/search?" + urllib.parse.urlencode({"q": q, "limit": 15})).get("data", [])
        except Exception:  # noqa: BLE001 - Deezer down or busy: Apple is tried below
            data = []
        for d in data:
            if d.get("id") in seen or not d.get("preview"):
                continue
            seen.add(d.get("id"))
            album = d.get("album", {}).get("title", "")
            pool.append(cand(d["artist"]["name"], d["title"], album, d.get("rank", 0),
                             Match(d["title"], d["artist"]["name"], album, d.get("album", {}).get("cover_medium", ""),
                                   d["preview"], "Deezer", d.get("link", ""), asked_via, 0)))

    best = best_of(pool)
    asked_text = " ".join([free_text] + [r.title for r in readings] + [n for r in readings for n, _ in r.artists])
    # Not an exact match: see what Apple has. A second search only if still unsure: the next query, or Japan's
    # store for Japanese and Chinese titles (the US store romanises them).
    apple = [(country, queries[0])] if queries else []
    for i in range(3):
        if i >= len(apple) or exact(best):
            break
        store, term = apple[i]
        params = {"term": term, "entity": "song", "limit": 15} | ({"country": store} if store else {})
        try:
            results = _json("https://itunes.apple.com/search?" + urllib.parse.urlencode(params)).get("results", [])
        except Exception:  # noqa: BLE001 - Apple rate-limits shared servers; keep what Deezer found
            results = []
        for r in results:
            if r.get("kind") != "song" or not r.get("previewUrl"):
                continue
            c = cand(r["artistName"], r["trackName"], r.get("collectionName", ""), -1,
                     Match(r["trackName"], r["artistName"], r.get("collectionName", ""),
                           r.get("artworkUrl100", "").replace("100x100", "300x300"), r["previewUrl"], "Apple",
                           r.get("trackViewUrl", ""), asked_via, 0))
            pool.append(c)
            if best is None or c.score > best.score:  # Apple must be strictly better; on a tie Deezer is kept
                best = c
        if i == 0 and (best is None or best.score < CONFIDENT):
            if store != "jp" and _japanese_or_chinese(asked_text):
                apple.append(("jp", queries[0]))
            elif len(queries) > 1:
                apple.append((country, queries[1]))
    # A title in another script than the services use ('晴天' is listed as 'Sunny Day'): the names cannot be
    # compared, so offer the services' first song by the right artist for the visitor to confirm.
    titles = [r.title for r in readings] + ([free_text] if free_text else [])
    crossed = False
    if (best is None or best.score < UNSURE) and any(_non_latin(t) for t in titles):
        if c := next((c for c in pool if c.artist_sim >= 0.9 and not _non_latin(c.match.title)), None):
            c.score, best, crossed = UNSURE, c, True
    asked = free_text or (" ".join(f"{readings[0].query_artist} {readings[0].title}".split()) if readings else "")
    if best is None:
        raise LookupError_(f"Could not find a preview clip for \"{asked}\". Try 'Artist - Song', or upload the audio file.")
    m = best.match
    if best.score < FLOOR or (named and best.artist_sim < 0.4) or (best.title_sim < 0.3 and not crossed):
        raise LookupError_(f"Could not find that song on Deezer or Apple (the closest was \"{m.title}\" by {m.artist}). "
                           "Try typing 'Artist - Song', or upload the audio file.")
    # Rounded for display only: a score just under CONFIDENT must not round up to it.
    m.similarity = round(best.score, 2) if best.score >= CONFIDENT else min(round(best.score, 2), _BELOW)
    return m


def search(artist: str, title: str, asked_via: str, country: str | None = None) -> Match:
    return _find([_Reading([(artist, 1.0)] if artist else [], title)], asked_via, country=country)


def _substitute(m: Match) -> Match:
    """A different release of the song than the link named: always let the visitor confirm it."""
    m.similarity = min(m.similarity, _BELOW)
    return m


# ------------------------------------------------------------------ exact lookups

def _deezer_match(d: dict, asked_via: str) -> Match:
    return Match(d["title"], d["artist"]["name"], d.get("album", {}).get("title", ""), d.get("album", {}).get("cover_medium", ""),
                 d["preview"], "Deezer", d.get("link", ""), asked_via, 1.0)


def _deezer_track(track_id: str, asked_via: str) -> Match:
    d = _json(f"https://api.deezer.com/track/{track_id}")
    if "error" in d and d["error"].get("code") in (4, 700):  # quota or service busy: one retry
        time.sleep(1)
        d = _json(f"https://api.deezer.com/track/{track_id}")
    if "error" in d:
        if d["error"].get("code") == 800:
            raise LookupError_("Deezer does not know that track.")
        raise LookupError_("Deezer is busy right now. Try again in a few seconds.")
    if d.get("preview"):
        return _deezer_match(d, asked_via)
    # No preview here (not offered where this site runs, or withdrawn): try the same recording, then the same song.
    if d.get("isrc"):
        try:
            alt = _json(f"https://api.deezer.com/track/isrc:{urllib.parse.quote(d['isrc'])}")
            if alt.get("preview") and alt.get("isrc") == d["isrc"]:
                return _deezer_match(alt, asked_via)
        except Exception:  # noqa: BLE001
            pass
    try:
        return _substitute(search(d.get("artist", {}).get("name", ""), d.get("title", ""), asked_via))
    except LookupError_:
        if not d.get("available_countries"):
            raise LookupError_("Deezer no longer offers this track, and the same song was not found on Apple either.") from None
        raise LookupError_("Deezer does not offer a preview of this track where this site runs, and the same song was not "
                           "found on Apple either. Try 'Artist - Song'.") from None


def _apple_track(track_id: str, asked_via: str, country: str | None) -> Match:
    def look(cc: str | None) -> list[dict]:
        url = f"https://itunes.apple.com/lookup?id={track_id}&entity=song" + (f"&country={cc}" if cc else "")
        try:
            data = _json(url)
        except urllib.error.HTTPError as e:
            if e.code == 400:  # an ID too long, or a store code that does not exist
                return []
            raise
        return [r for r in data.get("results", []) if r.get("wrapperType") == "track" and r.get("kind") == "song"]

    res = (look(country) if country else []) or (look(None) if country != "us" else [])  # the link's own store first
    if ok := [r for r in res if r.get("previewUrl")]:
        r = ok[0]
        return Match(r["trackName"], r["artistName"], r.get("collectionName", ""), r.get("artworkUrl100", "").replace("100x100", "300x300"),
                     r["previewUrl"], "Apple", r.get("trackViewUrl", ""), asked_via, 1.0)
    if res:  # the song exists but Apple has no preview of it: find the same song
        return _substitute(search(res[0]["artistName"], res[0]["trackName"], asked_via, country=country))
    if not country:
        raise LookupError_("Apple did not find that song in its US store. Paste the link with its country code "
                           "(music.apple.com/de/..., for example), or type 'Artist - Song'.")
    raise LookupError_("Apple Music has no song with that link. Check the link, or type 'Artist - Song'.")


# ------------------------------------------------------------------ title and artist from Spotify / YouTube

def _spotify(track_id: str) -> Match:
    try:
        _, page = _page(f"https://open.spotify.com/track/{track_id}")
    except urllib.error.HTTPError as e:
        if e.code in (400, 404, 410):
            raise LookupError_("Spotify could not find that track. Check the link is complete, or type 'Artist - Song'.") from None
        raise
    title, desc = _meta(page, "og:title"), _meta(page, "og:description")  # desc: "Artist · Album · Song · 1997"
    if not title:
        raise LookupError_("Could not read that Spotify link. Try 'Artist - Song' instead.")
    credit = desc.split(" · ")[0].strip() if desc else ""
    first = credit.split(", ")[0]  # "Madonna, David Reitzas" -> "Madonna" (but "Earth, Wind & Fire" stays whole below)
    artists = [(credit, 1.0)] + ([(first, 0.95)] if first != credit else []) if credit else []
    return _find([_Reading(artists, title, query_artist=first)], "Spotify link")


_YT_JUNK = re.compile(r"\b(?:official|oficial|offiziell\w*|video|vid[eé]o|videoclip|clip|audio|lyrics?|letra|paroles|"
                      r"visuali[sz]er|hd|hq|4k|8k|mv|m/v|full (?:video )?song|music video|lyric video|teaser|colou?r coded)\b", re.I)
_YT_KEEP = re.compile(r"\b(?:live|remix|mix|cover|acoustic|version|edit|instrumental|session|unplugged|piano|orchestral|"
                      r"karaoke|sped|slowed|feat|ft|featuring|with)\b", re.I)
_YT_TAIL = re.compile(r"\s*[-–—|:]*\s*(?:official\s+)?(?:music\s+)?(?:m/?v|video|audio|lyric video|visuali[sz]er)\s*$", re.I)
_YT_NOISE = re.compile(r"^(?:season\s*\d+|s\d+(?:\s*e\d+)?|ep(?:isode)?\.?\s*\d+|(?:\d+k\s*)?(?:full\s+)?(?:video\s+)?song|\d+k|"
                       r"(?:grand|semi)[\s-]*final.*|.*\bwinners?\b.*|.*\beurovision\b.*|.*\btiny desk\b.*|.*\bawards?\b.*|"
                       r"(?:official\s+)?(?:music\s+|lyrics?\s+)?(?:video|audio|visuali[sz]er)|lyrics?|m/?v|official|"
                       r"\d{4}|[\W\d_]*)$", re.I)
_YT_SPLIT = re.compile(r"\s*(?:\|\|?|//|•)\s*")
_YT_BRACKETS = re.compile(r"[\(\[（［【][^\)\]）］】]*[\)\]）］】]")
_YT_QUOTED_CJK = re.compile(r"[「『](?P<song>[^「」『』]+)[」』]")
_YT_QUOTED = re.compile(r"(?:^|\s)['\"‘“](?P<song>[^'\"‘’“”]{2,})['\"’”](?=\s|$)")


def _yt_clean(title: str) -> str:
    """Drop brackets that only say 'Official Video', '[Lyrics]', '(4K)'...; keep ones naming a recording ('(Live Aid)')."""
    def br(m: re.Match) -> str:
        return " " if _YT_JUNK.search(m.group(1)) and not _YT_KEEP.search(m.group(1)) else m.group(0)
    return " ".join(re.sub(r"\s*[\(\[（［【]([^\)\]）］】]*)[\)\]）］】]", br, title).split())


def _yt_song(song: str) -> str:
    """Drop an unbracketed 'M/V', 'Official Music Video' or '8K Full Song' at the end, and surrounding quotes."""
    s = re.sub(r"\s+(?:\d+k\s+)?full\s+(?:video\s+)?song\b|\s+\d+k\b", "", song, flags=re.I)
    s = _YT_TAIL.sub("", s).strip()
    return (s or song.strip()).strip(" '\"‘’“”「」『』")


def _yt_names(artist: str) -> list[str]:
    """'BTS (방탄소년단)' -> ['BTS', '방탄소년단']; 'Y2K & bbno$' -> ['Y2K & bbno$', 'Y2K'];
    '米津玄師 Kenshi Yonezu' -> [..., 'Kenshi Yonezu', '米津玄師']."""
    base = " ".join(_YT_BRACKETS.sub(" ", artist).split()).strip(" '\"‘’“”")
    names = [base] + [b.strip() for b in re.findall(r"[\(\[（［【]([^\)\]）］】]*)[\)\]）］】]", artist) if b.strip()]
    names.append(re.split(r"\s+(?:ft\.?|feat\.?|featuring|x|&|and)\s+|\s*,\s*", base, flags=re.I)[0])
    latin, other = (" ".join(w for w in base.split() if _non_latin(w) == f) for f in (False, True))
    if latin and other:  # one name written in two scripts
        names += [latin, other]
    return [n for n in dict.fromkeys(names) if n]


def _yt_readings(title: str, author: str) -> list[_Reading]:
    """Ways of reading a video title: 'Artist - Song', 'Song - Artist', ARTIST 'SONG' M/V, YOASOBI「Song」,
    'IU(아이유)_LILAC', '【Ado】Song', 'Adele performing Song', 'Song | Film | Singer'..."""
    channel = re.sub(r"\s*-\s*Topic$|\s*VEVO$|\s*Official(?: Channel)?$", "", author or "", flags=re.I).strip()
    ch = [(channel, 1.0 if re.search(r"-\s*Topic$|VEVO$", author or "", re.I) else 0.9)] if channel else []
    t = _yt_clean(title)
    segs = [s.strip() for s in _YT_SPLIT.split(t) if s.strip()]
    segs = [s for s in segs if not (_YT_NOISE.match(s) and not re.search(r"\s[-–—]\s", s))  # 'ABBA - The Winner...' stays
            and not (channel and _norm(s) and _norm(s) in _norm(channel))] or segs[:1]
    segs = segs or [" ".join(title.split())]  # the title was nothing but "(Official Video)"-style brackets
    cover = ""
    if m := re.search(r"[\(\[]([^\)\]]*\bcover\b[^\)\]]*)[\)\]]|\bcover(?:ed)? by\s+(.+)$", t, re.I):
        who = re.sub(r"\b(?:acoustic|piano|cover|covered|by|official|live|version)\b", "", m.group(1) or m.group(2) or "", flags=re.I)
        cover = " ".join(who.split()).strip(" -,") or channel
        if channel and set(_norm(cover).split()) <= set(_norm(channel).split()):
            cover = channel  # 'Acoustic Cover by Leroy' on the channel 'Leroy Sanchez'
    if dash := next((s for s in segs if re.search(r"\s[-–—]\s", s)), None):
        others = [(o, 0.9) for o in segs if o != dash][:3]  # the singer or composer often sits in another segment
        a, b = (x.strip() for x in re.split(r"\s+[-–—]\s+", dash, maxsplit=1))
        b_core, b_brackets = " ".join(_YT_BRACKETS.sub(" ", b).split()), " ".join(_YT_BRACKETS.findall(b))
        readings = []
        # 'Artist - Song', and 'Song - Artist' with the right half's brackets kept on the song ('(Imanbek Remix)')
        for artist, song in ((a, _yt_song(b)), (_yt_song(b_core), _yt_song(f"{a} {b_brackets}"))):
            names = _yt_names(artist) or [artist]
            arts = ([(cover, 1.0)] if cover else []) + [(n, 0.9 if cover else 1.0) for n in names] + others
            arts += [(n, w * 0.9) for n, w in ch]
            if cover:
                readings.append(_Reading(list(arts), song, query_artist=cover))
            readings.append(_Reading(list(arts), song, query_artist=names[0]))
        return readings
    seg = segs[0]

    def named(artist: str, song: str) -> list[_Reading]:
        names = _yt_names(artist) or ([channel] if channel else [])
        arts = [(n, 1.0) for n in names] + [(o, 0.9) for o in segs[1:4]] + [(n, 0.9) for n, _ in ch]
        return [_Reading(arts, _yt_song(song), query_artist=names[0] if names else "")]

    if m := re.match(r"(.+?[\)）])\s*_\s*(.+)$", seg):  # 1theK: 'IU(아이유)_LILAC(라일락)'
        return named(m.group(1), m.group(2))
    if m := re.match(r"[【\[]([^】\]]+)[】\]]\s*(.+)$", seg):  # '【Ado】うっせぇわ'
        return named(m.group(1), m.group(2))
    if m := _YT_QUOTED_CJK.search(seg) or _YT_QUOTED.search(seg):  # 'YOASOBI「夜に駆ける」', "BTS (방탄소년단) 'Dynamite'"
        return named(seg[:m.start()].strip() or channel, m.group("song"))
    if m := re.match(r"(.+?)\s+(?:performs?|performing|sings?|singing)\s+(.+)$", seg, re.I):  # 'Adele performing Song'
        return named(m.group(1), re.sub(r"\s+live$", "", m.group(2), flags=re.I))
    readings = []
    for s in segs[:3]:
        others = [(o, 0.9) for o in segs if o != s][:3]
        lead = ch[0][0] if ch and ch[0][1] >= 1.0 else ""
        song = _yt_song(s)
        readings.append(_Reading(others + ch, song, query_artist=lead))
        if ": " in song:  # "Aashiqui 2: Tum Hi Ho", "Artist: Song"
            before, after = song.rsplit(": ", 1)
            readings.append(_Reading(others + ch + [(before, 0.9)], after, query_artist=lead))
    return readings


def _youtube(video_id: str) -> Match:
    watch = f"https://www.youtube.com/watch?v={video_id}"
    try:
        d = _json("https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(watch, safe=""))
        title, author = d.get("title", ""), d.get("author_name", "")
    except urllib.error.HTTPError as e:
        if e.code in (400, 404):
            raise LookupError_("YouTube could not find that video (it may be private or removed). Try 'Artist - Song'.") from None
        if e.code not in (401, 403):
            raise
        # Embedding is turned off for this video: read its title from the video's own page instead.
        _, page = _page(watch)
        title = _meta(page, "og:title") or _meta(page, "title")
        m = re.search(r'<link itemprop="name" content="([^"]*)"', page)
        author = html.unescape(m.group(1)) if m else ""
    if not title:
        raise LookupError_("YouTube did not give that video's title. Try 'Artist - Song'.")
    return _find(_yt_readings(title, author), "YouTube link")


# ------------------------------------------------------------------ entry point

_HOSTS = (r"(?:[\w-]+\.)*(?:spotify\.com|spotify\.link|spotify\.app\.link|spoti\.fi|youtu\.be|youtube(?:-nocookie)?\.com|"
          r"deezer\.com|deezer\.page\.link|dzr\.page\.link|apple\.com|apple\.co)")
# The host must end where it seems to: 'open.spotify.com.evil.example' and 'spotify.com@evil.example' are not Spotify.
_URL = re.compile(rf"(?<![\w.@-])(?:https?://)?{_HOSTS}(?![\w-]|\.[\w-]|@)(?::\d+)?(?:[/?#][^\s<>\"']*)?", re.I)
_EMBEDDED = re.compile(r"https://(?:open\.spotify\.com/(?:intl-[\w-]+/)?(?:track|album|playlist|artist|episode|show)/[A-Za-z0-9]{22}"
                       r"|(?:www\.)?deezer\.com/(?:[a-z]{2}/)?(?:track|album|playlist|artist)/\d+|music\.apple\.com/[^\s\"'<>\\]+)")
_NOT_A_SONG = "That looks like {what}, not a single song. Share a single song instead."
_CUT_OFF = "That {service} song link looks cut off. Copy it again, or type 'Artist - Song'."


def _service(host: str) -> str:
    if host in SHORT_HOSTS:
        return "short"
    for name, suffixes in (("spotify", ("spotify.com",)), ("deezer", ("deezer.com",)), ("apple", ("apple.com",)),
                           ("youtube", ("youtube.com", "youtube-nocookie.com", "youtu.be"))):
        if any(host == s or host.endswith("." + s) for s in suffixes):
            return name
    return ""


def _resolve_url(url: str, follow: bool = True) -> Match:
    url = unicodedata.normalize("NFKC", url)  # full-width letters and digits, as some keyboards type them
    p = urllib.parse.urlsplit(url if re.match(r"https?://", url, re.I) else "https://" + url)
    host, path = (p.hostname or "").lower(), p.path
    query = urllib.parse.parse_qs(p.query)
    kind = _service(host)
    if kind == "short":
        if not follow:
            raise LookupError_("That share link did not lead to a song. Open it, copy the song's own link, and paste that.")
        # A share code is plain ASCII; anything after it ('…/hRkBrwub9xbこの曲いいよ') is a comment typed straight after.
        code, q = re.match(r"[!-~]*", path).group(0), re.match(r"[!-~]*", p.query).group(0)
        try:
            final, page = _page(urllib.parse.urlunsplit(("https", host, code, q, "")))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404, 410):
                raise LookupError_("That share link no longer opens a song (it may have expired). Share the song again, "
                                   "or type 'Artist - Song'.") from None
            raise
        if _service((urllib.parse.urlsplit(final).hostname or "").lower()) not in ("", "short"):
            return _resolve_url(final, follow=False)
        if m := _EMBEDDED.search(page.replace("\\/", "/")):  # e.g. spotify.link answers with a page that links on
            return _resolve_url(html.unescape(m.group(0)), follow=False)
        raise LookupError_("That share link did not lead to a song. Open it, copy the song's own link, and paste that.")
    if kind == "deezer":
        m = re.match(r"/(?:widget/[\w-]+/)?(?:[a-z]{2}(?:-[a-z]{2})?/)?(track|album|playlist|artist|episode|show|podcast)/([0-9]+)",
                     path, re.I)
        if m and m.group(1).lower() == "track":
            return _deezer_track(m.group(2), "Deezer link")
        raise LookupError_(_NOT_A_SONG.format(what=f"a Deezer {m.group(1).lower() if m else 'page'} link"))
    if kind == "apple":
        if host == "podcasts.apple.com":
            raise LookupError_(_NOT_A_SONG.format(what="an Apple Podcasts link"))
        if host not in APPLE_MUSIC_HOSTS:
            raise LookupError_(_ONLY)
        cc = m.group(1).lower() if (m := re.match(r"/([a-z]{2})(?:/|$)", path, re.I)) else None
        iid = re.match(r"[0-9]+", (query.get("i") or [""])[0])  # Apple's own share links can append '?app=music' to i
        song = re.search(r"/song/(?:[^/]+/)?(?:id)?([0-9]+)", path)
        if iid or song:
            return _apple_track(iid.group(0) if iid else song.group(1), "Apple Music link", cc)
        what = m.group(1).replace("-", " ") if (m := re.search(r"/(album|artist|playlist|music-video|station)/", path)) else "page"
        raise LookupError_(_NOT_A_SONG.format(what=f"an Apple Music {what} link"))
    if kind == "spotify":
        m = re.match(r"/(?:intl-[\w-]+/)?(?:embed/)?(track|album|playlist|artist|episode|show)/([A-Za-z0-9]*)", path)
        if m and m.group(1) == "track":
            if len(m.group(2)) == 22:
                return _spotify(m.group(2))
            raise LookupError_(_CUT_OFF.format(service="Spotify"))
        what = {"episode": "podcast", "show": "podcast"}.get(m.group(1), m.group(1)) if m else "page"
        raise LookupError_(_NOT_A_SONG.format(what=f"a Spotify {what} link"))
    if kind == "youtube":
        if host == "youtu.be":
            vid, is_video = path.strip("/").split("/")[0], True
        elif path.rstrip("/") == "/watch":
            vid, is_video = (query.get("v") or [""])[0], "list" not in query or bool(query.get("v"))
        elif m := re.match(r"/(?:shorts|embed|live|v)/([\w-]*)", path):
            vid, is_video = m.group(1), True
        else:
            vid, is_video = "", False
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid[:11]) and len(vid) >= 11:
            return _youtube(vid[:11])
        if is_video:
            raise LookupError_(_CUT_OFF.format(service="YouTube"))
        raise LookupError_(_NOT_A_SONG.format(what="a YouTube channel or playlist link"))
    raise LookupError_(_ONLY)


def _trim(url: str) -> str:
    """Drop punctuation, quotes and emoji stuck to the end of a pasted link ('…/song/123。', '“url”', 'url🔥')."""
    while url and url[-1] not in "/=&?#-_%+" and (unicodedata.category(url[-1])[0] == "P" or unicodedata.category(url[-1]) == "So"):
        url = url[:-1]
    return url


MAX_TEXT = 1000


def resolve(text: str) -> Match:
    try:
        return _resolve(text)
    except (UnicodeError, http.client.InvalidURL):  # characters no link can carry, e.g. a DEL copied along
        raise LookupError_("That link contains characters a link cannot have. Copy it again, or type 'Artist - Song'.") from None


def _resolve(text: str) -> Match:
    # Invisible marks (U+200B, U+200E, ...) and control characters out; runs of spaces made one.
    text = " ".join("".join(c for c in (text or "") if c.isspace() or unicodedata.category(c) not in ("Cf", "Cc")).split())
    if not text:
        raise LookupError_("Paste a song link, type 'Artist - Song', or upload an audio file.")
    if len(text) > MAX_TEXT:
        raise LookupError_("That is too long to be a song link or 'Artist - Song'.")
    if m := re.search(r"spotify:(track|album|playlist|artist|episode|show):([A-Za-z0-9]{22})", text):
        return _resolve_url(f"https://open.spotify.com/{m.group(1)}/{m.group(2)}")
    if found := list(_URL.finditer(text)):  # a full https:// link wins over a bare 'youtube.com' mentioned earlier
        m = next((x for x in found if re.match(r"https?://", x.group(0), re.I)), found[0])
        return _resolve_url(_trim(m.group(0)))
    if re.match(r"https?://", text, re.I) or re.fullmatch(r"(?:www\.)?(?:[\w-]+\.)+[a-z]{2,}/\S*", text, re.I):
        raise LookupError_(_ONLY)  # also 'soundcloud.com/...' pasted without https://
    if sep := next((s for s in (" - ", " – ", " — ") if s in text), None):
        a, b = (x.strip() for x in text.split(sep, 1))
        readings = [_Reading([(a, 1.0)], b), _Reading([(b, 1.0)], a)]  # "Artist - Song" or "Song - Artist"
        if _is_same(b) or _OTHER.match(b):  # or "Song - Remastered", "Song - Live at Wembley" (but "Kiss - Live Wire")
            return _find(readings + [_Reading([], text)], "search", free_text=text, named=False)
        return _find(readings, "search")
    # No clear split: these readings are guesses, so the artist does not count as named ("stand by me").
    readings = []
    if m := re.match(r"(.+)\s+by\s+(.+)$", text, re.I):  # the last " by ": "Stand by Me by Ben E. King"
        readings.append(_Reading([(m.group(2).strip(), 1.0)], m.group(1).strip()))
    for h in [h.start() for h in re.finditer(r"(?<=\w)[-–](?=\w)", text)][:2]:  # "Radiohead-Karma Police"
        a, b = text[:h].strip(), text[h + 1:].strip()
        readings += [_Reading([(a, 1.0)], b), _Reading([(b, 1.0)], a)]
    return _find(readings, "search", free_text=text, named=False)


def fetch_preview(m: Match) -> bytes:
    """The 30-second preview clip as MP3/AAC bytes (kept in memory only). Deezer's preview links expire after
    15 minutes (a visitor may take that long to confirm a match), so an expired one is asked for again."""
    try:
        return _get(m.preview_url, MAX_PREVIEW_BYTES)[1]
    except urllib.error.HTTPError as e:
        track = re.search(r"/track/([0-9]+)", m.page)
        if e.code not in (403, 410) or m.preview_from != "Deezer" or not track:
            raise
    d = _json(f"https://api.deezer.com/track/{track.group(1)}")
    if not d.get("preview"):
        raise LookupError_("Deezer no longer offers a preview of that song. Press Check to look for it again.")
    m.preview_url = d["preview"]
    return _get(m.preview_url, MAX_PREVIEW_BYTES)[1]
