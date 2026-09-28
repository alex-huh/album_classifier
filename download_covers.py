"""Download album art for every (artist, album) in RPM Gold Dot List.xlsx.

Only albums with copies in the store or warehouse (columns 3 and 4) are
fetched. Looks up each album as a MusicBrainz release group, then fetches its front
cover from the Cover Art Archive. Re-running skips albums already downloaded.
Albums with no match or no cover are logged to training_images/missing.csv.
"""
import csv
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

XLSX = Path("RPM Gold Dot List.xlsx")
OUT_DIR = Path("training_images")
MISSING_CSV = OUT_DIR / "missing.csv"
MB_API = "https://musicbrainz.org/ws/2/"
CAA_FRONT = "https://coverartarchive.org/release-group/{mbid}/front-500"
MIN_SCORE = 80  # MusicBrainz search score (0-100) needed to accept a match
MIN_TITLE_SIMILARITY = 0.85  # normalized title similarity needed to accept a match
DOWNLOAD_WORKERS = 8  # parallel Cover Art Archive downloads (not rate limited)
TYPE_PREFERENCE = {"Album": 0, "EP": 1, None: 2}  # anything else (Single, ...) ranks last
HEADERS = {"User-Agent": "album_classifier/0.1 (alexalhuh@gmail.com)"}

session = requests.Session()
session.headers.update(HEADERS)
_last_mb_call = 0.0


def lucene_escape(text):
    return re.sub(r'([+\-&|!(){}\[\]^"~*?:\\/])', r"\\\1", text)


def normalize(text):
    text = text.lower().replace("&", "and")
    text = re.sub(r"^the\s+", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def title_similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def titles_match(mb_title, album):
    """Close match, or the album equals the MB title minus a subtitle/qualifier
    (e.g. "Gold" vs "Gold: Greatest Hits", "Thriller" vs "Thriller (25th Anniversary)")."""
    if title_similarity(mb_title, album) >= MIN_TITLE_SIMILARITY:
        return True
    main_title = re.split(r"\s*[:(\[]|\s+-\s+", mb_title)[0]
    return bool(normalize(main_title)) and normalize(main_title) == normalize(album)


def rank_candidates(groups, album):
    """Keep release groups whose title matches, best first (albums before singles)."""
    good = [g for g in groups if titles_match(g["title"], album)]
    good.sort(key=lambda g: (
        TYPE_PREFERENCE.get(g.get("primary-type"), 3),
        -title_similarity(g["title"], album),
    ))
    return [g["id"] for g in good]


_artist_groups_cache = {}


def artist_release_groups(artist):
    """All release groups of the top few artists matching `artist` (name or alias)."""
    if artist not in _artist_groups_cache:
        artists = mb_get({"query": lucene_escape(artist), "limit": 3}, "artist/")["artists"]
        groups = []
        for a in artists:
            if int(a.get("score", 0)) < 60:
                continue
            for offset in range(0, 500, 100):
                page = mb_get({"artist": a["id"], "limit": 100, "offset": offset}, "release-group")
                groups += page.get("release-groups", [])
                if offset + 100 >= page.get("release-group-count", 0):
                    break
        _artist_groups_cache[artist] = groups
    return _artist_groups_cache[artist]


def in_stock(count):
    """False only when the count is definitely zero; unclear counts ("?", "4+?") are kept."""
    text = str(count).strip()
    if re.fullmatch(r"\d+(\.0+)?", text):
        return float(text) > 0
    return True


def load_albums(stocked_only=True):
    """(artist, album) rows from the sheet; an album counts as stocked if any of its rows is."""
    df = pd.read_excel(XLSX, usecols=[0, 1, 2, 3])
    df.columns = ["artist", "album", "store", "warehouse"]
    df = df.dropna(subset=["artist", "album"])
    df["artist"] = df["artist"].astype(str).str.strip()
    df["album"] = df["album"].astype(str).str.strip()
    df["stocked"] = df["store"].map(in_stock) | df["warehouse"].map(in_stock)
    df = df.groupby(["artist", "album"], sort=False, as_index=False)["stocked"].any()
    return df[df["stocked"] == stocked_only][["artist", "album"]]


def safe_filename(text):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text).strip(" .")


def mb_get(params, endpoint="release-group/"):
    """GET the MusicBrainz API, respecting the 1 req/sec rate limit."""
    global _last_mb_call
    for attempt in range(5):
        wait = 1.1 - (time.monotonic() - _last_mb_call)
        if wait > 0:
            time.sleep(wait)
        _last_mb_call = time.monotonic()
        resp = session.get(MB_API + endpoint, params={**params, "fmt": "json"}, timeout=30)
        if resp.status_code == 503:  # throttled
            time.sleep(2 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("MusicBrainz kept returning 503")


def find_release_groups(artist, album):
    """Return candidate release-group MBIDs, best match first.

    Candidates must have a title close to `album`; albums are preferred over
    singles/EPs so e.g. a self-titled album doesn't pick up a same-named single.
    """
    queries = [
        f'releasegroup:"{lucene_escape(album)}" AND artist:"{lucene_escape(artist)}"',
        f"releasegroup:({lucene_escape(album)}) AND artist:({lucene_escape(artist)})",
    ]
    for q in queries:
        groups = mb_get({"query": q, "limit": 25}).get("release-groups", [])
        groups = [g for g in groups if int(g.get("score", 0)) >= MIN_SCORE]
        if mbids := rank_candidates(groups, album):
            return mbids
    # Fallback: find the artist by name/alias and match against their whole catalogue.
    return rank_candidates(artist_release_groups(artist), album)


def download_cover(mbid, dest):
    resp = session.get(CAA_FRONT.format(mbid=mbid), timeout=60, allow_redirects=True)
    if resp.status_code != 200:
        return False
    dest.write_bytes(resp.content)
    return True


def fetch_first_cover(mbids, dest):
    """Try candidates in order until one has front cover art."""
    try:
        return "ok" if any(download_cover(m, dest) for m in mbids[:3]) else "no cover art"
    except Exception as e:  # keep going on network hiccups
        return f"error: {e}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT_DIR.mkdir(exist_ok=True)
    df = load_albums()

    missing = []
    downloads = {}  # (artist, album) -> future
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
        # MusicBrainz lookups run serially (rate limited); downloads run in parallel.
        for i, (artist, album) in enumerate(df.itertuples(index=False), 1):
            dest = OUT_DIR / f"{safe_filename(f'{artist} - {album}')}.jpg"
            if dest.exists():
                continue
            try:
                mbids = find_release_groups(artist, album)
            except Exception as e:
                mbids, reason = [], f"error: {e}"
            else:
                reason = "no match"
            if mbids:
                downloads[(artist, album)] = pool.submit(fetch_first_cover, mbids, dest)
            else:
                missing.append((artist, album, reason))
            print(f"[{i}/{len(df)}] {artist} - {album}: "
                  f"{'queued' if mbids else reason}", flush=True)

    for (artist, album), future in downloads.items():
        if future.result() != "ok":
            missing.append((artist, album, future.result()))

    with MISSING_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["artist", "album", "reason"])
        writer.writerows(missing)
    print(f"Done. {len(df) - len(missing)} covers saved; "
          f"{len(missing)} albums without art listed in {MISSING_CSV}")


if __name__ == "__main__":
    main()
