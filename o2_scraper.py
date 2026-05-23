"""
[2/6] scraper.py
----------------
HLTV data collection layer.  Two transport methods:

  API (fast, no browser)   : eupeutro HLTV REST API for match result lists
  Selenium (Cloudflare OK) : undetected-chromedriver for per-map player stats

All responses are cached in SQLite so re-runs never re-fetch the same URL.

Public interface
----------------
  scraper = HLTVScraper()
  results  = scraper.get_team_results(team_id)          # list of match dicts
  ms_ids   = scraper.get_mapstats_ids(match_url)        # [(mapstats_id, slug), ...]
  stats    = scraper.get_map_stats(mapstats_id, slug)   # per-player stat rows
  profile  = scraper.get_team_profile(team_id)          # ranking / age / lineup
  scraper.close()
"""

import json
import logging
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from o1_config import (
    API_BASE, HLTV_BASE, CACHE_DB,
    CHROME_VER, PAGE_WAIT, API_DELAY, CACHE_TTL_H,
)

log = logging.getLogger(__name__)

# Serialise all SQLite writes across threads. WAL mode allows concurrent reads;
# this lock prevents two browsers from colliding on writes at the same instant.
_DB_WRITE_LOCK = threading.Lock()

# undetected_chromedriver patches the chromedriver binary on first launch.
# On Windows that rename fails if two threads do it simultaneously, so only
# one thread may call uc.Chrome() at a time. After the first launch completes
# the binary is already patched and subsequent launches are instant.
_CHROME_LAUNCH_LOCK = threading.Lock()


class HLTVScraper:

    def __init__(
        self,
        cache_db: Path = CACHE_DB,
        chrome_version: int = CHROME_VER,
        headless: bool = False,
    ):
        self._chrome_version = chrome_version
        self._headless       = headless
        self._driver         = None          # lazy-loaded
        cache_db.parent.mkdir(exist_ok=True)
        self._conn = self._init_db(cache_db)

    # ------------------------------------------------------------------
    # SQLite cache
    # ------------------------------------------------------------------

    def _init_db(self, path: Path) -> sqlite3.Connection:
        # check_same_thread=False + WAL mode allow multiple scrapers to share
        # one cache file safely — WAL serialises writers without blocking readers.
        conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS page_cache (
                   url      TEXT PRIMARY KEY,
                   content  TEXT,
                   fetched  REAL
               )"""
        )
        conn.commit()
        return conn

    def _cached(self, url: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT content, fetched FROM page_cache WHERE url = ?", (url,)
        ).fetchone()
        if row and (time.time() - row[1]) / 3600 < CACHE_TTL_H:
            return row[0]
        return None

    def _store(self, url: str, content: str):
        with _DB_WRITE_LOCK:
            self._conn.execute(
                "INSERT OR REPLACE INTO page_cache VALUES (?, ?, ?)",
                (url, content, time.time()),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Selenium driver (lazy-init)
    # ------------------------------------------------------------------

    def _get_driver(self):
        if self._driver is not None:
            return self._driver
        import undetected_chromedriver as uc
        opts = uc.ChromeOptions()
        if self._headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        with _CHROME_LAUNCH_LOCK:
            self._driver = uc.Chrome(options=opts, version_main=self._chrome_version)
        log.info("Chrome %d launched", self._chrome_version)
        return self._driver

    def _fetch_page(self, url: str, wait: float = PAGE_WAIT) -> str:
        """Return page HTML, using cache when fresh."""
        cached = self._cached(url)
        if cached:
            log.debug("cache hit: %s", url)
            return cached

        log.info("fetching %s", url)
        driver = self._get_driver()
        driver.get(url)
        time.sleep(wait)

        if "Just a moment" in driver.title:
            log.warning("Cloudflare challenge on %s — waiting extra 10s", url)
            time.sleep(10)

        html = driver.page_source
        if "Just a moment" not in driver.title:
            self._store(url, html)
        return html

    # ------------------------------------------------------------------
    # eupeutro REST API helpers (no Selenium needed)
    # ------------------------------------------------------------------

    def _api_get(self, path: str) -> Optional[dict]:
        url = f"{API_BASE}{path}"
        cached = self._cached(url)
        if cached:
            return json.loads(cached)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self._store(url, json.dumps(data))
            time.sleep(API_DELAY)
            return data
        except Exception as exc:
            log.warning("API GET %s failed: %s", url, exc)
            return None

    # ------------------------------------------------------------------
    # Public API: team results (eupeutro API)
    # ------------------------------------------------------------------

    def get_team_results(self, team_id: int, limit: int = 100) -> List[dict]:
        """
        Return recent match results for a team.
        Each dict has: matchId, matchDate, matchUrl, team1Name, team2Name,
                       team1Score, team2Score, matchWon, matchType
        """
        data = self._api_get(f"/teams/{team_id}/results?limit={limit}")
        if data is None:
            return []
        results = data if isinstance(data, list) else data.get("results", [])
        out = []
        for r in results:
            try:
                out.append({
                    "matchId":    str(r.get("matchId",    r.get("match_id",    ""))),
                    "matchDate":  r.get("matchDate",      r.get("match_date",  "")),
                    "matchUrl":   r.get("matchUrl",       r.get("match_url",   "")),
                    "team1Name":  r.get("team1Name",      r.get("team1_name",  "")),
                    "team2Name":  r.get("team2Name",      r.get("team2_name",  "")),
                    "team1Score": int(r.get("team1Score", r.get("team1_score", 0))),
                    "team2Score": int(r.get("team2Score", r.get("team2_score", 0))),
                    "matchWon":   bool(r.get("matchWon",  r.get("match_won",   False))),
                    "matchType":  r.get("matchType",      r.get("match_type",  "bo3")),
                })
            except Exception:
                continue
        return out

    def get_team_profile(self, team_id: int) -> dict:
        """Return team ranking, age, lineup from eupeutro API."""
        data = self._api_get(f"/teams/{team_id}/profile")
        if data is None:
            return {}
        profile = data.get("teamProfile", data)
        lineup  = profile.get("lineup", [])
        return {
            "worldRanking":     int(profile.get("worldRanking",       profile.get("world_ranking",        999))),
            "valveRanking":     int(profile.get("valveRanking",       profile.get("valve_ranking",        999))),
            "weeksInTop30":     int(profile.get("weeksInTop30ForCore", profile.get("weeks_in_top30", 0))),
            "averagePlayerAge": float(profile.get("averagePlayerAge",  profile.get("average_player_age", 25.0))),
            "playerIds":        [str(p.get("id", p.get("player_id", ""))) for p in lineup],
        }

    # ------------------------------------------------------------------
    # Public API: mapstats IDs for a match (Selenium)
    # ------------------------------------------------------------------

    def get_mapstats_ids(self, match_url: str) -> List[Tuple[str, str]]:
        """
        Load the HLTV match page and extract all /stats/matches/mapstatsid/ links.
        Returns list of (mapstats_id, slug) tuples.
        """
        html = self._fetch_page(match_url)
        soup = BeautifulSoup(html, "html.parser")
        seen, results = set(), []
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            m = re.search(r"/stats/matches/mapstatsid/(\d+)/([^\"?#\s]+)", href)
            if m:
                mid, slug = m.group(1), m.group(2)
                if mid not in seen:
                    seen.add(mid)
                    results.append((mid, slug))
        return results

    # ------------------------------------------------------------------
    # Public API: per-player stats for one map (Selenium)
    # ------------------------------------------------------------------

    def get_map_stats(self, mapstats_id: str, slug: str) -> List[dict]:
        """
        Scrape the mapstatsid page and return a list of per-player stat rows.

        Each dict:
          player_name, team_slot (0 or 1),
          opening_kd, opening_kills, opening_deaths,
          multi_kills, kast_pct, adr, rating,
          clutches, flash_assists, hs_kills
        """
        url  = f"{HLTV_BASE}/stats/matches/mapstatsid/{mapstats_id}/{slug}"
        html = self._fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")

        players = []
        tables  = soup.find_all("table")

        for team_slot, table in enumerate(tables[:2]):
            rows = table.find_all("tr")
            if not rows:
                continue

            headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
            idx     = _col_indices(headers)

            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if not cells or len(cells) < 3:
                    continue
                p = _parse_player_row(cells, idx, team_slot)
                if p:
                    players.append(p)

        return players

    # ------------------------------------------------------------------
    # Public API: aggregate player stats page (Selenium)
    # ------------------------------------------------------------------

    def get_player_stats(self, player_id: int, slug: str) -> dict:
        """
        Scrape /stats/players/{id}/{slug} and return a dict of aggregate stats.
        Fields: total_kills, headshot_pct, kd_ratio, adr, grenade_dmg,
                kills_per_round, assists_per_round, deaths_per_round,
                saved_by_teammate, saved_teammates, impact_rating
        """
        url  = f"{HLTV_BASE}/stats/players/{player_id}/{slug}"
        html = self._fetch_page(url)
        soup = BeautifulSoup(html, "html.parser")

        stats = {}
        for row in soup.select(".stats-row"):
            spans = row.find_all("span")
            if len(spans) >= 2:
                key = _normalize_key(spans[0].get_text(strip=True))
                val = _parse_stat_value(spans[1].get_text(strip=True))
                stats[key] = val
        return stats

    # ------------------------------------------------------------------
    # Convenience: bulk-fetch player stats for a team's lineup
    # ------------------------------------------------------------------

    def get_team_player_stats(self, team_id: int) -> List[dict]:
        """Fetch aggregate stats for every player in the team's current lineup."""
        profile    = self.get_team_profile(team_id)
        player_ids = profile.get("playerIds", [])
        results    = []
        for pid in player_ids:
            pdata = self._api_get(f"/players/{pid}/profile")
            if not pdata:
                continue
            nickname = (
                pdata.get("nickname") or
                pdata.get("playerProfile", {}).get("nickname") or
                str(pid)
            )
            slug  = nickname.lower().replace(" ", "-")
            stats = self.get_player_stats(int(pid), slug)
            stats["player_id"] = pid
            stats["nickname"]  = nickname
            results.append(stats)
            time.sleep(1)
        return results

    # ------------------------------------------------------------------

    def close(self):
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_STAT_KEY_MAP = {
    "total kills":                "total_kills",
    "headshot %":                 "headshot_pct",
    "k/d ratio":                  "kd_ratio",
    "damage / round":             "adr",
    "grenade dmg / round":        "grenade_dmg",
    "kills / round":              "kills_per_round",
    "assists / round":            "assists_per_round",
    "deaths / round":             "deaths_per_round",
    "saved by teammate / round":  "saved_by_teammate",
    "saved teammates / round":    "saved_teammates",
    "impact rating":              "impact_rating",
    "maps played":                "maps_played",
    "rounds played":              "rounds_played",
}


def _normalize_key(raw: str) -> str:
    return _STAT_KEY_MAP.get(raw.lower().strip(), raw.lower().replace(" ", "_").replace("/", "_per_"))


def _parse_stat_value(raw: str) -> float:
    s = raw.strip().replace("%", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _col_indices(headers: List[str]) -> Dict[str, int]:
    """Map semantic column names to indices, tolerating varying header formats."""
    idx: Dict[str, int] = {}
    for i, h in enumerate(headers):
        h_low = h.lower()
        if i == 0:
            idx["player"] = 0
        elif "op.k-d" in h_low or "opkd" in h_low:
            idx.setdefault("opening_kd", i)
        elif "mks" in h_low or "multi" in h_low:
            idx.setdefault("multi_kills", i)
        elif "kast" in h_low and "ekast" not in h_low:
            idx.setdefault("kast", i)
        elif "a(f)" in h_low or "flash" in h_low:
            idx.setdefault("flash_assists", i)
        elif "1vsx" in h_low or "clutch" in h_low:
            idx.setdefault("clutches", i)
        elif "k(hs)" in h_low:
            idx.setdefault("hs_kills", i)
        elif h_low in ("adr", "eadr"):
            idx.setdefault("adr", i)
        elif "rating" in h_low:
            idx["rating"] = i   # keep last "rating" column
    return idx


def _parse_opening_kd(raw: str) -> Tuple[float, float, float]:
    """Parse '4 : 0' or '4:0' → (ratio, kills, deaths)."""
    m = re.match(r"(\d+)\s*:\s*(\d+)", raw)
    if m:
        k, d = float(m.group(1)), float(m.group(2))
        return k / max(d, 1.0), k, d
    try:
        v = float(raw)
        return v, v, 0.0
    except ValueError:
        return 1.0, 0.0, 0.0


def _pct(raw: str) -> float:
    return float(raw.replace("%", "").strip()) if raw else 0.0


def _parse_player_row(cells: List[str], idx: Dict[str, int], team_slot: int) -> Optional[dict]:
    try:
        player_name = cells[idx.get("player", 0)]
        if not player_name or player_name.lower() in ("player", "name", ""):
            return None

        ok_ratio, ok_kills, ok_deaths = _parse_opening_kd(
            cells[idx["opening_kd"]] if "opening_kd" in idx else "0:0"
        )

        def _safe(key: str, cast=float, default=0.0):
            if key in idx and idx[key] < len(cells):
                try:
                    return cast(cells[idx[key]])
                except (ValueError, TypeError):
                    pass
            return default

        return {
            "player_name":    player_name,
            "team_slot":      team_slot,
            "opening_kd":     ok_ratio,
            "opening_kills":  ok_kills,
            "opening_deaths": ok_deaths,
            "multi_kills":    _safe("multi_kills"),
            "kast_pct":       _pct(cells[idx["kast"]]) if "kast" in idx and idx["kast"] < len(cells) else 0.0,
            "adr":            _safe("adr"),
            "rating":         _safe("rating", default=1.0),
            "clutches":       _safe("clutches"),
            "flash_assists":  _safe("flash_assists"),
            "hs_kills":       _parse_hs(cells[idx["hs_kills"]]) if "hs_kills" in idx and idx["hs_kills"] < len(cells) else 0.0,
        }
    except Exception:
        return None


def _parse_hs(raw: str) -> float:
    """Parse '24(13)' → 13 headshot kills."""
    m = re.search(r"\((\d+)\)", raw)
    if m:
        return float(m.group(1))
    try:
        return float(raw)
    except ValueError:
        return 0.0
