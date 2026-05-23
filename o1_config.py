"""
[1/6] config.py
---------------
Central configuration for all IEM Cologne 2026 pipeline modules.
Import from here instead of scattering constants across files.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
CACHE_DB  = DATA_DIR / "hltv_cache.db"

# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

API_BASE    = "https://hltv-json-api.fly.dev"
HLTV_BASE   = "https://www.hltv.org"
CHROME_VER  = 148     # pin to locally installed Chrome version
PAGE_WAIT   = 8.0     # seconds to wait after page load (Cloudflare settle time)
API_DELAY   = 1.2     # seconds between eupeutro REST calls
CACHE_TTL_H = 72      # hours before cached HTML is re-fetched
N_BROWSERS  = 4       # parallel Chrome instances during data collection

# CS2 replaced CS:GO on this date — all pre-CS2 matches are excluded from training.
CS2_LAUNCH_DATE = datetime(2023, 9, 27)

# Only scrape per-player mapstats for matches on or after this date.
MAPSTATS_START_DATE = CS2_LAUNCH_DATE

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

ROLLING_WINDOWS: List[int] = [5, 10, 20]
ELO_K:   int   = 32
ELO_INIT: int  = 1000

# ---------------------------------------------------------------------------
# Time-based train / val / test split
# ---------------------------------------------------------------------------

TRAIN_END_DATE = datetime(2025, 12, 31)
VAL_END_DATE   = datetime(2026, 2, 28)

# ---------------------------------------------------------------------------
# Teams used to seed the training data collector
# ---------------------------------------------------------------------------

SEED_TEAM_IDS: List[int] = [
    # Stage 3 seeds
    9565,   # Team Vitality
    12467,  # PARIVISION
    11283,  # Falcons
    8297,   # FURIA
    4608,   # Natus Vincere
    11861,  # Aurora
    10503,  # MOUZ
    6248,   # The MongolZ
    # Stage 2 seeds
    7020,   # Team Spirit
    5995,   # G2 Esports
    11811,  # Monte
    4773,   # paiN Gaming
    6665,   # Astralis
    13286,  # FUT Esports
    9996,   # 9z Team
    12468,  # Legacy
    # Stage 1 seeds
    9928,   # GamerLegion
    12394,  # BetBoom
    7175,   # Heroic
    12376,  # M80
    8113,   # Sharks
    9215,   # MIBR
    4863,   # TYLOO
    13486,  # THUNDER dOWNUNDER
    7532,   # BIG
    11241,  # B8
    10577,  # SINNERS
    6673,   # NRG Esports
    11571,  # Gaimin Gladiators
    5973,   # Team Liquid
    8840,   # Lynn Vision
    12774,  # FlyQuest
]

# ---------------------------------------------------------------------------
# IEM Cologne 2026 roster — 32 teams, 3 Swiss stages + Playoffs
#
# Stage 3  (8 teams, Swiss BO3)  — top seeded, enter at Stage 3
# Stage 2  (8 teams, Swiss BO1)  — enter at Stage 2
# Stage 1  (16 teams, Swiss BO1) — enter at Stage 1 (Opening)
# Playoffs (8 teams, single-elim BO3)
# ---------------------------------------------------------------------------

STAGE3_TEAMS: List[str] = [
    "Vitality", "PARIVISION", "Falcons", "FURIA",
    "Natus Vincere", "Aurora", "MOUZ", "The MongolZ",
]

STAGE2_TEAMS: List[str] = [
    "Spirit", "G2", "Monte", "paiN",
    "Astralis", "FUT", "9z", "Legacy",
]

STAGE1_TEAMS: List[str] = [
    "GamerLegion", "NRG",
    "B8", "TYLOO",
    "HEROIC", "Sharks",
    "BetBoom", "Gaimin Gladiators",
    "BIG", "Liquid",
    "M80", "Lynn Vision",
    "MIBR", "THUNDER dOWNUNDER",
    "SINNERS", "FlyQuest",
]

ALL_TEAMS: List[str] = STAGE3_TEAMS + STAGE2_TEAMS + STAGE1_TEAMS

TEAM_IDS: Dict[str, int] = {
    # Stage 3
    "Vitality":           9565,
    "PARIVISION":         12467,
    "Falcons":            11283,
    "FURIA":              8297,
    "Natus Vincere":      4608,
    "Aurora":             11861,
    "MOUZ":               10503,
    "The MongolZ":        6248,
    # Stage 2
    "Spirit":             7020,
    "G2":                 5995,
    "Monte":              11811,
    "paiN":               4773,
    "Astralis":           6665,
    "FUT":                13286,
    "9z":                 9996,
    "Legacy":             12468,
    # Stage 1
    "GamerLegion":        9928,
    "BetBoom":            12394,
    "HEROIC":             7175,
    "M80":                12376,
    "Sharks":             8113,
    "MIBR":               9215,
    "TYLOO":              4863,
    "THUNDER dOWNUNDER":  13486,
    "BIG":                7532,
    "B8":                 11241,
    "SINNERS":            10577,
    "NRG":                6673,
    "Gaimin Gladiators":  11571,
    "Liquid":             5973,
    "Lynn Vision":        8840,
    "FlyQuest":           12774,
}

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------

PT_HIDDEN:    int   = 256
PT_N_BLOCKS:  int   = 4
PT_DROPOUT:   float = 0.3
PT_LR:        float = 1e-3
PT_WD:        float = 1e-4
PT_BATCH:     int   = 512
PT_EPOCHS:    int   = 300
PT_PATIENCE:  int   = 30
PT_T0:        int   = 25
PT_T_MULT:    int   = 2

XGB_ESTIMATORS:    int   = 600
XGB_MAX_DEPTH:     int   = 5
XGB_LR:            float = 0.03
XGB_SUBSAMPLE:     float = 0.8
XGB_COLSAMPLE:     float = 0.8
XGB_MIN_CHILD:     int   = 5
XGB_GAMMA:         float = 0.1
XGB_EARLY_STOP:    int   = 40

BLEND_STEPS: int = 21   # grid points for convex blend weight search

RANDOM_SEED: int = 42
