"""Central configuration: paths, tournament format, model + simulation parameters.

Everything that a maintainer might want to tweak (the field, the format, the number
of simulations, model hyper-parameters) lives here or in :mod:`iemcs.teams`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(ROOT, "outputs")
ARTIFACTS_DIR = os.path.join(OUTPUTS_DIR, "artifacts")

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, ARTIFACTS_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# Data source — Valve Regional Standings (official, CS2-only)
# --------------------------------------------------------------------------- #
VRS_REPO_URL = "https://github.com/ValveSoftware/counter-strike_regional_standings.git"
# Override with $VRS_REPO_DIR (e.g. a fast /tmp clone) for development.
VRS_REPO_DIR = os.environ.get("VRS_REPO_DIR", os.path.join(RAW_DIR, "vrs_repo"))

# CS2 launched 2023-09-27. The VRS data only covers the CS2 era, but we still guard
# against any stray earlier rows to honour the "CS2 matches only" requirement.
CS2_CUTOFF_DATE = "2023-09-27"

# Optional processed-data cache paths (the notebook builds these in memory)
MATCHES_CSV = os.path.join(PROCESSED_DIR, "matches.csv")
VRS_RATINGS_CSV = os.path.join(PROCESSED_DIR, "vrs_ratings.csv")
REGIONS_CSV = os.path.join(PROCESSED_DIR, "regions.csv")
FEATURES_CSV = os.path.join(PROCESSED_DIR, "features.csv")
TEAMS_CSV = os.path.join(DATA_DIR, "teams.csv")

# Fitted artefacts
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.joblib")
CONTEXT_PATH = os.path.join(ARTIFACTS_DIR, "context.joblib")
BACKTEST_PATH = os.path.join(ARTIFACTS_DIR, "backtest.json")

# Snapshot used as "today" for the prediction (latest pre-event VRS update)
CURRENT_SNAPSHOT = "2026_05_04"

# --------------------------------------------------------------------------- #
# Tournament format (IEM Cologne 2026 — confirmed via Liquipedia + Wikipedia)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StageFormat:
    name: str
    n_teams: int = 16
    n_advance: int = 8
    wins_to_advance: int = 3
    losses_to_eliminate: int = 3
    all_bo3: bool = False           # Stage 3 is entirely Bo3
    # In Bo1 stages, only matches that would advance OR eliminate a team are Bo3.


STAGE1 = StageFormat("Stage 1", all_bo3=False)
STAGE2 = StageFormat("Stage 2", all_bo3=False)
STAGE3 = StageFormat("Stage 3", all_bo3=True)
SWISS_STAGES = (STAGE1, STAGE2, STAGE3)

PLAYOFF_TEAMS = 8           # single-elimination
PLAYOFF_FINAL_BO5 = True    # grand final is Bo5, all other rounds Bo3

# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
N_SIMS = 50_000
RANDOM_SEED = 42
N_JOBS = -1                 # joblib parallelism for the Monte-Carlo
BOOTSTRAP_CI = 0.95         # confidence level for reported intervals

# --------------------------------------------------------------------------- #
# Rating models
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EloConfig:
    base_rating: float = 1500.0
    k_factor: float = 32.0
    scale: float = 400.0        # logistic denominator
    # Down-weight matches as they age (half-life in days) when seeding "current" form.
    decay_half_life_days: float = 240.0


@dataclass(frozen=True)
class GlickoConfig:
    base_rating: float = 1500.0
    base_rd: float = 350.0
    base_vol: float = 0.06
    tau: float = 0.5            # system constant (smaller -> steadier volatility)
    scale: float = 173.7178     # Glicko-2 <-> Elo scale


ELO = EloConfig()
GLICKO = GlickoConfig()

# --------------------------------------------------------------------------- #
# Match / series model
# --------------------------------------------------------------------------- #
# Most VRS-counted pro encounters are Bo3, so we treat the trained "win an
# encounter" probability as a Bo3 probability and invert it to a per-map prob to
# derive Bo1 and Bo5 series odds in the simulator. See iemcs.series.
TRAIN_ENCOUNTER_FORMAT = "bo3"

# Shrink per-map probability toward 0.5 to curb overconfidence (0 = no shrink).
PROB_SHRINK = 0.04

# Recency weighting: down-weight older matches when fitting the learners / stacker /
# calibrator / cross-region blend, so the model tracks the current meta rather than
# stale 2024 form. Half-life in days; set to None to disable.
RECENCY_HALF_LIFE_DAYS = 365

# Feature columns fed to the gradient-boosted / linear base learners.
FEATURE_COLUMNS = [
    "elo_diff",
    "glicko_diff",
    "glicko_rd_sum",
    "vrs_diff",
    "sos_diff",
    "form_diff",
    "h2h_diff",
    "activity_diff",
    "streak_diff",
    "roster_stab_diff",
    "same_region",
]

# Ensemble: probabilities blended by a logistic meta-learner trained out-of-fold.
ENSEMBLE_MEMBERS = ("elo", "glicko", "vrs", "logreg", "rf", "gbm")
