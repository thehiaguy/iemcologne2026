# IEM Cologne 2026 — CS2 ML Predictor

A machine-learning system that trains a match outcome classifier on historical CS2 data from HLTV and runs Monte Carlo simulation to predict bracket results for IEM Cologne 2026.

---

## How it works

### Data collection — `scraper.py`
- Fetches match result lists via the [eupeutro HLTV REST API](https://github.com/eupeutro/hltv-api)
- Scrapes per-map player stats directly from HLTV using Selenium + undetected-chromedriver (bypasses Cloudflare)
- Caches all responses in SQLite (`data/hltv_cache.db`) with a 72-hour TTL so re-runs never re-fetch

### Feature engineering — `features.py`
- Computes rolling averages for every team at the exact timestamp of each match (no lookahead)
- Maintains a live ELO rating updated chronologically across the full dataset
- Builds head-to-head records, differential features, and team profile data (ranking, average age)

### Model training — `train.py`
- Time-based train / val / test split (train ≤ Jun 2024, val Jul–Dec 2024, test 2025+)
- **PyTorch ResNet MLP** trained on RTX 5080 with mixed precision (CUDA 12.8)
- **XGBoost** trained on CPU
- **Ensemble** — convex blend weight grid-searched on validation AUC

### Simulation — `simulate.py`
- Loads trained artifacts and fetches current team stats
- Assembles feature vectors matching the training schema for any matchup
- Simulates Swiss stages (same-record pairing, BO1) and single-elimination playoffs (seeded BO3) via Monte Carlo

### Visualisation — `analysis.ipynb`
Jupyter notebook covering: data distribution, feature importance, ELO curves, ROC + calibration curves, and tournament probability heatmaps.

---

## Features engineered

Every feature is computed independently for each team at match time — no future data leaks.

| Category | Features |
|---|---|
| Firepower | `avg_rating` (HLTV Rating 3.0), `avg_adr`, `avg_kast` |
| Dueling | `avg_opening_kd`, `avg_opening_kills` |
| Impact | `avg_multi_kills`, `avg_clutches`, `avg_flash_assists`, `avg_hs_kills` |
| Consistency | `rating_std`, `adr_std` |
| Form | `win_rate`, `form_last5` |
| ELO | `t1_elo`, `t2_elo`, `elo_diff` |
| Ranking | `world_ranking`, `valve_ranking`, `weeks_top30`, `avg_age` |
| Head-to-head | `h2h_win_rate`, `h2h_total`, `h2h_last5_wins` |
| Differentials | `diff_<metric>_<window>` (team1 − team2 for every rolling metric) |

All rolling metrics are computed at three windows: **w5**, **w10**, **w20**.

---

## Tournament format

```
Opener Stage      8 Challengers   Swiss BO1 (3W/3L)   top 4 advance
Contender Stage  16 teams         Swiss BO1 (3W/3L)   top 8 advance
Legends Stage    16 teams         Swiss BO1 (3W/3L)   top 8 advance
Champions Stage   8 teams         Single-elimination BO3 (seeded 1v8, 2v7, ...)
```

---

## Setup

**Requirements:** Python 3.10+ (developed on Python 3.14 / Windows 11), CUDA optional

```powershell
# Create and activate venv
python -m venv venv
.\venv\Scripts\Activate.ps1

# If PowerShell blocks activation:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Install dependencies
# lxml must be binary-only on Python 3.14+
pip install lxml --only-binary :all:
pip install hltv-async-api --no-deps
pip install -r requirements.txt

# PyTorch with CUDA 12.8 (RTX 5080 / CUDA 13.x driver compatible)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Chrome:** Selenium uses your locally installed Google Chrome. The scraper pins `version_main=148` in `config.py` — update it to match your installed Chrome version.

---

## Usage

### 1. Train the model

```powershell
python o5_train.py
```

Pulls match history for all teams in `SEED_TEAM_IDS`, engineers features, trains both models, evaluates on the held-out test set, and saves all artifacts.

**Outputs:**
```
models/
  pytorch_model.pt      ResNet MLP state dict
  xgb_model.json        XGBoost booster
  scaler.pkl            StandardScaler
  feature_cols.pkl      ordered feature list
  elo.pkl               EloRatings object
  blend_weight.pkl      ensemble blend float
  input_dim.pkl         model input dimension
data/
  raw_matches.parquet
  training_data.parquet
  feature_importance.csv
```

### 2. Run the tournament simulation

```powershell
python o6_simulate.py --tournament --n 50000
```

Output example:
```
==============================================================================
  IEM COLOGNE 2026  --  Ensemble MC (50,000 simulations)
==============================================================================
Team                   Champion    2nd   Top4   Top8   ELeg   ECon  EOpen
------------------------------------------------------------------------------
Team Vitality            22.4%   14.1%  37.2%  61.8%  38.2%   0.0%  0.0%
Team Spirit              18.7%   12.3%  33.4%  58.1%  41.9%   0.0%  0.0%
...
```

Results are also saved to `data/tournament_odds.csv`.

### 3. Head-to-head matchup

```powershell
python o6_simulate.py --match "Team Vitality" "Team Spirit" --n 10000
```

### 4. Force-refresh team stats (re-opens Chrome)

```powershell
python o6_simulate.py --tournament --fetch-stats --n 50000
```

### 5. List the registered team roster

```powershell
python o6_simulate.py --list-teams
```

### 6. Open the analysis notebook

```powershell
jupyter lab analysis.ipynb
```

---

## Project structure

```
iemcologne/
├── o1_config.py           All shared constants (paths, team IDs, hyperparams)
├── o2_scraper.py          HLTVScraper — eupeutro API + Selenium
├── o3_features.py         EloRatings, rolling features, H2H, build_training_data
├── o4_neural_net.py       ResBlock, MatchPredictor, train_pytorch, pytorch_predict
├── o5_train.py            Full training pipeline — run this first
├── o6_simulate.py         Monte Carlo Swiss + bracket simulation
├── analysis.ipynb         Data exploration and visualisation notebook
├── smoke_test.py          Quick sanity check for both data sources
├── requirements.txt
├── data/
│   ├── hltv_cache.db          SQLite response cache (auto-created)
│   ├── raw_matches.parquet
│   ├── training_data.parquet
│   ├── feature_importance.csv
│   ├── current_team_stats.pkl
│   └── tournament_odds.csv
└── models/
    ├── pytorch_model.pt
    ├── xgb_model.json
    ├── scaler.pkl
    ├── feature_cols.pkl
    ├── elo.pkl
    ├── blend_weight.pkl
    └── input_dim.pkl
```

---

## Configuration

All constants live in `o1_config.py`. Key ones to adjust:

| Constant | Default | Description |
|---|---|---|
| `SEED_TEAM_IDS` | 20 top teams | HLTV team IDs to pull training data from |
| `TRAIN_END_DATE` | `2024-06-30` | Training cutoff |
| `VAL_END_DATE` | `2024-12-31` | Validation cutoff (test = everything after) |
| `ROLLING_WINDOWS` | `[5, 10, 20]` | Match windows for rolling features |
| `ELO_K` | `32` | ELO K-factor |
| `CHROME_VER` | `148` | Installed Chrome version (must match) |
| `LEGENDS_TEAMS` | 8 teams | Seeded directly into Legends stage |
| `CONTENDER_TEAMS` | 12 teams | Starting in Contender stage |
| `CHALLENGER_TEAMS` | 8 teams | Starting in Opener stage |
| `TEAM_IDS` | 28 entries | Name → HLTV ID — update once the 2026 roster is confirmed |

---

## Extending the model

**More training data** — add HLTV team IDs to `SEED_TEAM_IDS` in `o1_config.py`. The SQLite cache means duplicate match IDs are only fetched once.

**New features** — add columns in `rolling_team_features()` in `o3_features.py`. Mirror any new keys in `build_feature_vector()` in `o6_simulate.py`. The feature column list is saved to `models/feature_cols.pkl` so both scripts stay in sync.

**Feature importance** — after training, check `data/feature_importance.csv` or open `analysis.ipynb` for a bar chart of the top-25 XGBoost importances.
