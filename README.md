# IEM Cologne 2026 — CS2 ML Predictor & Pick'Em Optimizer

A full machine-learning pipeline that scrapes HLTV match history, trains a match-outcome classifier, runs Monte Carlo simulations to predict IEM Cologne 2026 bracket results, and finds the optimal Pick'Em entry to maximise your chances of qualifying for the next round.

---

## What it does

| Stage | Module | Description |
|---|---|---|
| Scrape | `o2_scraper.py` | Pulls CS2 match history + per-map player stats from HLTV |
| Features | `o3_features.py` | Engineers rolling stats, ELO ratings, H2H records |
| Collect | `o5_collect.py` | Orchestrates data collection for all seed teams |
| Train | `o6_train.py` | Trains PyTorch ResNet MLP + XGBoost ensemble |
| Simulate | `o7_simulate.py` | Monte Carlo Swiss + playoff simulation |
| Pick'Em | `o8_pickem.py` | Finds the pick set that maximises P(≥5 correct) |
| Analyse | `analysis.ipynb` | 20+ visualisations across all stages of the pipeline |

---

## How it works

### Data collection — `o2_scraper.py`
- Fetches match results via the [eupeutro HLTV REST API](https://github.com/eupeutro/hltv-api)
- Scrapes per-map player stats directly from HLTV using Selenium + undetected-chromedriver (bypasses Cloudflare)
- Caches all responses in SQLite (`data/hltv_cache.db`) with a 72-hour TTL so re-runs never re-fetch

### Feature engineering — `o3_features.py`
- Rolling averages (windows: 5, 10, 20 matches) computed at the exact timestamp of each match — zero lookahead
- Live ELO rating updated chronologically across the full CS2-era dataset
- Head-to-head records, differential features, team profile data (world ranking, average age)
- 114 features total

### Model training — `o6_train.py`
- Time-based train / val / test split (train ≤ Dec 2025, val Jan–Feb 2026, test Mar 2026+)
- **PyTorch ResNet MLP** — 256-wide, 4 residual blocks, trained with AdamW + CosineAnnealingWarmRestarts and mixed-precision (CUDA 12.8 / RTX 5080)
- **XGBoost** trained on CPU
- **Ensemble** — convex blend weight grid-searched on validation AUC
- Saves per-epoch train/val loss + LR to `data/training_history.csv` for visualisation

### Simulation — `o7_simulate.py`
- Assembles feature vectors for any matchup using live team stats + trained artifacts
- Swiss-stage simulation: same-record pairing, BO1
- Playoff simulation: seeded single-elimination BO3
- 50,000+ Monte Carlo iterations → champion, runner-up, top-4, top-8 probabilities per team

### Pick'Em Optimizer — `o8_pickem.py`
- Simulates the 16-team Stage 1 Swiss bracket up to 5 million times
- Enumerates ~15,000 candidate pick sets (2 teams to go 3-0, 6 to advance, 2 to go 0-3)
- Accumulates score histograms per combo in constant RAM using chunked multiprocessing
- numba JIT-compiled simulation + parallel accumulation for 30-50× speedup
- Convergence detection stops early once the optimal pick set has stabilised
- **Result: P(≥5 correct) ≈ 56.5%**, maximised over all possible pick combinations

---

## Features engineered

Every feature is computed independently for each team at match time.

| Category | Features |
|---|---|
| Firepower | `avg_rating`, `avg_adr`, `avg_kast` |
| Dueling | `avg_opening_kd`, `avg_opening_kills` |
| Impact | `avg_multi_kills`, `avg_clutches`, `avg_flash_assists`, `avg_hs_kills` |
| Consistency | `rating_std`, `adr_std` |
| Form | `win_rate`, `form_last5` |
| ELO | `t1_elo`, `t2_elo`, `elo_diff` |
| Ranking | `world_ranking`, `valve_ranking`, `weeks_top30`, `avg_age` |
| Head-to-head | `h2h_win_rate`, `h2h_total`, `h2h_last5_wins` |
| Differentials | `diff_<metric>_<window>` for every rolling stat |

All rolling metrics at three windows: **w5 / w10 / w20**.

---

## Tournament format — IEM Cologne 2026

```
Stage 1  (Opening)     16 teams   Swiss BO1 (3W / 3L)   top 8 advance
Stage 2  (Challenger)  16 teams   Swiss BO1 (3W / 3L)   top 8 advance
Stage 3  (Legends)     16 teams   Swiss BO3 (3W / 3L)   top 8 advance
Playoffs               8 teams    Single-elimination BO3 (seeded 1v8 … 4v5)
```

The Pick'Em optimizer targets **Stage 1** — the 16-team Opening stage.

---

## Visualisations — `analysis.ipynb`

### Model performance
- **5.1** Test-set metrics (AUC for each model + ensemble)
- **5.2** ROC curves & calibration curves
- **5.3** Predicted probability distribution by outcome
- **5.4** Confidence-accuracy: model accuracy as a function of certainty
- **5.5** Training loss curve + LR schedule + val AUC over epochs
- **5.6** Neural network weight diagram — node-link graph of top neurons, edges coloured by sign, width ∝ |weight|

### Pick'Em optimizer
- **7.1** Marginal outcome probabilities (P(3-0), P(advance), P(0-3)) per team
- **7.2** Optimal pick'em card — dark-themed panel layout
- **7.3** Score distribution for the optimal pick set
- **7.4** Swiss outcome distribution — stacked bar (3-0 / advance / elim / 0-3) per team
- **7.5** Monte Carlo convergence — P(≥5) vs simulations run
- **7.6** Pick stability heatmap — which picks locked in early vs changed across rounds

---

## Setup

**Requirements:** Python 3.10+, CUDA optional (developed on Python 3.14 / Windows 11 / RTX 5080)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1

# lxml must be binary-only on Python 3.14+
pip install lxml --only-binary :all:
pip install hltv-async-api --no-deps
pip install -r requirements.txt

# PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**Chrome:** Selenium uses your locally installed Google Chrome. Set `CHROME_VER` in `o1_config.py` to match your installed version.

---

## Usage

### 1. Collect data

```powershell
python o5_collect.py
```

### 2. Train the model

```powershell
python o6_train.py
```

Saves all artifacts to `models/` and training history to `data/training_history.csv`.

### 3. Run tournament simulation

```powershell
python o7_simulate.py --tournament --n 50000
```

### 4. Run the Pick'Em optimizer

```powershell
python o8_pickem.py --n-sims 5000000
```

### 5. Open the analysis notebook

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
├── o5_collect.py          Data collection orchestrator
├── o6_train.py            Full training pipeline
├── o7_simulate.py         Monte Carlo Swiss + bracket simulation
├── o8_pickem.py           Pick'Em optimizer (chunked MP + numba JIT)
├── analysis.ipynb         20+ visualisations across all pipeline stages
├── requirements.txt
├── data/
│   ├── raw_matches.parquet
│   ├── training_data.parquet
│   ├── feature_importance.csv
│   ├── training_history.csv   per-epoch loss/AUC (generated by o6_train.py)
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

All constants in `o1_config.py`:

| Constant | Default | Description |
|---|---|---|
| `SEED_TEAM_IDS` | 32 teams | HLTV team IDs to pull training data from |
| `TRAIN_END_DATE` | `2025-12-31` | Training cutoff |
| `VAL_END_DATE` | `2026-02-28` | Validation cutoff |
| `ROLLING_WINDOWS` | `[5, 10, 20]` | Match windows for rolling features |
| `ELO_K` | `32` | ELO K-factor |
| `CHROME_VER` | `148` | Installed Chrome version |
| `PT_HIDDEN` | `256` | MLP hidden dimension |
| `PT_N_BLOCKS` | `4` | Number of residual blocks |
