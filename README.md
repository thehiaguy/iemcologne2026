# IEM Cologne 2026 — CS2 Major Predictor

A machine-learning + Monte-Carlo system that predicts every stage of the **IEM
Cologne 2026 Counter-Strike 2 Major**: Stage 1 / Stage 2 / Stage 3 advancement,
playoff advancement (quarter-/semi-/final), and the champion.

The match model is a **stacked, calibrated ensemble** (Elo + Glicko-2 + Valve
Regional Standings + gradient-boosting/logistic/forest base learners) trained on
**CS2-only** results. A Major-format **Monte-Carlo** then plays the bracket
10,000,000 times to turn per-match odds into per-team advancement probabilities.

> **Note on the favourite.** The model is trained on 2024–2026 CS2 results, in which
> **Vitality** were historically dominant — so they emerge as a heavy title favourite.
> That reflects the data, not a hand-set prior; single-elimination variance still
> leaves the rest of the field meaningful chances.

---

## Results (10,000,000 simulations)

Out-of-sample walk-forward backtest (CS2 history):

| model | n | accuracy | log-loss | Brier | AUC |
| :- | -: | -: | -: | -: | -: |
| **ensemble** | 20,439 | **0.657** | **0.6309** | **0.2158** | **0.713** |
| elo-only | 20,439 | 0.639 | 0.6312 | 0.2208 | 0.694 |
| vrs-only | 20,439 | 0.571 | 0.7249 | 0.2577 | 0.597 |

On **cross-region** matches (the regime a Major lives in) the ensemble beats Elo-only
by **+3.2 pp accuracy** and **0.6136 vs 0.6346 log-loss** — the payoff of the
cross-region VRS correction.

Predicted title odds (top 8):

| Team | Champion | Reach Final | Reach Playoffs |
| :- | -: | -: | -: |
| Vitality | 64.2% | 71.4% | 98.0% |
| Falcons | 8.1% | 22.3% | 78.3% |
| Natus Vincere | 6.2% | 19.7% | 75.1% |
| The MongolZ | 3.9% | 13.5% | 65.4% |
| Spirit | 3.8% | 12.8% | 52.9% |
| FURIA | 3.8% | 13.0% | 63.9% |
| Aurora | 2.3% | 9.8% | 58.2% |
| PARIVISION | 2.0% | 9.0% | 56.4% |

Full tables, CSVs, and charts land in `outputs/` (see `outputs/REPORT.md`).

---

## Quick start

The stack is small and standard — **numpy, pandas, scikit-learn, joblib, matplotlib**,
**JupyterLab**, and an optional **PyTorch** ensemble member (all in `requirements.txt`).
If your conda env already has these, just activate it and skip to the notebook below.

Otherwise, create an isolated environment. `requirements.txt` is a pip-format file, so
install it with pip inside a conda env, or hand the file to conda directly:

```bash
# Option A: conda env, install with pip (most reliable)
conda create -n iemcs python=3.13 -y
conda activate iemcs
pip install -r requirements.txt

# Option B: pure conda (reads the same file)
conda create -n iemcs python=3.13 -c conda-forge --file requirements.txt
conda activate iemcs

# Option C: plain venv, no conda
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> A bare `pip install` against a **system** Python may fail with
> `externally-managed-environment` (PEP 668) — use one of the isolated envs above.

Then open the notebook and **Run All**:

```bash
jupyter lab notebooks/IEM_Cologne_2026_Major_Predictor.ipynb
```

It runs the whole pipeline end-to-end — collect CS2 data → engineer features → train the
ensemble → backtest → simulate the Major — with every table and chart inline, driving the
reusable `iemcs` package. The first run clones the Valve Regional Standings repo
(<https://github.com/ValveSoftware/counter-strike_regional_standings>) into `data/raw/`
(set `$VRS_REPO_DIR` to reuse an existing clone); tune `N_SIMS` in the setup cell.

Run `pytest` for the unit tests covering the Swiss / series / simulation logic.

---

## How it works

### Data — CS2 only
We used Valve's official **Regional Standings** repo as the basis. Its
per-team monthly "details" files list every match a roster played (date, opponent,
W/L), which we deduplicate into **~27,000 CS2 matches (Feb 2024 → May 2026)**.
The same repo provides VRS points, rosters and regions, and the loader
(`vrs.parse_matches`) filters to the CS2 era so no pre-CS2 rows survive.

### Features (leak-free, per match)
A single chronological pass evolves each team's state and records *pre-match*
features before observing the result:

* **Elo** and **Glicko-2** ratings (Glicko deviation propagates into uncertainty)
* **VRS** point differential (official, opponent-network aware)
* **Strength-of-schedule** (recency-weighted opponent VRS) — flags weak-region padding
* recent **form** (EWMA), **head-to-head**, **activity**, **streak**, **roster
  stability** (months the current lineup has been intact), **same-region** flag

### Model (the ensemble)
Base learners (logistic regression, random forest, hist-gradient-boosting,
**k-nearest-neighbours**, and a **PyTorch MLP**. XGBoost / LightGBM auto-added if
installed) are stacked by a logistic meta-learner on **out-of-fold** predictions, then
**isotonically calibrated**. All fitting is **recency-weighted** (exponential decay,
~1-year half-life) so the model tracks the current meta rather than stale form. Because
a Major is mostly cross-region — where region-inflated Elo misleads — cross-region
matchups are blended toward the VRS estimate, with the blend weight tuned
(recency-weighted) on held-out cross-region matches.

The model's native output is a (Bo3-equivalent) encounter probability; `series.py`
inverts it to a per-map probability and recomputes **Bo1 / Bo3 / Bo5** odds for the
exact format each match uses.

### Simulation (the format)
* **Three 16-team Swiss stages** — first to 3 wins advances, 3 losses out, top 8
  advance. Round 1 seeds `i` vs `i+8`; later rounds pair within win-loss groups by
  **Buchholz**, high-vs-low, avoiding rematches. Bo1 except advancement/elimination
  matches are Bo3; **Stage 3 is all Bo3**.
* **8-team single-elimination playoffs** — Bo3, **Bo5 grand final**.
* 10,000,000 runs (parallel) aggregate each team's probability of clearing every stage
  and winning, with 95% Monte-Carlo confidence intervals.

---

## Project layout

```
iemcs/            package: data, ratings, features, model, simulators, reporting
  vrs.py          parse the Valve repo -> matches / standings / rosters / regions
  dataset.py      leak-free chronological feature builder (Context)
  ratings.py      Elo + Glicko-2
  model.py        stacked ensemble + calibration + cross-region VRS blend
  torch_model.py  optional PyTorch MLP base learner (auto-detected)
  series.py       Bo1/Bo3/Bo5 math + encounter<->map inversion
  swiss.py        Major Swiss engine            playoffs.py   single-elim bracket
  tournament.py   Monte-Carlo driver + CIs      validate.py   walk-forward backtest + model comparison
  simbacktest.py  end-to-end backtest on real past Majors (playoff brackets)
  report.py       tables / CSV / charts / markdown
  config.py       format spec, paths, hyper-parameters
  teams.py  the 32-team field
notebooks/        end-to-end notebook
tests/            pytest suite
data/             raw and processed data
outputs/          processed tables / predictions, charts, report
```

## Assumptions & limitations
* The 32-team field, stage seeding and format are taken as confirmed (Liquipedia +
  Wikipedia); a roster/field change before June only needs edits to `teams.py`.
* VRS ranks *rosters*, so the top-rated roster per org is used; within-stage Swiss
  seeding uses current VRS points as a proxy for the official cut-off seeding.
* The Valve source records matches at the **series** level (no per-map or player
  data), so per-map Elo and player-rating features are intentionally omitted; series
  format is handled analytically instead.
* Probabilities reflect the model's read of 2024–2026 form — not injuries, roster
  swaps or patch changes after the last VRS snapshot.
