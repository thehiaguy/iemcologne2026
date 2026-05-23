"""Reporting: console tables, CSV exports, charts and a markdown summary."""
from __future__ import annotations

import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config  # noqa: E402

STAGE_LABEL = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3 (Legends)"}
PRETTY = {
    "adv_s1": "Adv S1", "adv_s2": "Adv S2", "adv_s3": "Playoffs",
    "semifinal": "Semifinal", "final": "Final", "champion": "Champion",
}


def _pct(x) -> str:
    return "  —  " if pd.isna(x) else f"{100 * x:5.1f}%"


# --------------------------------------------------------------------------- #
# Console
# --------------------------------------------------------------------------- #
def print_predictions(res: pd.DataFrame) -> None:
    cols = ["adv_s1", "adv_s2", "adv_s3", "semifinal", "final", "champion"]
    order = res.sort_values("champion", ascending=False)
    print("\n" + "=" * 92)
    print("IEM COLOGNE 2026 — PREDICTED ADVANCEMENT PROBABILITIES "
          f"({config.N_SIMS:,} simulations)")
    print("=" * 92)
    head = f"{'#':>2} {'Team':20} {'St':>2} " + " ".join(f"{PRETTY[c]:>9}" for c in cols)
    print(head)
    print("-" * len(head))
    for i, (team, row) in enumerate(order.iterrows(), 1):
        print(f"{i:>2} {team:20} {int(row['stage']):>2} "
              + " ".join(f"{_pct(row[c]):>9}" for c in cols))
    print("-" * len(head))
    print("Adv S1/S2 = reach next Swiss stage; Playoffs = top-8 of Stage 3; "
          "then single-elim rounds.\n")


def print_champion_table(res: pd.DataFrame, top: int = 12) -> None:
    print("Title odds (95% Monte-Carlo CI):")
    o = res.sort_values("champion", ascending=False).head(top)
    for team, r in o.iterrows():
        bar = "█" * int(round(r["champion"] * 60))
        print(f"  {team:20} {100*r['champion']:5.1f}%  "
              f"[{100*r['champion_lo']:4.1f}, {100*r['champion_hi']:4.1f}]  {bar}")


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def save_csvs(res: pd.DataFrame) -> list[str]:
    paths = []
    full = os.path.join(config.OUTPUTS_DIR, "predictions.csv")
    res.round(5).to_csv(full)
    paths.append(full)
    for metric, fname in [
        ("adv_s1", "stage1_advancement.csv"),
        ("adv_s2", "stage2_advancement.csv"),
        ("adv_s3", "stage3_advancement_playoffs.csv"),
        ("champion", "champions.csv"),
    ]:
        sub = res[["stage", "vrs", metric, f"{metric}_lo", f"{metric}_hi"]]
        sub = sub.dropna(subset=[metric]).sort_values(metric, ascending=False)
        p = os.path.join(config.OUTPUTS_DIR, fname)
        sub.round(5).to_csv(p)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _barh(series, lo, hi, title, xlabel, fname, color="#c0392b"):
    s = series.dropna().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(s))))
    err = None
    if lo is not None and hi is not None:
        err = [s.values - lo.loc[s.index].values, hi.loc[s.index].values - s.values]
    ax.barh(range(len(s)), s.values * 100, xerr=(np.array(err) * 100 if err is not None else None),
            color=color, alpha=0.85, error_kw=dict(ecolor="#555", lw=0.8))
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.OUTPUTS_DIR, fname)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def charts(res: pd.DataFrame, reliability=None) -> list[str]:
    paths = []
    top = res.sort_values("champion", ascending=False).head(16)
    paths.append(_barh(top["champion"], top["champion_lo"], top["champion_hi"],
                       "IEM Cologne 2026 — Title Odds", "Win probability (%)",
                       "chart_champion_odds.png"))
    paths.append(_barh(res["adv_s3"], res["adv_s3_lo"], res["adv_s3_hi"],
                       "Probability of Reaching Playoffs (Stage 3 top-8)",
                       "Probability (%)", "chart_playoff_reach.png", color="#2c7fb8"))

    # Stage-1 advancement (only the 16 Stage-1 teams)
    s1 = res[res["stage"] == 1]
    paths.append(_barh(s1["adv_s1"], s1["adv_s1_lo"], s1["adv_s1_hi"],
                       "Stage 1 Advancement (top-8 of 16)", "Probability (%)",
                       "chart_stage1_advancement.png", color="#31a354"))

    # Rating ladder
    ladder = res["vrs"].sort_values()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(range(len(ladder)), ladder.values, color="#7b6", alpha=0.85)
    ax.set_yticks(range(len(ladder)))
    ax.set_yticklabels(ladder.index)
    ax.set_xlabel("VRS points (current)")
    ax.set_title("Field strength — VRS points")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(config.OUTPUTS_DIR, "chart_rating_ladder.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    if reliability is not None:
        from sklearn.calibration import calibration_curve
        y, p_pred = reliability
        frac, mean_pred = calibration_curve(y, p_pred, n_bins=10, strategy="quantile")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.plot(mean_pred, frac, "o-", color="#c0392b", label="ensemble")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title("Calibration (reliability) — out-of-sample")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        pr = os.path.join(config.OUTPUTS_DIR, "chart_reliability.png")
        fig.savefig(pr, dpi=130)
        plt.close(fig)
        paths.append(pr)
    return paths


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def write_markdown(res: pd.DataFrame, backtest_out: dict | None) -> str:
    lines = ["# IEM Cologne 2026 — Predicted Results\n",
             f"_Monte-Carlo over {config.N_SIMS:,} simulated tournaments; "
             "match model = stacked Elo + Glicko-2 + VRS + gradient-boosting "
             "ensemble trained on CS2-only results._\n"]

    if backtest_out:
        e = backtest_out["ensemble"]
        lines += ["## Out-of-sample backtest\n",
                  "| model | n | accuracy | log-loss | Brier | AUC |",
                  "| :- | -: | -: | -: | -: | -: |"]
        for name in ["ensemble", "elo_only", "vrs_only"]:
            m = backtest_out[name]
            lines.append(f"| {name} | {m['n']} | {m['accuracy']:.3f} | "
                         f"{m['log_loss']:.4f} | {m['brier']:.4f} | {m['auc']:.3f} |")
        lines.append("")

    o = res.sort_values("champion", ascending=False)
    lines += ["## Title odds (top 12)\n", "| Team | Champion | Final | Semifinal | Playoffs |",
              "| :- | -: | -: | -: | -: |"]
    for team, r in o.head(12).iterrows():
        lines.append(f"| {team} | {100*r['champion']:.1f}% | {100*r['final']:.1f}% | "
                     f"{100*r['semifinal']:.1f}% | {100*r['adv_s3']:.1f}% |")

    lines += ["\n## Stage 1 advancement (16 → 8)\n", "| Team | Adv S1 |", "| :- | -: |"]
    for team, r in res[res.stage == 1].sort_values("adv_s1", ascending=False).iterrows():
        lines.append(f"| {team} | {100*r['adv_s1']:.1f}% |")

    path = os.path.join(config.OUTPUTS_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
