"""The IEM Cologne 2026 field: 32 teams, their stage seeding and VRS name aliases.

Stage assignment is fixed tournament knowledge (the official invite list); VRS
points / region / roster are parsed from the Valve repo by :mod:`iemcs.vrs`.

`vrs_name` is the name the team appears under in the Valve Regional Standings,
which occasionally differs from the common display name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldTeam:
    name: str       # display name
    vrs_name: str   # name as it appears in the VRS repo
    stage: int      # 1, 2 or 3 — the stage this team enters at


# Order within each stage is informational; the within-stage Swiss seed (1..16) is
# derived from current VRS points at simulation time.
FIELD: list[FieldTeam] = [
    # --- Stage 3 (Legends) seeds -------------------------------------------- #
    FieldTeam("Vitality", "Vitality", 3),
    FieldTeam("Natus Vincere", "Natus Vincere", 3),
    FieldTeam("Falcons", "Falcons", 3),
    FieldTeam("The MongolZ", "The MongolZ", 3),
    FieldTeam("PARIVISION", "PARIVISION", 3),
    FieldTeam("Aurora", "Aurora", 3),
    FieldTeam("FURIA", "FURIA", 3),
    FieldTeam("MOUZ", "MOUZ", 3),
    # --- Stage 2 (Challengers) seeds ---------------------------------------- #
    FieldTeam("FUT", "FUT", 2),
    FieldTeam("Spirit", "Spirit", 2),
    FieldTeam("Astralis", "Astralis", 2),
    FieldTeam("G2", "G2", 2),
    FieldTeam("Legacy", "Legacy", 2),
    FieldTeam("paiN", "paiN", 2),
    FieldTeam("Monte", "Monte", 2),
    FieldTeam("9z", "9z", 2),
    # --- Stage 1 (Contenders) ----------------------------------------------- #
    FieldTeam("GamerLegion", "GamerLegion", 1),
    FieldTeam("B8", "B8", 1),
    FieldTeam("HEROIC", "HEROIC", 1),
    FieldTeam("BetBoom", "BetBoom", 1),
    FieldTeam("BIG", "BIG", 1),
    FieldTeam("M80", "M80", 1),
    FieldTeam("MIBR", "MIBR", 1),
    FieldTeam("SINNERS", "SINNERS", 1),
    FieldTeam("NRG", "NRG", 1),
    FieldTeam("TYLOO", "TYLOO", 1),
    FieldTeam("Sharks", "Sharks", 1),
    FieldTeam("Gaimin Gladiators", "Gaimin Gladiators", 1),
    FieldTeam("Team Liquid", "Liquid", 1),
    FieldTeam("Lynn Vision", "Lynn Vision", 1),
    FieldTeam("THUNDERdOWNUNDER", "THUNDER dOWNUNDER", 1),
    FieldTeam("FlyQuest", "FlyQuest", 1),
]

assert len(FIELD) == 32, "field must contain 32 teams"
assert sum(t.stage == 1 for t in FIELD) == 16
assert sum(t.stage == 2 for t in FIELD) == 8
assert sum(t.stage == 3 for t in FIELD) == 8

# Display-name <-> VRS-name lookups
VRS_TO_DISPLAY = {t.vrs_name: t.name for t in FIELD}
DISPLAY_TO_VRS = {t.name: t.vrs_name for t in FIELD}


def stage_teams(stage: int) -> list[str]:
    """Display names of teams entering at `stage`."""
    return [t.name for t in FIELD if t.stage == stage]


# Extra aliases used when normalising opponent names found in match data so they
# collapse onto the canonical VRS name. Extend as needed.
NAME_ALIASES: dict[str, str] = {
    "Team Liquid": "Liquid",
    "THUNDERdOWNUNDER": "THUNDER dOWNUNDER",
    "Natus Vincere (NAVI)": "Natus Vincere",
    "NAVI": "Natus Vincere",
    "FaZe Clan": "FaZe",
    "G2 Esports": "G2",
    "FUT Esports": "FUT",
    "paiN Gaming": "paiN",
    "9z Team": "9z",
    "BIG Clan": "BIG",
    "Lynn Vision Gaming": "Lynn Vision",
    "Sharks Esports": "Sharks",
    "MOUZ NXT": "MOUZ",  # academy collapse (rare in tier-1 data)
}


def normalize_name(name: str) -> str:
    name = name.strip()
    return NAME_ALIASES.get(name, name)
