"""Game metadata (JSON) loading.

Team labels used everywhere downstream: ``A`` = home team, ``B`` = away team.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RGB = tuple[int, int, int]

_NAMED = {
    "white": (245, 245, 245), "black": (20, 20, 20), "grey": (128, 128, 128), "gray": (128, 128, 128),
    "red": (200, 30, 40), "orange": (240, 96, 40), "yellow": (240, 210, 40), "green": (30, 140, 60),
    "blue": (30, 70, 180), "navy": (20, 30, 80), "purple": (90, 60, 140), "maroon": (110, 20, 40),
}


def parse_color(value: Any) -> RGB:
    """Accepts '#rrggbb', a few colour names, or [r, g, b]."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(c) for c in value)  # type: ignore[return-value]
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _NAMED:
            return _NAMED[v]
        if v.startswith("#") and len(v) == 7:
            return tuple(int(v[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]
    raise ValueError(f"Cannot parse colour: {value!r}")


def rgb_to_hex(rgb: RGB) -> str:
    return "#%02x%02x%02x" % rgb


@dataclass
class TeamInfo:
    label: str  # "A" | "B"
    name: str
    primary: RGB
    secondary: RGB | None = None
    roster: list[dict[str, Any]] = field(default_factory=list)  # [{"number": 13, "name": "..."}]


@dataclass
class GameMeta:
    home: TeamInfo
    away: TeamInfo
    referee_primary: RGB = (128, 128, 128)
    referee_secondary: RGB | None = (20, 20, 20)
    game_date: str | None = None
    location: str | None = None
    court_standard: str = "FIBA"
    camera: dict[str, Any] = field(default_factory=dict)
    # Normalised [x0, y0, x1, y1] screen regions covered by broadcast graphics (scorebug, logo).
    overlay_regions: list[list[float]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None  # the meta.json this came from (the stable team binding is stored there)
    # extra public info per team label from the team binding (replay/team_binding.py): side, side_confirmed,
    # colour_name, ... - merged into to_public_dict()["teams"][label]
    team_extra: dict[str, dict[str, Any]] = field(default_factory=dict)

    def team(self, label: str) -> TeamInfo:
        return self.home if label == "A" else self.away

    def to_public_dict(self) -> dict[str, Any]:
        def t(info: TeamInfo) -> dict[str, Any]:
            return {"label": info.label, "name": info.name, "primary": rgb_to_hex(info.primary),
                    "secondary": rgb_to_hex(info.secondary) if info.secondary else None,
                    **self.team_extra.get(info.label, {})}

        return {
            "game_date": self.game_date, "location": self.location, "court_standard": self.court_standard,
            "teams": {"A": t(self.home), "B": t(self.away)},
            "referee_color": rgb_to_hex(self.referee_primary),
        }


def load_metadata(path: str | Path | None) -> GameMeta:
    data: dict[str, Any] = {}
    if path is not None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = metadata_from_dict(data)
    meta.path = Path(path) if path is not None else None
    return meta


def metadata_from_dict(data: dict[str, Any]) -> GameMeta:
    colors = data.get("jersey_colors", {})
    roster = data.get("roster", {}) or {}

    def team(label: str, key: str, default_name: str, default_color: RGB) -> TeamInfo:
        c = colors.get(key, {})
        return TeamInfo(
            label=label,
            name=data.get(f"{key}_team") or default_name,
            primary=parse_color(c["primary"]) if "primary" in c else default_color,
            secondary=parse_color(c["secondary"]) if c.get("secondary") else None,
            roster=list(roster.get(key, []) or []),
        )

    ref = colors.get("referee", {})
    standard = str(data.get("court_standard", "FIBA")).upper()
    if standard != "FIBA":
        raise ValueError(f"Only the FIBA court is supported so far (got {standard!r}).")
    return GameMeta(
        home=team("A", "home", "Home", (245, 245, 245)),
        away=team("B", "away", "Away", (20, 30, 80)),
        referee_primary=parse_color(ref["primary"]) if "primary" in ref else (128, 128, 128),
        referee_secondary=parse_color(ref["secondary"]) if ref.get("secondary") else (20, 20, 20),
        game_date=data.get("game_date"),
        location=data.get("location"),
        court_standard=standard,
        camera=dict(data.get("camera", {}) or {}),
        overlay_regions=[list(map(float, r)) for r in data.get("overlay_regions", [])],
        raw=data,
    )
