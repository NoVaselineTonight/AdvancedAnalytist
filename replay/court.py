"""FIBA court geometry and the court coordinate system.

Coordinate system (metres):
    origin  = corner where the LEFT baseline meets the NEAR sideline,
              "left" and "near" as seen from the main camera position.
    x axis  = along the sidelines, 0 .. 28 (left baseline -> right baseline)
    y axis  = along the baselines, 0 .. 15 (near sideline -> far sideline)

All dimensions are to the inner edge of the boundary lines (FIBA Official
Basketball Rules 2024, Diagram 1/2). Line width (5 cm) is ignored.

The playing area is 28 x 15 m by default, but many gyms have smaller courts (e.g.
the older 26 x 14 m). The markings around the baskets (key, free-throw circle,
3-point line, centre circle) are the same everywhere, so only LENGTH/WIDTH vary:
``set_dimensions`` changes them for the whole program (the automatic calibration
measures them; they are stored in the calibration file). Always read the values as
``court.LENGTH`` etc. at call time, never copy them at import.
"""

from __future__ import annotations

import math

import numpy as np

LENGTH = 28.0
WIDTH = 15.0
MID_Y = WIDTH / 2

CENTER_CIRCLE_R = 1.8
FT_LINE_X = 5.8  # baseline inner edge -> far edge of free-throw line
FT_CIRCLE_R = 1.8
PAINT_HALF_WIDTH = 4.9 / 2
BASKET_X = 1.575  # baseline -> centre of the ring, projected on the floor
THREE_R = 6.75
THREE_SIDE_Y = 0.9  # straight 3pt segments are 0.9 m from the sidelines
NO_CHARGE_R = 1.25

# x where the straight part of the 3-point line meets the arc
THREE_BREAK_X = BASKET_X + math.sqrt(THREE_R**2 - (MID_Y - THREE_SIDE_Y) ** 2)
DEFAULT_LENGTH, DEFAULT_WIDTH = 28.0, 15.0
SPEC = "FIBA"

# Marking standards (metres). The automatic calibration tries each and keeps the best fit.
STANDARDS: dict[str, dict[str, float]] = {
    "FIBA": dict(three_r=6.75, three_side=0.9, paint_half=2.45, ft_x=5.8, basket_x=1.575, circle_r=1.8,
                 ft_circle_r=1.8, no_charge_r=1.25, length=28.0, width=15.0),
    "NBA": dict(three_r=7.24, three_side=0.91, paint_half=2.44, ft_x=5.79, basket_x=1.60, circle_r=1.83,
                ft_circle_r=1.83, no_charge_r=1.22, length=28.65, width=15.24),
    "NCAA": dict(three_r=6.75, three_side=1.02, paint_half=1.83, ft_x=5.79, basket_x=1.60, circle_r=1.83,
                 ft_circle_r=1.83, no_charge_r=1.22, length=28.65, width=15.24),
    "FIBA_OLD": dict(three_r=6.25, three_side=1.25, paint_half=2.45, ft_x=5.8, basket_x=1.575, circle_r=1.8,
                     ft_circle_r=1.8, no_charge_r=1.25, length=28.0, width=15.0),  # 3-pt line before 2010
}


def _left_keypoints() -> dict[str, tuple[float, float]]:
    return {
        "corner_near": (0.0, 0.0),
        "corner_far": (0.0, WIDTH),
        "three_base_near": (0.0, THREE_SIDE_Y),
        "three_base_far": (0.0, WIDTH - THREE_SIDE_Y),
        "paint_base_near": (0.0, MID_Y - PAINT_HALF_WIDTH),
        "paint_base_far": (0.0, MID_Y + PAINT_HALF_WIDTH),
        "ft_near": (FT_LINE_X, MID_Y - PAINT_HALF_WIDTH),
        "ft_far": (FT_LINE_X, MID_Y + PAINT_HALF_WIDTH),
        "ft_circle_top": (FT_LINE_X + FT_CIRCLE_R, MID_Y),
        "three_break_near": (THREE_BREAK_X, THREE_SIDE_Y),
        "three_break_far": (THREE_BREAK_X, WIDTH - THREE_SIDE_Y),
        "three_top": (BASKET_X + THREE_R, MID_Y),
    }


def mirror_x(p: tuple[float, float]) -> tuple[float, float]:
    return (LENGTH - p[0], p[1])


def keypoints() -> dict[str, tuple[float, float]]:
    """Named floor-level keypoints a user can click during calibration."""
    pts: dict[str, tuple[float, float]] = {}
    for name, p in _left_keypoints().items():
        pts[f"L_{name}"] = p
        pts[f"R_{name}"] = mirror_x(p)
    pts["mid_near"] = (LENGTH / 2, 0.0)
    pts["mid_far"] = (LENGTH / 2, WIDTH)
    pts["center_circle_near"] = (LENGTH / 2, MID_Y - CENTER_CIRCLE_R)
    pts["center_circle_far"] = (LENGTH / 2, MID_Y + CENTER_CIRCLE_R)
    return pts


KEYPOINTS = keypoints()


def set_spec(name: str = "FIBA") -> None:
    """Switch the marking standard (key width, 3-point line, ...); keeps LENGTH/WIDTH."""
    global SPEC, THREE_R, THREE_SIDE_Y, PAINT_HALF_WIDTH, FT_LINE_X, BASKET_X, CENTER_CIRCLE_R, FT_CIRCLE_R
    global NO_CHARGE_R
    sp = STANDARDS.get(name, STANDARDS["FIBA"])
    SPEC = name if name in STANDARDS else "FIBA"
    THREE_R, THREE_SIDE_Y, PAINT_HALF_WIDTH = sp["three_r"], sp["three_side"], sp["paint_half"]
    FT_LINE_X, BASKET_X, CENTER_CIRCLE_R = sp["ft_x"], sp["basket_x"], sp["circle_r"]
    FT_CIRCLE_R, NO_CHARGE_R = sp["ft_circle_r"], sp["no_charge_r"]
    set_dimensions(LENGTH, WIDTH)


def configure(length: float | None = None, width: float | None = None, spec: str | None = None) -> None:
    """Set standard and size together (None = keep). Use with the "court" entry of a calibration file."""
    if spec is not None:
        set_spec(spec)
    set_dimensions(LENGTH if length is None else length, WIDTH if width is None else width)


def describe() -> dict:
    return {"length": round(LENGTH, 3), "width": round(WIDTH, 3), "spec": SPEC}


def set_dimensions(length: float = DEFAULT_LENGTH, width: float = DEFAULT_WIDTH) -> None:
    """Set the playing-area size (metres) used by every module."""
    global LENGTH, WIDTH, MID_Y, THREE_BREAK_X, KEYPOINTS
    LENGTH, WIDTH = float(length), float(width)
    MID_Y = WIDTH / 2
    THREE_BREAK_X = BASKET_X + math.sqrt(max(0.0, THREE_R**2 - (MID_Y - THREE_SIDE_Y) ** 2))
    KEYPOINTS.clear()
    KEYPOINTS.update(keypoints())  # same dict object: modules holding a reference stay current


def dimensions() -> tuple[float, float]:
    return LENGTH, WIDTH


def _arc(cx: float, cy: float, r: float, a0: float, a1: float, n: int) -> np.ndarray:
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def court_polylines(n_arc: int = 48) -> list[np.ndarray]:
    """Court markings as a list of (N, 2) polylines in metres."""
    lines: list[np.ndarray] = [
        np.array([[0, 0], [LENGTH, 0], [LENGTH, WIDTH], [0, WIDTH], [0, 0]], float),
        np.array([[LENGTH / 2, 0], [LENGTH / 2, WIDTH]], float),
        _arc(LENGTH / 2, MID_Y, CENTER_CIRCLE_R, 0, 2 * math.pi, n_arc),
    ]
    a3 = math.asin(min(1.0, (MID_Y - THREE_SIDE_Y) / THREE_R))
    half: list[np.ndarray] = [
        np.array([[0, MID_Y - PAINT_HALF_WIDTH], [FT_LINE_X, MID_Y - PAINT_HALF_WIDTH],
                  [FT_LINE_X, MID_Y + PAINT_HALF_WIDTH], [0, MID_Y + PAINT_HALF_WIDTH]], float),
        _arc(FT_LINE_X, MID_Y, FT_CIRCLE_R, 0, 2 * math.pi, n_arc),
        np.vstack([
            [[0, THREE_SIDE_Y]],
            _arc(BASKET_X, MID_Y, THREE_R, -a3, a3, n_arc),
            [[0, WIDTH - THREE_SIDE_Y]],
        ]),
        _arc(BASKET_X, MID_Y, NO_CHARGE_R, -math.pi / 2, math.pi / 2, n_arc // 2),
    ]
    for line in half:
        lines.append(line)
        m = line.copy()
        m[:, 0] = LENGTH - m[:, 0]
        lines.append(m)
    return lines


def baskets() -> list[tuple[float, float]]:
    return [(BASKET_X, MID_Y), (LENGTH - BASKET_X, MID_Y)]


def in_court(xy: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Boolean mask of points inside the playing area expanded by ``margin`` metres."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    x, y = xy[:, 0], xy[:, 1]
    with np.errstate(invalid="ignore"):
        return (x >= -margin) & (x <= LENGTH + margin) & (y >= -margin) & (y <= WIDTH + margin)
