from __future__ import annotations

import numpy as np


def blend_point(previous: tuple[float, float] | None, current: tuple[float, float], alpha: float) -> tuple[float, float]:
    if previous is None:
        return current
    x = previous[0] * (1.0 - alpha) + current[0] * alpha
    y = previous[1] * (1.0 - alpha) + current[1] * alpha
    return x, y


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))


def clamp_sensitivity(value: float, min_value: float, max_value: float) -> float:
    return float(np.clip(value, min_value, max_value))


def apply_precision_curve(norm_value: float, sensitivity: float, gamma: float) -> float:
    gamma = float(max(gamma, 1.0))
    centered = (norm_value - 0.5) * 2.0
    curved = np.sign(centered) * (abs(centered) ** gamma)
    scaled = curved * sensitivity
    return float(np.clip((scaled * 0.5) + 0.5, 0.0, 1.0))


def expand_uncalibrated(norm_value: float, gain: float) -> float:
    expanded = (norm_value - 0.5) * gain + 0.5
    return float(np.clip(expanded, 0.0, 1.0))


def stabilize_norm(
    previous: tuple[float, float] | None,
    current: tuple[float, float],
    deadband_norm: float,
) -> tuple[float, float]:
    if previous is None:
        return current
    if distance(previous, current) < deadband_norm:
        return previous
    return current
