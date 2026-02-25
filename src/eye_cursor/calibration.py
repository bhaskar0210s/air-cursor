from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CALIBRATION_STEPS = (
    ("top_left", "TOP-LEFT"),
    ("top_mid_left", "TOP-MID-LEFT"),
    ("top_center", "TOP-CENTER"),
    ("top_mid_right", "TOP-MID-RIGHT"),
    ("top_right", "TOP-RIGHT"),
    ("upper_left", "UPPER-LEFT"),
    ("upper_mid_left", "UPPER-MID-LEFT"),
    ("upper_center", "UPPER-CENTER"),
    ("upper_mid_right", "UPPER-MID-RIGHT"),
    ("upper_right", "UPPER-RIGHT"),
    ("center_left", "CENTER-LEFT"),
    ("center_mid_left", "CENTER-MID-LEFT"),
    ("center", "CENTER"),
    ("center_mid_right", "CENTER-MID-RIGHT"),
    ("center_right", "CENTER-RIGHT"),
    ("lower_left", "LOWER-LEFT"),
    ("lower_mid_left", "LOWER-MID-LEFT"),
    ("lower_center", "LOWER-CENTER"),
    ("lower_mid_right", "LOWER-MID-RIGHT"),
    ("lower_right", "LOWER-RIGHT"),
    ("bottom_left", "BOTTOM-LEFT"),
    ("bottom_mid_left", "BOTTOM-MID-LEFT"),
    ("bottom_center", "BOTTOM-CENTER"),
    ("bottom_mid_right", "BOTTOM-MID-RIGHT"),
    ("bottom_right", "BOTTOM-RIGHT"),
)

CALIBRATION_TARGETS = {
    "top_left": (0.1, 0.1),
    "top_mid_left": (0.3, 0.1),
    "top_center": (0.5, 0.1),
    "top_mid_right": (0.7, 0.1),
    "top_right": (0.9, 0.1),
    "upper_left": (0.1, 0.3),
    "upper_mid_left": (0.3, 0.3),
    "upper_center": (0.5, 0.3),
    "upper_mid_right": (0.7, 0.3),
    "upper_right": (0.9, 0.3),
    "center_left": (0.1, 0.5),
    "center_mid_left": (0.3, 0.5),
    "center": (0.5, 0.5),
    "center_mid_right": (0.7, 0.5),
    "center_right": (0.9, 0.5),
    "lower_left": (0.1, 0.7),
    "lower_mid_left": (0.3, 0.7),
    "lower_center": (0.5, 0.7),
    "lower_mid_right": (0.7, 0.7),
    "lower_right": (0.9, 0.7),
    "bottom_left": (0.1, 0.9),
    "bottom_mid_left": (0.3, 0.9),
    "bottom_center": (0.5, 0.9),
    "bottom_mid_right": (0.7, 0.9),
    "bottom_right": (0.9, 0.9),
}


@dataclass(frozen=True)
class CalibrationMap:
    left: float
    right: float
    top: float
    bottom: float

    def apply(self, x_norm: float, y_norm: float) -> tuple[float, float]:
        x_span = max(self.right - self.left, 1e-4)
        y_span = max(self.bottom - self.top, 1e-4)
        calibrated_x = (x_norm - self.left) / x_span
        calibrated_y = (y_norm - self.top) / y_span
        return (
            float(np.clip(calibrated_x, 0.0, 1.0)),
            float(np.clip(calibrated_y, 0.0, 1.0)),
        )


class CalibrationSession:
    def __init__(self) -> None:
        self._active = False
        self._index = 0
        self._samples: dict[str, tuple[float, float]] = {}
        self.map: CalibrationMap | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def is_calibrated(self) -> bool:
        return self.map is not None

    @property
    def index(self) -> int:
        return self._index

    def start(self) -> str:
        self._active = True
        self._index = 0
        self._samples = {}
        return self.current_prompt()

    def reset(self) -> str:
        self._active = False
        self._index = 0
        self._samples = {}
        self.map = None
        return "Calibration reset"

    def current_prompt(self) -> str:
        clamped_index = int(np.clip(self._index, 0, len(CALIBRATION_STEPS) - 1))
        _, step_label = CALIBRATION_STEPS[clamped_index]
        return f"Calibration {clamped_index + 1}/{len(CALIBRATION_STEPS)}: look {step_label} and press SPACE"

    def current_target(self) -> tuple[float, float] | None:
        if not self._active:
            return None
        clamped_index = int(np.clip(self._index, 0, len(CALIBRATION_STEPS) - 1))
        step_id, _ = CALIBRATION_STEPS[clamped_index]
        return CALIBRATION_TARGETS.get(step_id)

    def capture(self, sample: tuple[float, float] | None) -> str:
        if not self._active:
            return "Calibration is not active"
        if sample is None:
            return "Calibration needs visible eyes"

        step_id, step_label = CALIBRATION_STEPS[self._index]
        self._samples[step_id] = sample
        self._index += 1

        if self._index < len(CALIBRATION_STEPS):
            return f"Captured {step_label}. {self.current_prompt()}"

        candidate_map = self._build_map(self._samples)
        self._active = False
        self._index = 0
        self._samples = {}

        if candidate_map is None:
            return "Calibration failed. Press c and capture wider corners."

        self.map = candidate_map
        return "Calibration complete"

    @staticmethod
    def _build_map(samples: dict[str, tuple[float, float]]) -> CalibrationMap | None:
        required_keys = {step_id for step_id, _ in CALIBRATION_STEPS}
        if not required_keys.issubset(samples):
            return None

        x_values = [point[0] for point in samples.values()]
        y_values = [point[1] for point in samples.values()]
        left = min(x_values)
        right = max(x_values)
        top = min(y_values)
        bottom = max(y_values)

        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top

        x_span = right - left
        y_span = bottom - top
        if x_span < 0.04 or y_span < 0.04:
            return None

        x_margin = x_span * 0.08
        y_margin = y_span * 0.08

        return CalibrationMap(
            left=float(np.clip(left - x_margin, 0.0, 1.0)),
            right=float(np.clip(right + x_margin, 0.0, 1.0)),
            top=float(np.clip(top - y_margin, 0.0, 1.0)),
            bottom=float(np.clip(bottom + y_margin, 0.0, 1.0)),
        )
