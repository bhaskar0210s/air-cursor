from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CALIBRATION_STEPS = (
    ("top_left", "TOP-LEFT"),
    ("top_right", "TOP-RIGHT"),
    ("bottom_left", "BOTTOM-LEFT"),
    ("bottom_right", "BOTTOM-RIGHT"),
)


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

        top_left = samples["top_left"]
        top_right = samples["top_right"]
        bottom_left = samples["bottom_left"]
        bottom_right = samples["bottom_right"]

        left = (top_left[0] + bottom_left[0]) * 0.5
        right = (top_right[0] + bottom_right[0]) * 0.5
        top = (top_left[1] + top_right[1]) * 0.5
        bottom = (bottom_left[1] + bottom_right[1]) * 0.5

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
