from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


def _poly2_features(x: float, y: float) -> np.ndarray:
    return np.array([1.0, x, y, x * y, x * x, y * y], dtype=np.float64)


@dataclass(frozen=True)
class CalibrationModel:
    coefficients: np.ndarray

    def map(self, x: float, y: float) -> tuple[float, float]:
        feature = _poly2_features(x, y)
        output = feature @ self.coefficients
        return float(output[0]), float(output[1])


class Calibrator:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        grid_size: int = 3,
        samples_required: int = 20,
    ) -> None:
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_size = grid_size
        self.samples_required = samples_required

        self.targets = self._generate_targets(grid_size)
        self.active = False
        self.index = 0

        self._samples: list[tuple[float, float]] = []
        self._paired_features: list[tuple[float, float]] = []
        self._paired_targets: list[tuple[float, float]] = []

        self._canvas = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)

    def _generate_targets(self, grid_size: int) -> list[tuple[float, float]]:
        x_margin = self.screen_width * 0.12
        y_margin = self.screen_height * 0.12
        xs = np.linspace(x_margin, self.screen_width - x_margin, grid_size)
        ys = np.linspace(y_margin, self.screen_height - y_margin, grid_size)

        targets: list[tuple[float, float]] = []
        for row, y in enumerate(ys):
            x_order: Iterable[float] = xs if row % 2 == 0 else xs[::-1]
            for x in x_order:
                targets.append((float(x), float(y)))
        return targets

    def start(self) -> None:
        self.active = True
        self.index = 0
        self._samples.clear()
        self._paired_features.clear()
        self._paired_targets.clear()

    def stop(self) -> None:
        self.active = False

    def current_target(self) -> tuple[float, float] | None:
        if not self.active or self.index >= len(self.targets):
            return None
        return self.targets[self.index]

    def add_sample(self, feature_x: float, feature_y: float) -> None:
        if not self.active:
            return
        self._samples.append((feature_x, feature_y))
        max_samples = self.samples_required * 2
        if len(self._samples) > max_samples:
            self._samples = self._samples[-max_samples:]

    def capture_current_point(self) -> tuple[bool, str]:
        if not self.active:
            return False, "calibration is not active"

        if len(self._samples) < self.samples_required:
            return False, f"need {self.samples_required - len(self._samples)} more samples"

        target = self.current_target()
        if target is None:
            return False, "no active target"

        selected = np.array(self._samples[-self.samples_required :], dtype=np.float64)
        mean_feature = selected.mean(axis=0)

        self._paired_features.append((float(mean_feature[0]), float(mean_feature[1])))
        self._paired_targets.append(target)

        self.index += 1
        self._samples.clear()

        if self.index >= len(self.targets):
            return True, "done"

        return True, f"captured point {self.index}/{len(self.targets)}"

    def fit(self) -> CalibrationModel | None:
        if len(self._paired_features) < 6:
            return None

        x_matrix = np.array([
            _poly2_features(fx, fy) for fx, fy in self._paired_features
        ], dtype=np.float64)
        y_matrix = np.array(self._paired_targets, dtype=np.float64)

        coeffs, _, _, _ = np.linalg.lstsq(x_matrix, y_matrix, rcond=None)
        return CalibrationModel(coefficients=coeffs)

    def render(self, message: str) -> np.ndarray:
        self._canvas[:] = (10, 10, 10)

        target = self.current_target()
        if target is not None:
            tx, ty = int(target[0]), int(target[1])
            cv2.circle(self._canvas, (tx, ty), 22, (0, 0, 255), -1)
            cv2.circle(self._canvas, (tx, ty), 35, (255, 255, 255), 2)

        status = f"Point {min(self.index + 1, len(self.targets))}/{len(self.targets)}"
        cv2.putText(self._canvas, status, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2)
        cv2.putText(
            self._canvas,
            "Point index fingertip at red dot, then press SPACE. Press q to quit.",
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
        )
        cv2.putText(self._canvas, message, (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)

        return self._canvas
