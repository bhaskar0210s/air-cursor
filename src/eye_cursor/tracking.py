from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from urllib import request

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True)
class TrackingResult:
    x_norm: float
    y_norm: float
    handedness: str


class HandTracker:
    _DEFAULT_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    )

    def __init__(self, model_path: str | None = None, fingertip_landmark_index: int = 8) -> None:
        self._hands = None
        self._landmarker = None
        self._backend = ""
        self._last_timestamp_ms = 0
        self._fingertip_landmark_index = fingertip_landmark_index

        if hasattr(mp, "solutions"):
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._backend = "solutions"
            return

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        resolved_model_path = self._resolve_model_path(model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=resolved_model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._backend = "tasks"

    @classmethod
    def _resolve_model_path(cls, model_path: str | None) -> str:
        if model_path:
            path = Path(model_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            return str(path)

        cache_path = Path.home() / ".cache" / "eye-cursor" / "hand_landmarker.task"
        if cache_path.exists():
            return str(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(".download")

        try:
            with request.urlopen(cls._DEFAULT_MODEL_URL, timeout=60) as response:
                with temp_path.open("wb") as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)
            temp_path.replace(cache_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(
                "Failed to download MediaPipe hand landmarker model. "
                "Check internet access or pass --model-path."
            )

        return str(cache_path)

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms
        return now_ms

    def _extract_from_solutions(self, frame_rgb: np.ndarray) -> TrackingResult | None:
        result = self._hands.process(frame_rgb)
        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0].landmark
        if self._fingertip_landmark_index >= len(hand_landmarks):
            return None

        point = hand_landmarks[self._fingertip_landmark_index]
        handedness = "unknown"
        if result.multi_handedness and result.multi_handedness[0].classification:
            handedness = result.multi_handedness[0].classification[0].label.lower()

        return TrackingResult(
            x_norm=float(np.clip(point.x, 0.0, 1.0)),
            y_norm=float(np.clip(point.y, 0.0, 1.0)),
            handedness=handedness,
        )

    def _extract_from_tasks(self, frame_rgb: np.ndarray) -> TrackingResult | None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = self._next_timestamp_ms()
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]
        if self._fingertip_landmark_index >= len(hand_landmarks):
            return None

        point = hand_landmarks[self._fingertip_landmark_index]
        handedness = "unknown"
        if result.handedness and result.handedness[0]:
            first = result.handedness[0][0]
            handedness = (first.display_name or first.category_name or "unknown").lower()

        return TrackingResult(
            x_norm=float(np.clip(point.x, 0.0, 1.0)),
            y_norm=float(np.clip(point.y, 0.0, 1.0)),
            handedness=handedness,
        )

    def process(self, frame_bgr: np.ndarray) -> TrackingResult | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":
            return self._extract_from_solutions(frame_rgb)

        return self._extract_from_tasks(frame_rgb)
