from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from urllib import request

import cv2
import mediapipe as mp
import numpy as np

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_BOUNDS = [33, 133, 159, 145, 160, 144, 158, 153]
RIGHT_EYE_BOUNDS = [362, 263, 386, 374, 385, 380, 387, 373]


def _resolve_or_download_model(
    model_path: str | None,
    cache_filename: str,
    model_url: str,
    model_name: str,
) -> str:
    if model_path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return str(path)

    cache_path = Path.home() / ".cache" / "eye-cursor" / cache_filename
    if cache_path.exists():
        return str(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(".download")

    try:
        with request.urlopen(model_url, timeout=60) as response:
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
            f"Failed to download MediaPipe {model_name} model. "
            "Check internet access or pass a local model path."
        )

    return str(cache_path)


@dataclass(frozen=True)
class TrackingResult:
    x_norm: float
    y_norm: float
    handedness: str


@dataclass(frozen=True)
class EyeTrackingResult:
    x_norm: float
    y_norm: float
    left_iris: tuple[float, float]
    right_iris: tuple[float, float]


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

        resolved_model_path = _resolve_or_download_model(
            model_path=model_path,
            cache_filename="hand_landmarker.task",
            model_url=self._DEFAULT_MODEL_URL,
            model_name="hand landmarker",
        )

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


class EyeTracker:
    _DEFAULT_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

    def __init__(self, model_path: str | None = None) -> None:
        self._mesh = None
        self._landmarker = None
        self._backend = ""
        self._last_timestamp_ms = 0

        if hasattr(mp, "solutions"):
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._backend = "solutions"
            return

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        resolved_model_path = _resolve_or_download_model(
            model_path=model_path,
            cache_filename="face_landmarker.task",
            model_url=self._DEFAULT_MODEL_URL,
            model_name="face landmarker",
        )

        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=resolved_model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._backend = "tasks"

    def close(self) -> None:
        if self._mesh is not None:
            self._mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def _next_timestamp_ms(self) -> int:
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._last_timestamp_ms:
            now_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = now_ms
        return now_ms

    @staticmethod
    def _to_xy(landmarks, idx: int, width: int, height: int) -> np.ndarray:
        lm = landmarks[idx]
        return np.array([lm.x * width, lm.y * height], dtype=np.float64)

    @staticmethod
    def _mean_point(points: list[np.ndarray]) -> np.ndarray:
        return np.mean(np.stack(points, axis=0), axis=0)

    @staticmethod
    def _normalized_position(point: np.ndarray, bounds: list[np.ndarray]) -> tuple[float, float]:
        stacked = np.stack(bounds, axis=0)
        min_xy = stacked.min(axis=0)
        max_xy = stacked.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1e-6)

        rel = (point - min_xy) / span
        rel = np.clip(rel, 0.0, 1.0)
        return float(rel[0]), float(rel[1])

    def _extract_features(self, landmarks, width: int, height: int) -> EyeTrackingResult | None:
        required_index = max(max(LEFT_IRIS), max(RIGHT_IRIS), max(LEFT_EYE_BOUNDS), max(RIGHT_EYE_BOUNDS))
        if len(landmarks) <= required_index:
            return None

        left_iris_points = [self._to_xy(landmarks, idx, width, height) for idx in LEFT_IRIS]
        right_iris_points = [self._to_xy(landmarks, idx, width, height) for idx in RIGHT_IRIS]
        left_eye_bounds = [self._to_xy(landmarks, idx, width, height) for idx in LEFT_EYE_BOUNDS]
        right_eye_bounds = [self._to_xy(landmarks, idx, width, height) for idx in RIGHT_EYE_BOUNDS]

        left_iris = self._mean_point(left_iris_points)
        right_iris = self._mean_point(right_iris_points)

        lx, ly = self._normalized_position(left_iris, left_eye_bounds)
        rx, ry = self._normalized_position(right_iris, right_eye_bounds)

        x_norm = float(np.clip((lx + rx) * 0.5, 0.0, 1.0))
        y_norm = float(np.clip((ly + ry) * 0.5, 0.0, 1.0))

        return EyeTrackingResult(
            x_norm=x_norm,
            y_norm=y_norm,
            left_iris=(float(left_iris[0]), float(left_iris[1])),
            right_iris=(float(right_iris[0]), float(right_iris[1])),
        )

    def _extract_from_solutions(self, frame_rgb: np.ndarray, width: int, height: int) -> EyeTrackingResult | None:
        output = self._mesh.process(frame_rgb)
        if not output.multi_face_landmarks:
            return None
        landmarks = output.multi_face_landmarks[0].landmark
        return self._extract_features(landmarks, width, height)

    def _extract_from_tasks(self, frame_rgb: np.ndarray, width: int, height: int) -> EyeTrackingResult | None:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = self._next_timestamp_ms()
        output = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not output.face_landmarks:
            return None
        landmarks = output.face_landmarks[0]
        return self._extract_features(landmarks, width, height)

    def process(self, frame_bgr: np.ndarray) -> EyeTrackingResult | None:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "solutions":
            return self._extract_from_solutions(frame_rgb, width, height)

        return self._extract_from_tasks(frame_rgb, width, height)
