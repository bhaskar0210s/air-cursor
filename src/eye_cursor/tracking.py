from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import Quartz
import Vision


@dataclass(frozen=True)
class TrackingResult:
    x_norm: float
    y_norm: float


class EyeTracker:
    def __init__(self, min_confidence: float = 0.2) -> None:
        self._request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        self._min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self._color_space = Quartz.CGColorSpaceCreateDeviceRGB()

    def close(self) -> None:
        return

    @staticmethod
    def _to_cgimage(frame_bgr: np.ndarray, color_space) -> Quartz.CGImageRef | None:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            return None

        frame_rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        height, width = frame_rgba.shape[:2]
        bytes_per_row = width * 4

        image_bytes = frame_rgba.tobytes()
        provider = Quartz.CGDataProviderCreateWithData(None, image_bytes, len(image_bytes), None)
        bitmap_info = Quartz.kCGImageAlphaNoneSkipLast | Quartz.kCGBitmapByteOrderDefault

        return Quartz.CGImageCreate(
            width,
            height,
            8,
            32,
            bytes_per_row,
            color_space,
            bitmap_info,
            provider,
            None,
            False,
            Quartz.kCGRenderingIntentDefault,
        )

    @staticmethod
    def _to_xy(point) -> tuple[float, float] | None:
        if point is None:
            return None

        if hasattr(point, "x") and hasattr(point, "y"):
            point_x = point.x() if callable(point.x) else point.x
            point_y = point.y() if callable(point.y) else point.y
            return float(point_x), float(point_y)

        if isinstance(point, (tuple, list)) and len(point) >= 2:
            return float(point[0]), float(point[1])

        return None

    @staticmethod
    def _to_xywh(rect) -> tuple[float, float, float, float] | None:
        if rect is None:
            return None

        if hasattr(rect, "origin") and hasattr(rect, "size"):
            ox = rect.origin.x() if callable(rect.origin.x) else rect.origin.x
            oy = rect.origin.y() if callable(rect.origin.y) else rect.origin.y
            width = rect.size.width() if callable(rect.size.width) else rect.size.width
            height = rect.size.height() if callable(rect.size.height) else rect.size.height
            return float(ox), float(oy), float(width), float(height)

        if isinstance(rect, (tuple, list)) and len(rect) >= 2:
            origin = EyeTracker._to_xy(rect[0])
            size = EyeTracker._to_xy(rect[1])
            if origin is None or size is None:
                return None
            return origin[0], origin[1], size[0], size[1]

        return None

    @staticmethod
    def _region_points(region) -> list[tuple[float, float]]:
        if region is None:
            return []

        point_count = int(region.pointCount())
        if point_count <= 0:
            return []

        points_ptr = region.normalizedPoints()
        if points_ptr is None:
            return []

        out: list[tuple[float, float]] = []
        for index in range(point_count):
            point = EyeTracker._to_xy(points_ptr[index])
            if point is None:
                continue
            out.append(
                (
                    float(np.clip(point[0], 0.0, 1.0)),
                    float(np.clip(point[1], 0.0, 1.0)),
                )
            )
        return out

    @staticmethod
    def _center(points: list[tuple[float, float]]) -> tuple[float, float] | None:
        if not points:
            return None

        points_array = np.array(points, dtype=float)
        return (
            float(np.clip(points_array[:, 0].mean(), 0.0, 1.0)),
            float(np.clip(points_array[:, 1].mean(), 0.0, 1.0)),
        )

    @staticmethod
    def _landmark_center(landmarks, pupil_selector: str, eye_selector: str) -> tuple[float, float] | None:
        pupil_region = getattr(landmarks, pupil_selector)()
        center = EyeTracker._center(EyeTracker._region_points(pupil_region))
        if center is not None:
            return center

        eye_region = getattr(landmarks, eye_selector)()
        return EyeTracker._center(EyeTracker._region_points(eye_region))

    def process(self, frame_bgr: np.ndarray) -> TrackingResult | None:
        cgimage = self._to_cgimage(frame_bgr, self._color_space)
        if cgimage is None:
            return None

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, {})
        ok, error = handler.performRequests_error_([self._request], None)
        if not ok or error is not None:
            return None

        results = self._request.results()
        if results is None or len(results) == 0:
            return None

        observation = max(results, key=lambda item: float(item.confidence()))
        if float(observation.confidence()) < self._min_confidence:
            return None

        landmarks = observation.landmarks()
        if landmarks is None:
            return None

        left_eye_center = self._landmark_center(landmarks, "leftPupil", "leftEye")
        right_eye_center = self._landmark_center(landmarks, "rightPupil", "rightEye")

        if left_eye_center is None and right_eye_center is None:
            return None

        if left_eye_center is None:
            left_eye_center = right_eye_center
        if right_eye_center is None:
            right_eye_center = left_eye_center

        if left_eye_center is None or right_eye_center is None:
            return None

        eyes_x = (left_eye_center[0] + right_eye_center[0]) * 0.5
        eyes_y = (left_eye_center[1] + right_eye_center[1]) * 0.5

        face_rect = self._to_xywh(observation.boundingBox())
        if face_rect is None:
            return None

        face_x, face_y, face_width, face_height = face_rect
        x_norm = float(np.clip(face_x + eyes_x * face_width, 0.0, 1.0))
        y_norm_bottom = float(np.clip(face_y + eyes_y * face_height, 0.0, 1.0))

        return TrackingResult(
            x_norm=x_norm,
            y_norm=float(np.clip(1.0 - y_norm_bottom, 0.0, 1.0)),
        )
