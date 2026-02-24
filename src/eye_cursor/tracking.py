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
    left_x_norm: float
    left_y_norm: float
    right_x_norm: float
    right_y_norm: float


class EyeTracker:
    def __init__(self, min_confidence: float = 0.2) -> None:
        self._request = Vision.VNDetectFaceLandmarksRequest.alloc().init()
        self._min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self._color_space = Quartz.CGColorSpaceCreateDeviceRGB()
        self.last_status = "Need both eyes visible"

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
    def _landmark_region(landmarks, selector: str):
        method = getattr(landmarks, selector, None)
        if method is None or not callable(method):
            return None
        return method()

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
    def _face_point_to_frame_point(
        face_point: tuple[float, float] | None,
        face_rect: tuple[float, float, float, float],
        frame_shape: tuple[int, ...],
    ) -> tuple[float, float] | None:
        if face_point is None:
            return None

        frame_h, frame_w = frame_shape[:2]
        if frame_w <= 1 or frame_h <= 1:
            return None

        face_x, face_y, face_width, face_height = face_rect
        point_x, point_y = face_point

        x_norm = face_x + point_x * face_width
        y_norm_bottom = face_y + point_y * face_height

        frame_x = float(np.clip(x_norm * (frame_w - 1), 0.0, frame_w - 1))
        frame_y = float(np.clip((1.0 - y_norm_bottom) * (frame_h - 1), 0.0, frame_h - 1))
        return frame_x, frame_y

    @staticmethod
    def _face_region_to_frame_points(
        face_points: list[tuple[float, float]],
        face_rect: tuple[float, float, float, float],
        frame_shape: tuple[int, ...],
    ) -> list[tuple[float, float]]:
        frame_points: list[tuple[float, float]] = []
        for face_point in face_points:
            frame_point = EyeTracker._face_point_to_frame_point(face_point, face_rect, frame_shape)
            if frame_point is not None:
                frame_points.append(frame_point)
        return frame_points

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
    def _region_ratio(
        point: tuple[float, float] | None,
        region_points: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        if point is None or len(region_points) < 3:
            return None

        points_array = np.array(region_points, dtype=float)
        x_min = float(np.percentile(points_array[:, 0], 5))
        x_max = float(np.percentile(points_array[:, 0], 95))
        y_min = float(np.percentile(points_array[:, 1], 5))
        y_max = float(np.percentile(points_array[:, 1], 95))

        x_span = x_max - x_min
        y_span = y_max - y_min
        if x_span <= 1e-4 or y_span <= 1e-4:
            return None

        ratio_x = (point[0] - x_min) / x_span
        ratio_y = (point[1] - y_min) / y_span
        return (
            float(np.clip(ratio_x, 0.0, 1.0)),
            float(np.clip(ratio_y, 0.0, 1.0)),
        )

    @staticmethod
    def _detect_pupil_center(
        frame_gray: np.ndarray,
        eye_points_frame: list[tuple[float, float]],
    ) -> tuple[float, float] | None:
        if len(eye_points_frame) < 3:
            return None

        points_array = np.array(eye_points_frame, dtype=np.float32)
        frame_h, frame_w = frame_gray.shape[:2]

        min_x = max(int(np.floor(points_array[:, 0].min())) - 2, 0)
        max_x = min(int(np.ceil(points_array[:, 0].max())) + 2, frame_w - 1)
        min_y = max(int(np.floor(points_array[:, 1].min())) - 2, 0)
        max_y = min(int(np.ceil(points_array[:, 1].max())) + 2, frame_h - 1)
        if min_x >= max_x or min_y >= max_y:
            return None

        roi = frame_gray[min_y : max_y + 1, min_x : max_x + 1]
        if roi.size == 0:
            return None

        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        local_points = np.round(points_array - np.array([min_x, min_y], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [local_points], 255)

        valid_pixel_count = int(np.count_nonzero(mask))
        if valid_pixel_count < 20:
            return None

        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        masked_values = blurred[mask > 0]
        if masked_values.size < 20:
            return None

        dark_threshold = float(np.percentile(masked_values, 25))
        dark_mask = np.zeros_like(mask)
        dark_mask[(blurred <= dark_threshold) & (mask > 0)] = 255

        kernel = np.ones((3, 3), dtype=np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        if int(np.count_nonzero(dark_mask)) < 10:
            fallback_count = int(max(8, masked_values.size * 0.08))
            fallback_count = min(fallback_count, masked_values.size)
            if fallback_count <= 0:
                return None

            partition_index = fallback_count - 1
            fallback_threshold = float(np.partition(masked_values, partition_index)[partition_index])
            dark_mask[(blurred <= fallback_threshold) & (mask > 0)] = 255

        moments = cv2.moments(dark_mask, binaryImage=True)
        if moments["m00"] > 0:
            local_x = float(moments["m10"] / moments["m00"])
            local_y = float(moments["m01"] / moments["m00"])
        else:
            ys, xs = np.where(mask > 0)
            if xs.size == 0:
                return None

            masked_intensity = blurred[ys, xs]
            fallback_count = int(max(8, xs.size * 0.08))
            fallback_count = min(fallback_count, xs.size)
            if fallback_count <= 0:
                return None

            selected = np.argpartition(masked_intensity, fallback_count - 1)[:fallback_count]
            local_x = float(xs[selected].mean())
            local_y = float(ys[selected].mean())

        return (
            float(np.clip(min_x + local_x, 0.0, frame_w - 1)),
            float(np.clip(min_y + local_y, 0.0, frame_h - 1)),
        )

    def process(self, frame_bgr: np.ndarray) -> TrackingResult | None:
        cgimage = self._to_cgimage(frame_bgr, self._color_space)
        if cgimage is None:
            self.last_status = "Invalid camera frame"
            return None

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, {})
        ok, error = handler.performRequests_error_([self._request], None)
        if not ok or error is not None:
            self.last_status = "Vision request failed"
            return None

        results = self._request.results()
        if results is None or len(results) == 0:
            self.last_status = "No face detected"
            return None

        observation = max(results, key=lambda item: float(item.confidence()))
        if float(observation.confidence()) < self._min_confidence:
            self.last_status = "Face confidence too low"
            return None

        landmarks = observation.landmarks()
        if landmarks is None:
            self.last_status = "Face landmarks unavailable"
            return None

        face_rect = self._to_xywh(observation.boundingBox())
        if face_rect is None:
            self.last_status = "Face bounding box unavailable"
            return None

        left_eye_face_points = self._region_points(self._landmark_region(landmarks, "leftEye"))
        right_eye_face_points = self._region_points(self._landmark_region(landmarks, "rightEye"))
        if not left_eye_face_points or not right_eye_face_points:
            self.last_status = "Need both eyes visible"
            return None

        left_eye_frame_points = self._face_region_to_frame_points(left_eye_face_points, face_rect, frame_bgr.shape)
        right_eye_frame_points = self._face_region_to_frame_points(right_eye_face_points, face_rect, frame_bgr.shape)
        if len(left_eye_frame_points) < 3 or len(right_eye_frame_points) < 3:
            self.last_status = "Need both eyes visible"
            return None

        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        left_pupil_frame = self._detect_pupil_center(frame_gray, left_eye_frame_points)
        right_pupil_frame = self._detect_pupil_center(frame_gray, right_eye_frame_points)

        if left_pupil_frame is None:
            left_pupil_face_points = self._region_points(self._landmark_region(landmarks, "leftPupil"))
            left_pupil_face_center = self._center(left_pupil_face_points)
            left_pupil_frame = self._face_point_to_frame_point(left_pupil_face_center, face_rect, frame_bgr.shape)

        if right_pupil_frame is None:
            right_pupil_face_points = self._region_points(self._landmark_region(landmarks, "rightPupil"))
            right_pupil_face_center = self._center(right_pupil_face_points)
            right_pupil_frame = self._face_point_to_frame_point(right_pupil_face_center, face_rect, frame_bgr.shape)
        if left_pupil_frame is None or right_pupil_frame is None:
            self.last_status = "Unable to detect both pupils"
            return None

        left_ratio = self._region_ratio(left_pupil_frame, left_eye_frame_points)
        right_ratio = self._region_ratio(right_pupil_frame, right_eye_frame_points)
        if left_ratio is None or right_ratio is None:
            self.last_status = "Unable to estimate both eye directions"
            return None

        frame_h, frame_w = frame_bgr.shape[:2]
        if frame_w <= 1 or frame_h <= 1:
            self.last_status = "Invalid frame size"
            return None

        gaze_x = (left_ratio[0] + right_ratio[0]) * 0.5
        gaze_y = (left_ratio[1] + right_ratio[1]) * 0.5
        left_x_norm = float(np.clip(left_pupil_frame[0] / (frame_w - 1), 0.0, 1.0))
        left_y_norm = float(np.clip(left_pupil_frame[1] / (frame_h - 1), 0.0, 1.0))
        right_x_norm = float(np.clip(right_pupil_frame[0] / (frame_w - 1), 0.0, 1.0))
        right_y_norm = float(np.clip(right_pupil_frame[1] / (frame_h - 1), 0.0, 1.0))

        self.last_status = "Tracking both eyes"
        return TrackingResult(
            x_norm=float(np.clip(gaze_x, 0.0, 1.0)),
            y_norm=float(np.clip(gaze_y, 0.0, 1.0)),
            left_x_norm=left_x_norm,
            left_y_norm=left_y_norm,
            right_x_norm=right_x_norm,
            right_y_norm=right_y_norm,
        )
