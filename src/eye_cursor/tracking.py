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
    handedness: str


class HandTracker:
    def __init__(self, fingertip_landmark_index: int = 8, min_confidence: float = 0.2) -> None:
        self._request = Vision.VNDetectHumanHandPoseRequest.alloc().init()
        self._request.setMaximumHandCount_(1)

        self._min_confidence = float(np.clip(min_confidence, 0.0, 1.0))
        self._fingertip_landmark_index = fingertip_landmark_index

        self._joint_for_index = {
            4: Vision.VNHumanHandPoseObservationJointNameThumbTip,
            8: Vision.VNHumanHandPoseObservationJointNameIndexTip,
            12: Vision.VNHumanHandPoseObservationJointNameMiddleTip,
            16: Vision.VNHumanHandPoseObservationJointNameRingTip,
            20: Vision.VNHumanHandPoseObservationJointNameLittleTip,
        }.get(fingertip_landmark_index, Vision.VNHumanHandPoseObservationJointNameIndexTip)

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
    def _map_chirality(chirality: int) -> str:
        if chirality == Vision.VNChiralityLeft:
            return "left"
        if chirality == Vision.VNChiralityRight:
            return "right"
        return "unknown"

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

        observation = results[0]
        point, point_error = observation.recognizedPointForJointName_error_(self._joint_for_index, None)
        if point_error is not None or point is None:
            return None

        confidence = float(point.confidence())
        if confidence < self._min_confidence:
            return None

        x_norm = float(np.clip(point.x(), 0.0, 1.0))

        # Vision uses bottom-left origin for normalized points; convert to top-left.
        y_norm = float(np.clip(1.0 - point.y(), 0.0, 1.0))

        handedness = self._map_chirality(int(observation.chirality()))

        return TrackingResult(x_norm=x_norm, y_norm=y_norm, handedness=handedness)
