from __future__ import annotations

import argparse

import cv2
import numpy as np

from hand_cursor.macos_mouse import MacOSMouseController

from .calibration import CalibrationSession
from .config import EyeCursorConfig
from .mapping import (
    apply_precision_curve,
    blend_point,
    clamp_sensitivity,
    distance,
    expand_uncalibrated,
    stabilize_norm,
)
from .tracking import EyeTracker, TrackingResult

DEBUG_WINDOW = "Eye Cursor Debug"


def _build_arg_parser() -> argparse.ArgumentParser:
    default_config = EyeCursorConfig()

    parser = argparse.ArgumentParser(description="Control macOS cursor with eye tracking")
    parser.add_argument("--camera-index", type=int, default=default_config.camera_index, help="Camera index for OpenCV")
    parser.add_argument("--frame-width", type=int, default=default_config.frame_width, help="Capture width")
    parser.add_argument("--frame-height", type=int, default=default_config.frame_height, help="Capture height")
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=default_config.cursor_sensitivity,
        help="Cursor sensitivity around screen center (0.4 to 3.0)",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=default_config.smooth_alpha,
        help="Smoothing alpha for cursor movement (lower is smoother)",
    )
    parser.add_argument(
        "--deadzone-px",
        type=float,
        default=default_config.move_deadzone_px,
        help="Minimum cursor movement in pixels before updating pointer",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=default_config.min_confidence,
        help="Minimum face landmark confidence required (0.0 to 1.0)",
    )
    parser.add_argument(
        "--gaze-deadband",
        type=float,
        default=default_config.gaze_deadband_norm,
        help="Ignore tiny normalized eye jitter below this threshold",
    )
    parser.add_argument(
        "--curve-gamma",
        type=float,
        default=default_config.curve_gamma,
        help="Precision curve gamma (>= 1.0, higher gives finer center control)",
    )
    parser.add_argument(
        "--uncalibrated-gain",
        type=float,
        default=default_config.uncalibrated_gain,
        help="Gain used before calibration to make movement responsive",
    )
    parser.add_argument(
        "--eye-zoom",
        type=float,
        default=default_config.eye_zoom,
        help="Eye ROI zoom used for pupil detection (1.0 to 10.0)",
    )
    parser.add_argument(
        "--no-preview-mirror",
        action="store_true",
        help="Disable mirrored preview/cursor mapping",
    )
    return parser


def _open_camera(camera_index: int, frame_width: int, frame_height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capture.set(cv2.CAP_PROP_FPS, 60)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _extract_eye_patch(
    frame: np.ndarray,
    center_norm: tuple[float, float],
    radius_px: int,
) -> np.ndarray | None:
    frame_h, frame_w = frame.shape[:2]
    if frame_h <= 2 or frame_w <= 2:
        return None

    cx = int(np.clip(center_norm[0] * frame_w, 0, frame_w - 1))
    cy = int(np.clip(center_norm[1] * frame_h, 0, frame_h - 1))
    radius_px = int(max(8, radius_px))

    x0 = max(cx - radius_px, 0)
    x1 = min(cx + radius_px, frame_w - 1)
    y0 = max(cy - radius_px, 0)
    y1 = min(cy + radius_px, frame_h - 1)
    if x0 >= x1 or y0 >= y1:
        return None

    patch = frame[y0 : y1 + 1, x0 : x1 + 1]
    if patch.size == 0:
        return None
    return patch


def _draw_eye_zoom_insets(
    frame: np.ndarray,
    left_norm: tuple[float, float] | None,
    right_norm: tuple[float, float] | None,
) -> None:
    if left_norm is None or right_norm is None:
        return

    frame_h, frame_w = frame.shape[:2]
    patch_size = int(np.clip(min(frame_w, frame_h) * 0.22, 110, 220))
    radius_px = int(np.clip(min(frame_w, frame_h) * 0.07, 18, 60))
    margin = 12

    left_patch = _extract_eye_patch(frame, left_norm, radius_px)
    right_patch = _extract_eye_patch(frame, right_norm, radius_px)
    if left_patch is None or right_patch is None:
        return

    left_patch = cv2.resize(left_patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)
    right_patch = cv2.resize(right_patch, (patch_size, patch_size), interpolation=cv2.INTER_CUBIC)

    total_width = (patch_size * 2) + margin
    if total_width + margin > frame_w or patch_size + margin > frame_h:
        return

    x0 = frame_w - total_width - margin
    y0 = margin

    frame[y0 : y0 + patch_size, x0 : x0 + patch_size] = left_patch
    frame[y0 : y0 + patch_size, x0 + patch_size + margin : x0 + (patch_size * 2) + margin] = right_patch
    cv2.rectangle(frame, (x0, y0), (x0 + patch_size, y0 + patch_size), (0, 255, 0), 2)
    cv2.rectangle(
        frame,
        (x0 + patch_size + margin, y0),
        (x0 + (patch_size * 2) + margin, y0 + patch_size),
        (255, 200, 0),
        2,
    )
    cv2.putText(frame, "L", (x0 + 6, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(
        frame,
        "R",
        (x0 + patch_size + margin + 6, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 200, 0),
        2,
    )


def _draw_debug(
    frame: np.ndarray,
    tracking: TrackingResult | None,
    display_center_norm: tuple[float, float] | None,
    display_left_norm: tuple[float, float] | None,
    display_right_norm: tuple[float, float] | None,
    status: str,
    paused: bool,
    mapped_point: tuple[float, float] | None,
    screen_size: tuple[int, int],
    sensitivity: float,
    smooth_alpha: float,
    curve_gamma: float,
    gaze_deadband_norm: float,
    eye_zoom: float,
    calibration: CalibrationSession,
) -> np.ndarray:
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    eyes_text = "Eyes: not detected"
    if tracking is not None and display_center_norm is not None:
        center_x = int(np.clip(display_center_norm[0] * frame_w, 0, frame_w - 1))
        center_y = int(np.clip(display_center_norm[1] * frame_h, 0, frame_h - 1))
        cv2.circle(out, (center_x, center_y), 8, (0, 255, 255), 2)
        if display_left_norm is not None:
            left_x = int(np.clip(display_left_norm[0] * frame_w, 0, frame_w - 1))
            left_y = int(np.clip(display_left_norm[1] * frame_h, 0, frame_h - 1))
            cv2.circle(out, (left_x, left_y), 6, (0, 255, 0), 2)
        if display_right_norm is not None:
            right_x = int(np.clip(display_right_norm[0] * frame_w, 0, frame_w - 1))
            right_y = int(np.clip(display_right_norm[1] * frame_h, 0, frame_h - 1))
            cv2.circle(out, (right_x, right_y), 6, (255, 200, 0), 2)
        eyes_text = f"Eyes: ({tracking.x_norm:.3f}, {tracking.y_norm:.3f})"
        _draw_eye_zoom_insets(out, display_left_norm, display_right_norm)

    if mapped_point is not None:
        sx, sy = mapped_point
        sw, sh = screen_size
        preview_x = int(np.clip((sx / max(sw, 1)) * frame_w, 0, frame_w - 1))
        preview_y = int(np.clip((sy / max(sh, 1)) * frame_h, 0, frame_h - 1))
        cv2.circle(out, (preview_x, preview_y), 10, (0, 255, 255), 2)

    if calibration.is_active:
        target = calibration.current_target()
        if target is not None:
            target_x = int(np.clip(target[0] * frame_w, 0, frame_w - 1))
            target_y = int(np.clip(target[1] * frame_h, 0, frame_h - 1))
            cv2.circle(out, (target_x, target_y), 12, (0, 0, 255), -1)
            cv2.circle(out, (target_x, target_y), 18, (0, 0, 255), 2)

    if calibration.is_active:
        calibration_text = calibration.current_prompt()
    elif calibration.is_calibrated:
        calibration_text = "Calibration: complete (r reset)"
    else:
        calibration_text = "Calibration: press c to start"

    mode = "PAUSED" if paused else "LIVE"
    cv2.putText(out, eyes_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, f"Mode: {mode} | Sensitivity: {sensitivity:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, f"Smooth alpha: {smooth_alpha:.2f} | Gamma: {curve_gamma:.2f}", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, f"Gaze deadband: {gaze_deadband_norm:.4f} | Eye zoom: {eye_zoom:.1f}x", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, calibration_text, (12, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 220, 160), 2)
    cv2.putText(out, status, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
    cv2.putText(out, "Keys: +/- sensitivity | ,/. smoothing | z/x eye zoom | c calibrate | r reset | p pause | q quit", (12, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 2)
    return out


def run(config: EyeCursorConfig) -> int:
    cv2.setUseOptimized(True)

    mouse = MacOSMouseController()
    calibration = CalibrationSession()

    try:
        tracker = EyeTracker(min_confidence=config.min_confidence, eye_zoom=config.eye_zoom)
        tracker.set_eye_zoom(float(np.clip(config.eye_zoom, config.min_eye_zoom, config.max_eye_zoom)))
    except Exception as exc:
        print(f"Failed to initialize eye tracker: {exc}")
        return 1

    cap = _open_camera(config.camera_index, config.frame_width, config.frame_height)
    if not cap.isOpened():
        print("Failed to open webcam. Check camera index/permissions.")
        tracker.close()
        return 1

    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW, config.frame_width, config.frame_height)

    paused = False
    smooth_alpha = float(np.clip(config.smooth_alpha, 0.05, 0.95))
    sensitivity = clamp_sensitivity(config.cursor_sensitivity, config.min_sensitivity, config.max_sensitivity)
    gaze_deadband_norm = float(np.clip(config.gaze_deadband_norm, 0.0, 0.05))
    curve_gamma = float(np.clip(config.curve_gamma, 1.0, 4.0))
    uncalibrated_gain = float(np.clip(config.uncalibrated_gain, 1.0, 4.0))
    status = "Eye tracking active. Press c to calibrate for precision."

    stabilized_norm: tuple[float, float] | None = None
    smoothed_point: tuple[float, float] | None = None
    last_sent_point: tuple[float, float] | None = None
    last_input_norm: tuple[float, float] | None = None

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                status = "Camera frame read failed."
                break

            tracking = tracker.process(raw_frame)
            display_frame = cv2.flip(raw_frame, 1) if config.preview_mirror else raw_frame

            display_center_norm: tuple[float, float] | None = None
            display_left_norm: tuple[float, float] | None = None
            display_right_norm: tuple[float, float] | None = None
            mapped_point: tuple[float, float] | None = None
            last_input_norm = None

            if tracking is not None:
                stabilized_norm = stabilize_norm(
                    stabilized_norm,
                    (tracking.x_norm, tracking.y_norm),
                    gaze_deadband_norm,
                )

                input_x_norm = 1.0 - stabilized_norm[0] if config.preview_mirror else stabilized_norm[0]
                input_y_norm = stabilized_norm[1]
                last_input_norm = (input_x_norm, input_y_norm)
                display_center_norm = (input_x_norm, input_y_norm)
                display_left_norm = (
                    1.0 - tracking.left_x_norm if config.preview_mirror else tracking.left_x_norm,
                    tracking.left_y_norm,
                )
                display_right_norm = (
                    1.0 - tracking.right_x_norm if config.preview_mirror else tracking.right_x_norm,
                    tracking.right_y_norm,
                )

                if calibration.map is not None:
                    mapped_input_x, mapped_input_y = calibration.map.apply(input_x_norm, input_y_norm)
                else:
                    mapped_input_x = expand_uncalibrated(input_x_norm, uncalibrated_gain)
                    mapped_input_y = expand_uncalibrated(input_y_norm, uncalibrated_gain)

                cursor_x_norm = apply_precision_curve(mapped_input_x, sensitivity, curve_gamma)
                cursor_y_norm = apply_precision_curve(mapped_input_y, sensitivity, curve_gamma)

                mapped_x = cursor_x_norm * (mouse.screen.width - 1)
                mapped_y = cursor_y_norm * (mouse.screen.height - 1)
                mapped_point = mouse.clamp(mapped_x, mapped_y)

                smoothed_point = blend_point(smoothed_point, mapped_point, smooth_alpha)
                if not paused and not calibration.is_active and smoothed_point is not None:
                    if last_sent_point is None or distance(smoothed_point, last_sent_point) >= config.move_deadzone_px:
                        mouse.move(smoothed_point[0], smoothed_point[1])
                        last_sent_point = smoothed_point

                if calibration.is_active:
                    status = calibration.current_prompt()
                else:
                    status = "Eye tracking active (calibrated)" if calibration.is_calibrated else "Eye tracking active (press c to calibrate)"
            else:
                stabilized_norm = None
                if calibration.is_active:
                    status = f"{calibration.current_prompt()} | {tracker.last_status}"
                else:
                    status = tracker.last_status

            debug = _draw_debug(
                frame=display_frame,
                tracking=tracking,
                display_center_norm=display_center_norm,
                display_left_norm=display_left_norm,
                display_right_norm=display_right_norm,
                status=status,
                paused=paused,
                mapped_point=smoothed_point,
                screen_size=(mouse.screen.width, mouse.screen.height),
                sensitivity=sensitivity,
                smooth_alpha=smooth_alpha,
                curve_gamma=curve_gamma,
                gaze_deadband_norm=gaze_deadband_norm,
                eye_zoom=tracker.eye_zoom,
                calibration=calibration,
            )
            cv2.imshow(DEBUG_WINDOW, debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                status = "Paused" if paused else "Resumed"
            if key in (ord("-"), ord("_"), ord("[")):
                sensitivity = clamp_sensitivity(sensitivity - config.sensitivity_step, config.min_sensitivity, config.max_sensitivity)
                status = f"Sensitivity: {sensitivity:.2f}"
            if key in (ord("+"), ord("="), ord("]")):
                sensitivity = clamp_sensitivity(sensitivity + config.sensitivity_step, config.min_sensitivity, config.max_sensitivity)
                status = f"Sensitivity: {sensitivity:.2f}"
            if key in (ord(","), ord("<")):
                smooth_alpha = float(np.clip(smooth_alpha - 0.02, 0.05, 0.95))
                status = f"Smooth alpha: {smooth_alpha:.2f}"
            if key in (ord("."), ord(">")):
                smooth_alpha = float(np.clip(smooth_alpha + 0.02, 0.05, 0.95))
                status = f"Smooth alpha: {smooth_alpha:.2f}"
            if key in (ord("z"), ord("Z")):
                tracker.set_eye_zoom(float(np.clip(tracker.eye_zoom + config.eye_zoom_step, config.min_eye_zoom, config.max_eye_zoom)))
                status = f"Eye zoom: {tracker.eye_zoom:.1f}x"
            if key in (ord("x"), ord("X")):
                tracker.set_eye_zoom(float(np.clip(tracker.eye_zoom - config.eye_zoom_step, config.min_eye_zoom, config.max_eye_zoom)))
                status = f"Eye zoom: {tracker.eye_zoom:.1f}x"
            if key in (ord("c"), ord("C")):
                status = calibration.start()
            if key in (ord("r"), ord("R")):
                status = calibration.reset()
            if key == ord(" ") and calibration.is_active:
                status = calibration.capture(last_input_norm)

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = EyeCursorConfig(
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        preview_mirror=not args.no_preview_mirror,
        smooth_alpha=args.smooth_alpha,
        move_deadzone_px=args.deadzone_px,
        cursor_sensitivity=args.sensitivity,
        min_confidence=args.min_confidence,
        gaze_deadband_norm=args.gaze_deadband,
        curve_gamma=args.curve_gamma,
        uncalibrated_gain=args.uncalibrated_gain,
        eye_zoom=args.eye_zoom,
    )
    raise SystemExit(run(config))


if __name__ == "__main__":
    main()
