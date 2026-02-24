from __future__ import annotations

import argparse

import cv2
import numpy as np

from air_cursor.macos_mouse import MacOSMouseController

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

    if mapped_point is not None:
        sx, sy = mapped_point
        sw, sh = screen_size
        preview_x = int(np.clip((sx / max(sw, 1)) * frame_w, 0, frame_w - 1))
        preview_y = int(np.clip((sy / max(sh, 1)) * frame_h, 0, frame_h - 1))
        cv2.circle(out, (preview_x, preview_y), 10, (0, 255, 255), 2)

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
    cv2.putText(out, f"Gaze deadband: {gaze_deadband_norm:.4f}", (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, calibration_text, (12, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 220, 160), 2)
    cv2.putText(out, status, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
    cv2.putText(out, "Keys: +/- sensitivity | ,/. smoothing | c calibrate | r reset | p pause | q quit", (12, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 2)
    return out


def run(config: EyeCursorConfig) -> int:
    cv2.setUseOptimized(True)

    mouse = MacOSMouseController()
    calibration = CalibrationSession()

    try:
        tracker = EyeTracker(min_confidence=config.min_confidence)
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
    )
    raise SystemExit(run(config))


if __name__ == "__main__":
    main()
