from __future__ import annotations

import argparse

import cv2
import numpy as np

from .calibration import CalibrationModel, Calibrator
from .config import AppConfig
from .macos_mouse import MacOSMouseController
from .tracking import EyeTracker, EyeTrackingResult, HandTracker, TrackingResult

DEBUG_WINDOW = "Cursor Fusion Debug"
CALIB_WINDOW = "Cursor Fusion Calibration"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control macOS cursor with hand + eye tracking")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for OpenCV")
    parser.add_argument("--frame-width", type=int, default=640, help="Capture width")
    parser.add_argument("--frame-height", type=int, default=480, help="Capture height")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional local path to MediaPipe hand_landmarker.task",
    )
    parser.add_argument(
        "--eye-model-path",
        type=str,
        default=None,
        help="Optional local path to MediaPipe face_landmarker.task",
    )
    parser.add_argument(
        "--eye-fusion-weight",
        type=float,
        default=0.25,
        help="Eye influence when both hand and eye are available (0.0 to 1.0)",
    )
    return parser


def _open_camera(camera_index: int, frame_width: int, frame_height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capture.set(cv2.CAP_PROP_FPS, 60)
    return capture


def _blend_point(previous: tuple[float, float] | None, current: tuple[float, float], alpha: float) -> tuple[float, float]:
    if previous is None:
        return current
    x = previous[0] * (1.0 - alpha) + current[0] * alpha
    y = previous[1] * (1.0 - alpha) + current[1] * alpha
    return x, y


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))


def _normalize_handedness(label: str, swap_labels: bool) -> str:
    lowered = label.lower().strip()
    if lowered not in {"left", "right"}:
        return "unknown"
    if not swap_labels:
        return lowered
    return "right" if lowered == "left" else "left"


def _fuse_points(
    hand_point: tuple[float, float] | None,
    eye_point: tuple[float, float] | None,
    eye_weight: float,
) -> tuple[float, float] | None:
    if hand_point is not None and eye_point is not None:
        weight = float(np.clip(eye_weight, 0.0, 1.0))
        x = hand_point[0] * (1.0 - weight) + eye_point[0] * weight
        y = hand_point[1] * (1.0 - weight) + eye_point[1] * weight
        return x, y

    if hand_point is not None:
        return hand_point

    return eye_point


def _draw_debug(
    frame: np.ndarray,
    hand_tracking: TrackingResult | None,
    eye_tracking: EyeTrackingResult | None,
    display_hand_x_norm: float | None,
    status: str,
    paused: bool,
    mapped_point: tuple[float, float] | None,
    screen_size: tuple[int, int],
    is_calibrated: bool,
    is_calibrating: bool,
    swap_handedness: bool,
    eye_fusion_weight: float,
    hand_samples: int,
    eye_samples: int,
    samples_required: int,
) -> np.ndarray:
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    if hand_tracking is not None and display_hand_x_norm is not None:
        px = int(np.clip(display_hand_x_norm * frame_w, 0, frame_w - 1))
        py = int(np.clip(hand_tracking.y_norm * frame_h, 0, frame_h - 1))
        cv2.circle(out, (px, py), 8, (0, 255, 0), 2)

    if mapped_point is not None:
        sx, sy = mapped_point
        sw, sh = screen_size
        preview_x = int(np.clip((sx / max(sw, 1)) * frame_w, 0, frame_w - 1))
        preview_y = int(np.clip((sy / max(sh, 1)) * frame_h, 0, frame_h - 1))
        cv2.circle(out, (preview_x, preview_y), 10, (0, 255, 255), 2)

    mode = "PAUSED" if paused else "LIVE"
    calibration_state = "READY" if is_calibrated else ("CALIBRATING" if is_calibrating else "NOT CALIBRATED")
    handedness_state = "ON" if swap_handedness else "OFF"

    hand_text = "Hand: not detected"
    if hand_tracking is not None:
        hand_label = _normalize_handedness(hand_tracking.handedness, swap_handedness)
        hand_text = f"Hand: ({hand_tracking.x_norm:.3f}, {hand_tracking.y_norm:.3f}) [{hand_label}]"

    eye_text = "Eye: not detected"
    if eye_tracking is not None:
        eye_text = f"Eye: ({eye_tracking.x_norm:.3f}, {eye_tracking.y_norm:.3f})"

    cv2.putText(out, hand_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
    cv2.putText(out, eye_text, (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
    cv2.putText(out, f"Mode: {mode} | Calibration: {calibration_state}", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
    cv2.putText(out, f"Handedness swap: {handedness_state} | Eye weight: {eye_fusion_weight:.2f}", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
    cv2.putText(out, f"Calibration samples - hand {hand_samples}/{samples_required}, eye {eye_samples}/{samples_required}", (12, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
    cv2.putText(out, status, (12, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 2)
    cv2.putText(out, "Keys: c calibrate | space capture | h hand swap | p pause | q quit", (12, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2)

    return out


def run(config: AppConfig) -> int:
    mouse = MacOSMouseController()
    try:
        hand_tracker = HandTracker(
            model_path=config.model_path,
            fingertip_landmark_index=config.fingertip_landmark_index,
        )
        eye_tracker = EyeTracker(model_path=config.eye_model_path)
    except Exception as exc:
        print(f"Failed to initialize trackers: {exc}")
        return 1

    hand_calibrator = Calibrator(
        screen_width=mouse.screen.width,
        screen_height=mouse.screen.height,
        grid_size=config.calibration_grid_size,
        samples_required=config.calibration_samples_required,
    )
    eye_calibrator = Calibrator(
        screen_width=mouse.screen.width,
        screen_height=mouse.screen.height,
        grid_size=config.calibration_grid_size,
        samples_required=config.calibration_samples_required,
    )

    cap = _open_camera(config.camera_index, config.frame_width, config.frame_height)
    if not cap.isOpened():
        print("Failed to open webcam. Check camera index/permissions.")
        hand_tracker.close()
        eye_tracker.close()
        return 1

    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW, config.frame_width, config.frame_height)

    hand_model: CalibrationModel | None = None
    eye_model: CalibrationModel | None = None

    smoothed_point: tuple[float, float] | None = None
    last_sent_point: tuple[float, float] | None = None

    paused = False
    swap_handedness = config.swap_handedness_labels
    status = "Calibration started. Show your face and index finger, then press SPACE per point."

    hand_calibrator.start()
    eye_calibrator.start()
    calibration_window_open = False

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                status = "Camera frame read failed."
                break

            hand_tracking = hand_tracker.process(raw_frame)
            eye_tracking = eye_tracker.process(raw_frame)

            display_frame = cv2.flip(raw_frame, 1) if config.preview_mirror else raw_frame
            display_hand_x_norm: float | None = None

            if hand_tracking is not None:
                display_hand_x_norm = 1.0 - hand_tracking.x_norm if config.preview_mirror else hand_tracking.x_norm
                hand_calibrator.add_sample(hand_tracking.x_norm, hand_tracking.y_norm)

            if eye_tracking is not None:
                eye_calibrator.add_sample(eye_tracking.x_norm, eye_tracking.y_norm)

            mapped_hand: tuple[float, float] | None = None
            mapped_eye: tuple[float, float] | None = None
            fused_point: tuple[float, float] | None = None

            is_calibrating = hand_calibrator.active and eye_calibrator.active
            is_calibrated = hand_model is not None and eye_model is not None

            if not is_calibrating and is_calibrated and not paused:
                if hand_tracking is not None:
                    mapped_hand = hand_model.map(hand_tracking.x_norm, hand_tracking.y_norm)
                    mapped_hand = mouse.clamp(mapped_hand[0], mapped_hand[1])

                if eye_tracking is not None:
                    mapped_eye = eye_model.map(eye_tracking.x_norm, eye_tracking.y_norm)
                    mapped_eye = mouse.clamp(mapped_eye[0], mapped_eye[1])

                fused_point = _fuse_points(mapped_hand, mapped_eye, config.eye_fusion_weight)
                if fused_point is not None:
                    fused_point = mouse.clamp(fused_point[0], fused_point[1])
                    smoothed_point = _blend_point(smoothed_point, fused_point, config.smooth_alpha)
                    if smoothed_point is not None:
                        if last_sent_point is None or _distance(smoothed_point, last_sent_point) >= config.move_deadzone_px:
                            mouse.move(smoothed_point[0], smoothed_point[1])
                            last_sent_point = smoothed_point

                if mapped_hand is not None and mapped_eye is not None:
                    status = "Tracking with hand + eye fusion"
                elif mapped_hand is not None:
                    status = "Tracking with hand only"
                elif mapped_eye is not None:
                    status = "Tracking with eye only"
                else:
                    status = "No hand/face detected"
            elif is_calibrating:
                status = (
                    "Calibrating - align both face and fingertip, "
                    f"samples hand {hand_calibrator.sample_count()}/{config.calibration_samples_required}, "
                    f"eye {eye_calibrator.sample_count()}/{config.calibration_samples_required}"
                )
            elif not is_calibrated:
                status = "Press c to run calibration"

            debug = _draw_debug(
                frame=display_frame,
                hand_tracking=hand_tracking,
                eye_tracking=eye_tracking,
                display_hand_x_norm=display_hand_x_norm,
                status=status,
                paused=paused,
                mapped_point=smoothed_point,
                screen_size=(mouse.screen.width, mouse.screen.height),
                is_calibrated=is_calibrated,
                is_calibrating=is_calibrating,
                swap_handedness=swap_handedness,
                eye_fusion_weight=config.eye_fusion_weight,
                hand_samples=hand_calibrator.sample_count(),
                eye_samples=eye_calibrator.sample_count(),
                samples_required=config.calibration_samples_required,
            )
            cv2.imshow(DEBUG_WINDOW, debug)

            if is_calibrating:
                if not calibration_window_open:
                    cv2.namedWindow(CALIB_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(CALIB_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    calibration_window_open = True

                calibration_message = (
                    "Look at and point to the red target. "
                    f"Samples hand {hand_calibrator.sample_count()}/{config.calibration_samples_required}, "
                    f"eye {eye_calibrator.sample_count()}/{config.calibration_samples_required}"
                )
                cv2.imshow(CALIB_WINDOW, hand_calibrator.render(calibration_message))
            else:
                if calibration_window_open:
                    cv2.destroyWindow(CALIB_WINDOW)
                    calibration_window_open = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                status = "Paused" if paused else "Resumed"
            if key == ord("h"):
                swap_handedness = not swap_handedness
                status = f"Handedness swap {'enabled' if swap_handedness else 'disabled'}"
            if key == ord("c"):
                hand_calibrator.start()
                eye_calibrator.start()
                hand_model = None
                eye_model = None
                smoothed_point = None
                last_sent_point = None
                status = "Calibration started"
            if key == 32 and hand_calibrator.active and eye_calibrator.active:
                if not hand_calibrator.can_capture() or not eye_calibrator.can_capture():
                    status = (
                        "Need more samples before capture - "
                        f"hand {hand_calibrator.samples_needed()} remaining, "
                        f"eye {eye_calibrator.samples_needed()} remaining"
                    )
                    continue

                hand_captured, hand_message = hand_calibrator.capture_current_point()
                eye_captured, eye_message = eye_calibrator.capture_current_point()

                if not hand_captured or not eye_captured:
                    status = f"Capture failed. hand: {hand_message}, eye: {eye_message}"
                    continue

                if hand_message == "done" and eye_message == "done":
                    maybe_hand_model = hand_calibrator.fit()
                    maybe_eye_model = eye_calibrator.fit()
                    hand_calibrator.stop()
                    eye_calibrator.stop()

                    if maybe_hand_model is None or maybe_eye_model is None:
                        status = "Calibration failed. Try again."
                    else:
                        hand_model = maybe_hand_model
                        eye_model = maybe_eye_model
                        status = "Calibration complete. Cursor fusion live."
                else:
                    status = f"Captured point {hand_calibrator.index}/{len(hand_calibrator.targets)}"

    finally:
        cap.release()
        hand_tracker.close()
        eye_tracker.close()
        cv2.destroyAllWindows()

    return 0


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = AppConfig(
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        model_path=args.model_path,
        eye_model_path=args.eye_model_path,
        eye_fusion_weight=float(np.clip(args.eye_fusion_weight, 0.0, 1.0)),
    )
    raise SystemExit(run(config))


if __name__ == "__main__":
    main()
