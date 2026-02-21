from __future__ import annotations

import argparse

import cv2
import numpy as np

from .calibration import CalibrationModel, Calibrator
from .config import AppConfig
from .macos_mouse import MacOSMouseController
from .tracking import HandTracker, TrackingResult

DEBUG_WINDOW = "Hand Cursor Debug"
CALIB_WINDOW = "Hand Cursor Calibration"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control macOS cursor with one finger")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for OpenCV")
    parser.add_argument("--frame-width", type=int, default=640, help="Capture width")
    parser.add_argument("--frame-height", type=int, default=480, help="Capture height")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional local path to MediaPipe hand_landmarker.task",
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


def _draw_debug(
    frame: np.ndarray,
    tracking: TrackingResult | None,
    display_x_norm: float | None,
    status: str,
    paused: bool,
    mapped_point: tuple[float, float] | None,
    screen_size: tuple[int, int],
    is_calibrated: bool,
    is_calibrating: bool,
    swap_handedness: bool,
) -> np.ndarray:
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    if tracking is not None and display_x_norm is not None:
        px = int(np.clip(display_x_norm * frame_w, 0, frame_w - 1))
        py = int(np.clip(tracking.y_norm * frame_h, 0, frame_h - 1))
        cv2.circle(out, (px, py), 8, (0, 255, 0), 2)

        hand = _normalize_handedness(tracking.handedness, swap_handedness)
        cv2.putText(
            out,
            f"Finger: ({tracking.x_norm:.3f}, {tracking.y_norm:.3f}) | Hand: {hand}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            2,
        )

    if mapped_point is not None:
        sx, sy = mapped_point
        sw, sh = screen_size
        preview_x = int(np.clip((sx / max(sw, 1)) * frame_w, 0, frame_w - 1))
        preview_y = int(np.clip((sy / max(sh, 1)) * frame_h, 0, frame_h - 1))
        cv2.circle(out, (preview_x, preview_y), 10, (0, 255, 255), 2)

    mode = "PAUSED" if paused else "LIVE"
    calibration_state = "READY" if is_calibrated else ("CALIBRATING" if is_calibrating else "NOT CALIBRATED")
    handedness_state = "ON" if swap_handedness else "OFF"

    cv2.putText(out, f"Mode: {mode} | Calibration: {calibration_state}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, f"Handedness swap: {handedness_state}", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, status, (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
    cv2.putText(out, "Keys: c calibrate | space capture | h hand swap | p pause | q quit", (12, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    return out


def run(config: AppConfig) -> int:
    mouse = MacOSMouseController()
    try:
        tracker = HandTracker(
            model_path=config.model_path,
            fingertip_landmark_index=config.fingertip_landmark_index,
        )
    except Exception as exc:
        print(f"Failed to initialize hand tracker: {exc}")
        return 1

    calibrator = Calibrator(
        screen_width=mouse.screen.width,
        screen_height=mouse.screen.height,
        grid_size=config.calibration_grid_size,
        samples_required=config.calibration_samples_required,
    )

    cap = _open_camera(config.camera_index, config.frame_width, config.frame_height)
    if not cap.isOpened():
        print("Failed to open webcam. Check camera index/permissions.")
        tracker.close()
        return 1

    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW, config.frame_width, config.frame_height)

    calibration_model: CalibrationModel | None = None
    smoothed_point: tuple[float, float] | None = None
    last_sent_point: tuple[float, float] | None = None

    paused = False
    swap_handedness = config.swap_handedness_labels
    status = "Press c to calibrate. Then use your index fingertip to move cursor."

    calibrator.start()
    status = "Calibration started"
    calibration_window_open = False

    try:
        while True:
            ok, raw_frame = cap.read()
            if not ok:
                status = "Camera frame read failed."
                break

            tracking = tracker.process(raw_frame)

            display_frame = cv2.flip(raw_frame, 1) if config.preview_mirror else raw_frame
            display_x_norm: float | None = None
            mapped_point: tuple[float, float] | None = None

            if tracking is not None:
                feature_x = tracking.x_norm
                feature_y = tracking.y_norm

                if config.preview_mirror:
                    display_x_norm = 1.0 - feature_x
                else:
                    display_x_norm = feature_x

                calibrator.add_sample(feature_x, feature_y)

                if calibration_model is not None and not calibrator.active and not paused:
                    mapped_point = calibration_model.map(feature_x, feature_y)
                    mapped_point = mouse.clamp(mapped_point[0], mapped_point[1])

                    smoothed_point = _blend_point(smoothed_point, mapped_point, config.smooth_alpha)
                    if smoothed_point is not None:
                        if last_sent_point is None or _distance(smoothed_point, last_sent_point) >= config.move_deadzone_px:
                            mouse.move(smoothed_point[0], smoothed_point[1])
                            last_sent_point = smoothed_point
                elif not calibrator.active and calibration_model is None:
                    status = "Press c to run calibration"
            else:
                if calibrator.active:
                    status = "No hand detected during calibration"
                else:
                    status = "No hand detected"

            debug = _draw_debug(
                frame=display_frame,
                tracking=tracking,
                display_x_norm=display_x_norm,
                status=status,
                paused=paused,
                mapped_point=smoothed_point,
                screen_size=(mouse.screen.width, mouse.screen.height),
                is_calibrated=calibration_model is not None,
                is_calibrating=calibrator.active,
                swap_handedness=swap_handedness,
            )
            cv2.imshow(DEBUG_WINDOW, debug)

            if calibrator.active:
                if not calibration_window_open:
                    cv2.namedWindow(CALIB_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(CALIB_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    calibration_window_open = True

                calibration_message = "Point fingertip at the red target and press SPACE"
                cv2.imshow(CALIB_WINDOW, calibrator.render(calibration_message))
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
                calibrator.start()
                calibration_model = None
                smoothed_point = None
                last_sent_point = None
                status = "Calibration started"
            if key == 32 and calibrator.active:
                captured, message = calibrator.capture_current_point()
                status = message
                if captured and message == "done":
                    model = calibrator.fit()
                    calibrator.stop()
                    if model is None:
                        status = "Calibration failed. Try again."
                    else:
                        calibration_model = model
                        status = "Calibration complete. Cursor control live."

    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    return 0


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = AppConfig(
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        model_path=args.model_path,
    )
    raise SystemExit(run(config))


if __name__ == "__main__":
    main()
