from __future__ import annotations

import argparse

import cv2
import numpy as np

from .config import AppConfig
from .macos_mouse import MacOSMouseController
from .tracking import HandTracker, TrackingResult

DEBUG_WINDOW = "Hand Cursor Debug"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control macOS cursor with fingertip tracking")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for OpenCV")
    parser.add_argument("--frame-width", type=int, default=640, help="Capture width")
    parser.add_argument("--frame-height", type=int, default=480, help="Capture height")
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.2,
        help="Cursor sensitivity around screen center (0.4 to 3.0)",
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


def _clamp_sensitivity(value: float, min_value: float, max_value: float) -> float:
    return float(np.clip(value, min_value, max_value))


def _apply_sensitivity(norm_value: float, sensitivity: float) -> float:
    centered = (norm_value - 0.5) * sensitivity + 0.5
    return float(np.clip(centered, 0.0, 1.0))


def _draw_debug(
    frame: np.ndarray,
    tracking: TrackingResult | None,
    display_x_norm: float | None,
    status: str,
    paused: bool,
    mapped_point: tuple[float, float] | None,
    screen_size: tuple[int, int],
    swap_handedness: bool,
    sensitivity: float,
) -> np.ndarray:
    out = frame.copy()
    frame_h, frame_w = out.shape[:2]

    hand_text = "Hand: not detected"
    if tracking is not None and display_x_norm is not None:
        px = int(np.clip(display_x_norm * frame_w, 0, frame_w - 1))
        py = int(np.clip(tracking.y_norm * frame_h, 0, frame_h - 1))
        cv2.circle(out, (px, py), 8, (0, 255, 0), 2)

        hand_label = _normalize_handedness(tracking.handedness, swap_handedness)
        hand_text = f"Hand: ({tracking.x_norm:.3f}, {tracking.y_norm:.3f}) [{hand_label}]"

    if mapped_point is not None:
        sx, sy = mapped_point
        sw, sh = screen_size
        preview_x = int(np.clip((sx / max(sw, 1)) * frame_w, 0, frame_w - 1))
        preview_y = int(np.clip((sy / max(sh, 1)) * frame_h, 0, frame_h - 1))
        cv2.circle(out, (preview_x, preview_y), 10, (0, 255, 255), 2)

    mode = "PAUSED" if paused else "LIVE"
    handedness_state = "ON" if swap_handedness else "OFF"

    cv2.putText(out, hand_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, f"Mode: {mode} | Sensitivity: {sensitivity:.2f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, f"Handedness swap: {handedness_state}", (12, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
    cv2.putText(out, status, (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)
    cv2.putText(out, "Keys: +/- sensitivity | h hand swap | p pause | q quit", (12, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    return out


def run(config: AppConfig) -> int:
    mouse = MacOSMouseController()
    try:
        tracker = HandTracker(
            fingertip_landmark_index=config.fingertip_landmark_index,
        )
    except Exception as exc:
        print(f"Failed to initialize hand tracker: {exc}")
        return 1

    cap = _open_camera(config.camera_index, config.frame_width, config.frame_height)
    if not cap.isOpened():
        print("Failed to open webcam. Check camera index/permissions.")
        tracker.close()
        return 1

    cv2.namedWindow(DEBUG_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(DEBUG_WINDOW, config.frame_width, config.frame_height)

    paused = False
    swap_handedness = config.swap_handedness_labels
    sensitivity = _clamp_sensitivity(config.cursor_sensitivity, config.min_sensitivity, config.max_sensitivity)
    status = "Show one hand and move your index fingertip to control cursor."

    smoothed_point: tuple[float, float] | None = None
    last_sent_point: tuple[float, float] | None = None

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
                display_x_norm = 1.0 - tracking.x_norm if config.preview_mirror else tracking.x_norm

                cursor_x_norm = display_x_norm
                cursor_y_norm = tracking.y_norm

                cursor_x_norm = _apply_sensitivity(cursor_x_norm, sensitivity)
                cursor_y_norm = _apply_sensitivity(cursor_y_norm, sensitivity)

                mapped_x = cursor_x_norm * (mouse.screen.width - 1)
                mapped_y = cursor_y_norm * (mouse.screen.height - 1)
                mapped_point = mouse.clamp(mapped_x, mapped_y)

                smoothed_point = _blend_point(smoothed_point, mapped_point, config.smooth_alpha)
                if not paused and smoothed_point is not None:
                    if last_sent_point is None or _distance(smoothed_point, last_sent_point) >= config.move_deadzone_px:
                        mouse.move(smoothed_point[0], smoothed_point[1])
                        last_sent_point = smoothed_point

                hand_label = _normalize_handedness(tracking.handedness, swap_handedness)
                status = f"Tracking {hand_label} hand"
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
                swap_handedness=swap_handedness,
                sensitivity=sensitivity,
            )
            cv2.imshow(DEBUG_WINDOW, debug)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("p"):
                paused = not paused
                status = "Paused" if paused else "Resumed"
            if key == ord("h"):
                swap_handedness = not swap_handedness
                status = f"Handedness swap {'enabled' if swap_handedness else 'disabled'}"
            if key in (ord("-"), ord("_"), ord("[")):
                sensitivity = _clamp_sensitivity(sensitivity - config.sensitivity_step, config.min_sensitivity, config.max_sensitivity)
                status = f"Sensitivity: {sensitivity:.2f}"
            if key in (ord("+"), ord("="), ord("]")):
                sensitivity = _clamp_sensitivity(sensitivity + config.sensitivity_step, config.min_sensitivity, config.max_sensitivity)
                status = f"Sensitivity: {sensitivity:.2f}"

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
        cursor_sensitivity=args.sensitivity,
    )
    raise SystemExit(run(config))


if __name__ == "__main__":
    main()
