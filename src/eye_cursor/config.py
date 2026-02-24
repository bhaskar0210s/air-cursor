from dataclasses import dataclass


@dataclass(frozen=True)
class EyeCursorConfig:
    camera_index: int = 0
    frame_width: int = 960
    frame_height: int = 540
    preview_mirror: bool = True

    smooth_alpha: float = 0.18
    move_deadzone_px: float = 0.5

    cursor_sensitivity: float = 1.0
    min_sensitivity: float = 0.4
    max_sensitivity: float = 3.0
    sensitivity_step: float = 0.1

    min_confidence: float = 0.35
    gaze_deadband_norm: float = 0.0006
    curve_gamma: float = 1.3
    uncalibrated_gain: float = 2.6
