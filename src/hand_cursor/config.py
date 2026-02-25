from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    smooth_alpha: float = 0.25
    move_deadzone_px: float = 1.5

    fingertip_landmark_index: int = 8
    preview_mirror: bool = True
    swap_handedness_labels: bool = True
    cursor_sensitivity: float = 1.2
    min_sensitivity: float = 0.4
    max_sensitivity: float = 3.0
    sensitivity_step: float = 0.1
