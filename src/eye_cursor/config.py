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

    calibration_grid_size: int = 3
    calibration_samples_required: int = 20

    model_path: str | None = None
