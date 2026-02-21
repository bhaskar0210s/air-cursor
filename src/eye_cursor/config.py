from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    smooth_alpha: float = 0.25
    move_deadzone_px: float = 1.5

    hand_input_padding: float = 0.08
    fingertip_landmark_index: int = 8

    model_path: str | None = None
