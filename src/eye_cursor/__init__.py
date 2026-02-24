"""Eye tracking package."""

from .app import main, run
from .config import EyeCursorConfig
from .tracking import EyeTracker, TrackingResult

__all__ = ["EyeCursorConfig", "EyeTracker", "TrackingResult", "run", "main"]
