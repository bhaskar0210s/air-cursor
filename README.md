# Hand Cursor (macOS Apple Silicon)

Control your mouse pointer with one finger using webcam hand tracking.

This implementation includes:
- Real-time hand tracking with MediaPipe Hand Landmarker
- Index fingertip cursor control
- Native macOS cursor movement through Quartz APIs
- Exponential smoothing and deadzone filtering for stability

## Requirements

- macOS (Apple Silicon / M1, M2, M3 supported)
- Python 3.12
- Webcam access
- Accessibility permission (for mouse control)

## Setup

```bash
cd /Users/bhaskars/air-cursor
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Run

```bash
eye-cursor
```

or

```bash
hand-cursor
```

On first run, the app auto-downloads `hand_landmarker.task` to:
- `~/.cache/eye-cursor/hand_landmarker.task`

If your network is restricted, pass a local model file:

```bash
eye-cursor --model-path /path/to/hand_landmarker.task
```

## Controls

- `p`: pause/resume cursor control
- `q`: quit

## macOS Permissions

1. Camera: allow your terminal/IDE app to use camera.
2. Accessibility: System Settings -> Privacy & Security -> Accessibility -> enable your terminal/IDE.

Without Accessibility permission, cursor events are blocked by macOS.
