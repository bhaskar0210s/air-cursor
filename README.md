# Cursor Fusion (macOS Apple Silicon)

Control your mouse pointer using fused hand + eye tracking.

This implementation includes:
- Real-time hand tracking with MediaPipe Hand Landmarker
- Real-time eye tracking with MediaPipe Face Landmarker (iris features)
- Joint calibration for hand and eye models
- Weighted hand+eye fusion for smoother startup and better precision
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

On first run, the app auto-downloads models to:
- `~/.cache/eye-cursor/hand_landmarker.task`
- `~/.cache/eye-cursor/face_landmarker.task`

If your network is restricted, pass local model files:

```bash
eye-cursor --model-path /path/to/hand_landmarker.task --eye-model-path /path/to/face_landmarker.task
```

## Controls

- `c`: start/restart calibration
- `space`: capture current calibration point
- `h`: toggle handedness label swap
- `p`: pause/resume cursor control
- `q`: quit

Calibration starts automatically at launch. Complete all target points for cursor control.

## Optional Tuning

- `--eye-fusion-weight 0.25` controls eye influence when both trackers are available.
  - `0.0` = hand only
  - `1.0` = eye only
  - default `0.25` = hand-led with eye assist

## Calibration Flow

1. Keep your face visible and point your index fingertip at each red target dot.
2. Press `space` to capture that point.
3. Repeat until all calibration points are captured.
4. Cursor control becomes live after calibration completes.

## macOS Permissions

1. Camera: allow your terminal/IDE app to use camera.
2. Accessibility: System Settings -> Privacy & Security -> Accessibility -> enable your terminal/IDE.

Without Accessibility permission, cursor events are blocked by macOS.
