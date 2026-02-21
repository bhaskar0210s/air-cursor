# Hand Cursor (macOS Apple Silicon)

Control your mouse pointer using index-fingertip hand tracking.

This implementation includes:
- Apple Vision Framework hand-pose tracking (`VNDetectHumanHandPoseRequest`)
- Direct fingertip-to-cursor mapping (no calibration)
- Adjustable cursor sensitivity
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

## Sensitivity

- CLI: `--sensitivity 1.2`
- Runtime keys:
  - `+` / `]` increase
  - `-` / `[` decrease

Sensitivity changes how strongly fingertip movement maps to screen movement.

## Controls

- `+`, `-`, `[`, `]`: adjust sensitivity
- `h`: toggle handedness label swap
- `p`: pause/resume cursor control
- `q`: quit

## macOS Permissions

1. Camera: allow your terminal/IDE app to use camera.
2. Accessibility: System Settings -> Privacy & Security -> Accessibility -> enable your terminal/IDE.

Without Accessibility permission, cursor events are blocked by macOS.
