# Real-Time Video Surveillance with Alerts

Detect intruders, violence, or accidents in live CCTV feeds and send SMS alerts automatically.

## Features
- Live video streaming via Flask (MJPEG).
- Object detection with **YOLOv8** (Ultralytics). If YOLOv8 isn't installed, a **Mock/HOG person detector** is used to keep the app running.
- Heuristic **action recognition**:
  - **Intruder**: any detected person.
  - **Accident/Fall**: person aspect ratio suggests fallen posture for N frames.
  - **Violence**: two people in close proximity + high motion in ROI.
- Alert throttling to avoid spam (configurable).
- **Twilio SMS** notifications (optional; logs to console if not configured).
- Save event snapshots to `/events` folder with labels and timestamps.
- Configurable via `.env` and `config.yaml`.

## Quick Start

> Python 3.9+ recommended. GPU not required but supported if PyTorch+CUDA are installed.

1) **Create and edit `.env`** (or copy from `config.example.env`):
```
cp config.example.env .env
# edit .env to set SOURCE (0 for webcam, or RTSP/HTTP/FILE path) and Twilio creds
```

2) **(Option A) Full AI stack with YOLOv8 (recommended):**
```
pip install -r requirements.txt
# First run will download YOLOv8n automatically via Ultralytics
python app.py
```

3) **(Option B) No heavy installs / quick demo (Mock mode):**
```
pip install -r requirements-lite.txt
# Force mock detector
export DETECTOR=mock   # (Windows PowerShell: $env:DETECTOR='mock')
python app.py
```

4) Open the app:
```
http://127.0.0.1:5000
```

## File Structure
```
surveillance/
├─ app.py
├─ camera.py
├─ config.yaml
├─ config.example.env
├─ requirements.txt
├─ requirements-lite.txt
├─ README.md
├─ detector/
│  ├─ __init__.py
│  ├─ yolo_detector.py
│  └─ actions.py
├─ alerting/
│  ├─ __init__.py
│  ├─ twilio_client.py
│  └─ notifier.py
├─ templates/
│  └─ index.html
├─ static/
│  └─ styles.css
├─ events/           # snapshots saved here
└─ samples/
   └─ sample.mp4     # placeholder (bring your own video or webcam)
```

## Config
- **.env**
  - `SOURCE=0` (webcam index) **or** RTSP/HTTP/file path
  - `DETECTOR=yolo` or `mock`
  - Twilio: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM`, `ALERT_TO`
- **config.yaml**
  - Motion, fall/violence thresholds and alert cooldowns

## Notes
- YOLOv8 models are large. Use `DETECTOR=mock` if you need a lightweight demo.
- For multiple cameras, run multiple app instances with different `SOURCE` and ports.
- Violence/accident classification is heuristic unless you plug in a trained action-recognition model.
  You can extend `detector/actions.py` to integrate your own model.
