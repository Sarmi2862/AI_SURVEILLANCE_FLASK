import os, time, cv2, yaml
import numpy as np

from detector.yolo_detector import build_detector
from detector.actions import HeuristicActionRecognizer
from alerting.twilio_client import TwilioClient
from alerting.notifier import AlertNotifier

class VideoProcessor:
    def __init__(self, source, cfg, detector_kind='yolo'):
        self.cap = cv2.VideoCapture(int(source)) if str(source).isdigit() else cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        self.cfg = cfg
        self.detector = build_detector(detector_kind)
        self.actions = HeuristicActionRecognizer(cfg)
        self.notifier = AlertNotifier(TwilioClient(), cooldown_sec=cfg['alerts']['cooldown_sec'])
        self.draw = bool(cfg['video'].get('draw', True))
        self.max_width = int(cfg['video'].get('max_width', 1024))

    def _draw_box(self, frame, box, label, color=(0,255,0)):
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def gen_frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            # resize for speed/stream
            h, w = frame.shape[:2]
            if self.max_width and w > self.max_width:
                scale = self.max_width / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            # Detect
            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                # Fail-safe: no detections
                detections = []
                cv2.putText(frame, f"Detector error: {e}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Person boxes
            person_boxes = [(x1,y1,x2,y2) for (x1,y1,x2,y2,cls,conf) in detections if str(cls) == 'person' or str(cls) == '0']

            # Intruder alert: any person
            if self.cfg['intruder']['enabled']:
                for box in person_boxes:
                    snap = self.notifier.notify('INTRUDER', frame, box)
                    # throttle handles spam; we still draw label
                    if self.draw:
                        self._draw_box(frame, box, 'INTRUDER', (0,255,0))

            # Heuristic actions
            labels = self.actions.infer(frame, person_boxes)
            for label, box in labels:
                snap = self.notifier.notify(label, frame, box)
                if self.draw:
                    self._draw_box(frame, box, label, (0,0,255))

            # Draw other detections (non-person) if using YOLO
            for (x1,y1,x2,y2,cls,conf) in detections:
                if str(cls) in ['person','0']:
                    continue
                if self.draw:
                    self._draw_box(frame, (x1,y1,x2,y2), f"{cls}:{conf:.2f}", (255,255,0))

            # Encode JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    def latest_events(self, limit=12):
        try:
            files = sorted([f for f in os.listdir('events') if f.lower().endswith('.jpg')],
                           key=lambda x: os.path.getmtime(os.path.join('events', x)), reverse=True)[:limit]
            return [{'filename': f, 'label': f.split('_')[0], 'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(os.path.join('events', f))))} for f in files]
        except Exception:
            return []
