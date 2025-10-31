import os
import cv2
import numpy as np

class BaseDetector:
    def detect(self, frame):
        """Return list of (x1,y1,x2,y2,cls,conf)."""
        raise NotImplementedError

class MockHOGDetector(BaseDetector):
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    def detect(self, frame):
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
        results = []
        for (x,y,w,h), conf in zip(rects, weights):
            results.append((x, y, x+w, y+h, 'person', float(conf)))
        return results

class YOLOv8Detector(BaseDetector):
    def __init__(self, model_name='yolov8n.pt'):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics not installed. Use DETECTOR=mock or `pip install ultralytics`. Original: %s" % e)
        self.model = YOLO(model_name)
        self.names = self.model.model.names if hasattr(self.model, 'model') else None
    def detect(self, frame):
        results = self.model.predict(source=frame, verbose=False)[0]
        out = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = self.names.get(cls_id, str(cls_id)) if self.names else str(cls_id)
            out.append((x1,y1,x2,y2,name,conf))
        return out

def build_detector(kind='yolo'):
    kind = (kind or 'yolo').lower()
    if kind == 'mock':
        return MockHOGDetector()
    return YOLOv8Detector()
