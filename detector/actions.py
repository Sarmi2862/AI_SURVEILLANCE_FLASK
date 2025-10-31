import numpy as np
import cv2

class HeuristicActionRecognizer:
    """Heuristic action recognition:
    - fall: width/height >= aspect_ratio for min_frames
    - violence: two people overlap (IoU >= proximity_iou) and local motion >= motion_mag
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.fall_counter = {}  # id -> count
        self.prev_gray = None

    @staticmethod
    def iou(a, b):
        (x1,y1,x2,y2) = a
        (x3,y3,x4,y4) = b
        xi1, yi1 = max(x1,x3), max(y1,y3)
        xi2, yi2 = min(x2,x4), min(y2,y4)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = (x2-x1)*(y2-y1)
        area_b = (x4-x3)*(y4-y3)
        union = area_a + area_b - inter + 1e-6
        return inter/union

    def update_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray, dtype=np.float32)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev_gray = gray
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        return mag

    def infer(self, frame, person_boxes):
        labels = []  # list of (label, box)
        mag = self.update_motion(frame)
        hcfg = self.cfg

        # FALL
        if hcfg['fall']['enabled']:
            for i, (x1,y1,x2,y2) in enumerate(person_boxes):
                w = x2 - x1
                h = y2 - y1
                pid = i  # simple stable id for demo; replace with tracker for robustness
                if h > 0 and (w / (h+1e-6)) >= hcfg['fall']['aspect_ratio']:
                    self.fall_counter[pid] = self.fall_counter.get(pid, 0) + 1
                else:
                    self.fall_counter[pid] = 0
                if self.fall_counter.get(pid, 0) >= hcfg['fall']['min_frames']:
                    labels.append(('ACCIDENT/FALL', (x1,y1,x2,y2)))

        # VIOLENCE
        if hcfg['violence']['enabled'] and len(person_boxes) >= 2:
            for i in range(len(person_boxes)):
                for j in range(i+1, len(person_boxes)):
                    a = person_boxes[i]; b = person_boxes[j]
                    iou = self.iou(a, b)
                    if iou >= hcfg['violence']['proximity_iou']:
                        xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
                        xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
                        roi = mag[yi1:yi2, xi1:xi2]
                        if roi.size > 0 and float(np.mean(roi)) >= hcfg['violence']['motion_mag']:
                            labels.append(('VIOLENCE', (min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3]))))
        return labels
