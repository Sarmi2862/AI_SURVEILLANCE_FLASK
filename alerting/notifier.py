import os, time, cv2

class AlertNotifier:
    def __init__(self, twilio_client, cooldown_sec=60, snapshot_dir='events'):
        self.twilio = twilio_client
        self.cooldown = cooldown_sec
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.last_time = {}

    def _throttle(self, key):
        now = time.time()
        t0 = self.last_time.get(key, 0)
        if now - t0 >= self.cooldown:
            self.last_time[key] = now
            return True
        return False

    def notify(self, label, frame, box=None):
        key = label
        if not self._throttle(key):
            return False
        # Snapshot
        filename = None
        if frame is not None:
            img = frame.copy()
            if box is not None:
                x1,y1,x2,y2 = map(int, box)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(img, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            ts = time.strftime('%Y%m%d-%H%M%S')
            filename = f"{label.replace('/', '_')}_{ts}.jpg"
            cv2.imwrite(os.path.join(self.snapshot_dir, filename), img)
        # SMS
        msg = f"ALERT: {label} detected."
        self.twilio.send_sms(msg)
        return filename
