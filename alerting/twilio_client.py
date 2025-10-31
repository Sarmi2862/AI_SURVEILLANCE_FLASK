import os

class TwilioClient:
    def __init__(self):
        self.sid = os.getenv('TWILIO_ACCOUNT_SID', '').strip()
        self.token = os.getenv('TWILIO_AUTH_TOKEN', '').strip()
        self.src = os.getenv('TWILIO_FROM', '').strip()
        self.dst = os.getenv('ALERT_TO', '').strip()
        self.enabled = all([self.sid, self.token, self.src, self.dst])
        self.client = None
        if self.enabled:
            try:
                from twilio.rest import Client
                self.client = Client(self.sid, self.token)
            except Exception as e:
                print(f"[Twilio] Failed to init: {e}. Falling back to console logs.")
                self.enabled = False

    def send_sms(self, body):
        if self.enabled and self.client:
            try:
                self.client.messages.create(body=body, from_=self.src, to=self.dst)
                print(f"[Twilio] SMS sent: {body}")
                return True
            except Exception as e:
                print(f"[Twilio] Error sending SMS: {e}")
                return False
        print(f"[Twilio] (console) {body}")
        return True
