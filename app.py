import os, yaml
from flask import Flask, render_template, Response, send_from_directory
from dotenv import load_dotenv
from camera import VideoProcessor

load_dotenv()

with open('config.yaml', 'r') as f:
    CFG = yaml.safe_load(f)

SOURCE = os.getenv('SOURCE', '0')
DETECTOR = os.getenv('DETECTOR', 'yolo')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '5000'))
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

app = Flask(__name__)
processor = VideoProcessor(SOURCE, CFG, detector_kind=DETECTOR)

@app.route('/')
def index():
    events = processor.latest_events()
    return render_template('index.html',
                           detector_name=DETECTOR.upper(),
                           source_label=str(SOURCE),
                           cooldown=CFG['alerts']['cooldown_sec'],
                           events=events,
                           year=os.environ.get('YEAR_OVERRIDE') or '2025')

@app.route('/video_feed')
def video_feed():
    return Response(processor.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/events/<path:filename>')
def event_image(filename):
    return send_from_directory('events', filename, as_attachment=False)

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, debug=DEBUG, threaded=True)
