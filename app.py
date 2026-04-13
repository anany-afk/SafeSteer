import os
import cv2
import time
import threading
import logging
from flask import Flask, render_template, Response, jsonify
from src.core.detector import RealTimeDetector
from src.config import CFG

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)

# Global detector state
detector = None
detector_thread = None

def run_detector():
    global detector
    try:
        model_path = os.path.join(CFG["output_dir"], "drowsiness_model_full.pth")
        scaler_path = os.path.join(CFG["output_dir"], "scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            log.error(f"Model or Scaler not found in {CFG['output_dir']}. Please train the model first.")
            return

        detector = RealTimeDetector(model_path, scaler_path, CFG)
        detector.web_mode = True
        detector.stop_flag = False
        log.info("Starting Detector Session (Web Mode)...")
        detector.run()
    except Exception as e:
        log.error(f"Detector thread error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global detector, detector_thread
    if detector_thread is None or not detector_thread.is_alive():
        detector_thread = threading.Thread(target=run_detector)
        detector_thread.daemon = True
        detector_thread.start()
        # Wait for detector to initialize landmarks
        time.sleep(2)
        log.info("Monitoring started.")
    return jsonify({"status": "success"})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global detector, detector_thread
    if detector is not None:
        detector.stop_flag = True
        log.info("Stop signal sent to detector.")
    if detector_thread is not None:
        detector_thread.join(timeout=3)
        detector_thread = None
        detector = None
    return jsonify({"status": "success"})

def generate_frames():
    global detector
    while True:
        if detector is None or detector.latest_frame is None:
            time.sleep(0.1)
            continue
        
        try:
            # Re-encode for MJPEG stream
            ret, buffer = cv2.imencode('.jpg', detector.latest_frame)
            if not ret:
                time.sleep(0.05)
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Match browser refresh rate
            time.sleep(1/30) 
        except Exception as e:
            log.error(f"Stream encoding error: {e}")
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    global detector
    if detector is not None:
        # Return the latest captured stats (prob, ear, yawns, etc.)
        return jsonify(detector.latest_stats)
    return jsonify({"status": "starting"})

if __name__ == '__main__':
    # Disable debug mode to prevent double-threaded detector issues
    app.run(host='127.0.0.1', port=5000, debug=False)
