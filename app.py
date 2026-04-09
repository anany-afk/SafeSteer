from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import os

from model import RealTimeDetector, CFG

app = Flask(__name__)

detector = None
detector_thread = None

def run_detector():
    global detector
    model_path = os.path.join(CFG["output_dir"], "drowsiness_model_full.pth")
    scaler_path = os.path.join(CFG["output_dir"], "scaler.pkl")
    
    if not os.path.exists(model_path):
        print("Model not found. Ensure training_output exists.")
        return

    detector = RealTimeDetector(model_path, scaler_path, CFG)
    detector.web_mode = True
    # Reset flags if restarting
    detector.stop_flag = False
    detector.run()

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
        # Wait a moment for detector to initialize
        time.sleep(2)
    return jsonify({"status": "success"})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global detector, detector_thread
    if detector is not None:
        detector.stop_flag = True
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
            ret, buffer = cv2.imencode('.jpg', detector.latest_frame)
            if not ret:
                time.sleep(0.1)
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # cap frame rate to ~30 FPS
        except Exception as e:
            print("Encoding error:", e)
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    global detector
    if detector is not None and len(detector.latest_stats) > 0:
        return jsonify(detector.latest_stats)
    return jsonify({"status": "starting"})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
