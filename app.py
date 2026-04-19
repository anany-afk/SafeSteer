import os
import cv2
import time
import logging
import numpy as np
from flask import Flask, render_template, Response, jsonify

from src.core.detector import RealTimeDetector
from src.core.camera import ThreadedCamera
from src.core.pipeline import PipelineEngine
from src.data.extractor import FacialFeatureExtractor
from src.models.inference import InferenceEngine
from src.utils.performance import PerformanceBenchmarker
from src.config import CFG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
pipeline = None

def start_pipeline():
    global pipeline
    try:
        base_dir = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, CFG["output_dir"], "drowsiness_model_full.pth")
        scaler_path = os.path.join(base_dir, CFG["output_dir"], "scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            log.error("Model or Scaler not found. Please train first.")
            return

        # 1. Initialize Components
        inference_engine = InferenceEngine(model_path, CFG)
        extractor = FacialFeatureExtractor()
        detector = RealTimeDetector(model_path, scaler_path, CFG)
        
        # 2. Benchmark & Adjust Mode
        if CFG.get("performance_mode") == "AUTO":
            benchmarker = PerformanceBenchmarker(CFG)
            mode, overrides = benchmarker.run_benchmark(model=inference_engine.model, extractor=extractor)
            CFG.update(overrides)
            CFG["performance_mode"] = mode
            log.info(f"System benchmarked. Using {mode} overrides: {overrides}")

        # 3. Initialize Camera & Pipeline
        camera = ThreadedCamera(
            source=0, 
            resolution=CFG["camera_res"], 
            fps=CFG["camera_fps"]
        ).start()
        
        pipeline = PipelineEngine(camera, extractor, inference_engine, detector, CFG)
        pipeline.start()
        log.info("SafeSteer Pipeline initiated and running.")
        
    except Exception as e:
        log.error(f"Failed to start pipeline: {e}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global pipeline
    if pipeline is None:
        start_pipeline()
        # Wait for camera warmup
        time.sleep(1.0)
    return jsonify({"status": "success"})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global pipeline
    if pipeline is not None:
        pipeline.stop()
        pipeline.camera.stop()
        pipeline = None
        log.info("Monitoring halted.")
    return jsonify({"status": "success"})

def generate_frames():
    global pipeline
    while True:
        if pipeline is None:
            time.sleep(0.1)
            continue
        
        try:
            # Consume from display queue (stale frame dropping is handled by PipelineEngine)
            display_data = pipeline.display_queue.get(timeout=1.0)
            if display_data is None: continue
            
            frame, status = display_data
            
            # Non-blocking encode (relative to the MJPEG loop)
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), CFG["jpeg_quality"]])
            if not ret: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except Exception as e:
            # log.debug(f"Stream error or queue timeout: {e}")
            time.sleep(0.01)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recalibrate')
def recalibrate():
    if pipeline:
        pipeline.detector.recalibrate()
        return jsonify({"status": "success", "message": "Recalibration triggered"})
    return jsonify({"status": "error", "message": "Pipeline not running"}), 400

@app.route('/stats')
def get_stats():
    global pipeline
    if pipeline is not None:
        with pipeline.detector.lock:
            stats = {
                "prob": round(pipeline.detector.last_ai_prob, 2),
                "ear": round(float(np.mean(pipeline.detector.ear_buffer)) if pipeline.detector.ear_buffer else 0.0, 3),
                "pitch": round(pipeline.detector.current_pitch, 1),
                "yaw": round(pipeline.detector.current_yaw, 1),
                "roll": round(pipeline.detector.current_roll, 1),
                "yawns": pipeline.detector.yawn_count,
                "droop_pct": int(pipeline.detector.droop_pct),
                "posture": pipeline.detector.posture_state,
                "perclos": round(float(np.mean(pipeline.detector.perclos_window)) if pipeline.detector.perclos_window else 0.0, 2),
                "status": "ACTIVE", 
            }
        # Merge metrics
        stats.update(pipeline.metrics.get_all())
        # Round metrics for JSON
        for k in ["processing_fps", "camera_fps", "inference_fps", "avg_landmark_ms", "avg_inference_ms"]:
            if k in stats: stats[k] = round(stats[k], 1)
            
        return jsonify(stats)
    return jsonify({"status": "waiting"})

if __name__ == '__main__':
    # Disable debug to prevent multiple pipeline instances
    import numpy as np # needed for get_stats
    app.run(host='127.0.0.1', port=5000, debug=False)
