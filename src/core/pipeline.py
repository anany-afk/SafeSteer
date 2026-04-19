import time
import threading
import queue
import logging
import numpy as np
from collections import deque

log = logging.getLogger(__name__)

class PipelineMetrics:
    """Tracks latency and FPS across different stages."""
    def __init__(self):
        self.stats = {
            "camera_fps": 0.0,
            "processing_fps": 0.0,
            "inference_fps": 0.0,
            "avg_landmark_ms": 0.0,
            "avg_inference_ms": 0.0,
            "dropped_frames": 0,
            "queue_latency_ms": 0.0
        }
        self._lock = threading.Lock()
        
    def update(self, key, value):
        with self._lock:
            self.stats[key] = value

    def get_all(self):
        with self._lock:
            return self.stats.copy()

class PipelineEngine:
    """
    Coordinates the multi-stage detection pipeline.
    Uses bounded buffers to ensure zero-lag ('stale frame dropping').
    """

    def __init__(self, camera, extractor, inference_engine, detector, cfg):
        self.camera = camera
        self.extractor = extractor
        self.inference_engine = inference_engine
        self.detector = detector
        self.cfg = cfg
        
        self.metrics = PipelineMetrics()
        self.running = False
        
        # Bounded buffers (1-slot) to prioritize newest frames
        self.frame_queue = queue.Queue(maxsize=1)
        self.display_queue = queue.Queue(maxsize=1)
        
        # State tracking
        self.latest_state = {
            "prob": 0.0,
            "features": None,
            "ok": False,
            "fps": 0.0
        }
        self._state_lock = threading.Lock()

    def start(self):
        self.running = True
        self.threads = [
            threading.Thread(target=self._worker_loop, name="WorkerStage", daemon=True),
            threading.Thread(target=self._inference_loop, name="InferenceStage", daemon=True),
        ]
        for t in self.threads:
            t.start()
        log.info("Pipeline Engine started.")

    def _worker_loop(self):
        """
        FAST PATH: Camera -> Landmarking -> Geometry -> State Update
        """
        prev_time = time.time()
        inf_counter = 0
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            
            start_ext = time.perf_counter()
            
            # 1. Feature Extraction (Fast Path)
            ts = int(time.perf_counter() * 1000)
            feat, ok = self.extractor.extract(frame, timestamp_ms=ts)
            
            ext_ms = (time.perf_counter() - start_ext) * 1000
            self.metrics.update("avg_landmark_ms", ext_ms)
            
            # 2. Update Detector State (Immediate Counters)
            # This handles EAR/MAR/Droop immediately
            status = "N/A"
            if ok:
                # Synchronous state update in detector
                self.detector.update_geometric_state(feat)
                
                # 3. Handle Inference Interval (Slow Path trigger)
                inf_counter += 1
                if inf_counter >= self.cfg.get("inference_interval", 1):
                    inf_counter = 0
                    # Push sequence to inference thread (this might be throttled)
                    if len(self.detector.frame_buffer) == self.cfg["sequence_len"]:
                        seq = list(self.detector.frame_buffer)
                        # We don't block here; if inference is busy, we skip
                        # but in a production system we might want to ensure 
                        # the latest sequence is always being inferred.
                        pass # handled by inference loop checking detector buffer
            
            # 4. Prepare Display Frame
            now = time.time()
            proc_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            self.metrics.update("processing_fps", proc_fps)
            
            # Overlay and buffer for MJPEG
            # Note: _overlay is actually fast enough to run here
            display_frame, status = self.detector.create_display_frame(frame.copy(), ok, proc_fps, self.metrics.get_all())
            
            # Push to MJPEG stream
            try:
                if self.display_queue.full():
                    self.display_queue.get_nowait()
                    self.metrics.update("dropped_frames", self.metrics.stats["dropped_frames"] + 1)
                self.display_queue.put_nowait((display_frame, status))
            except queue.Full:
                pass

    def _inference_loop(self):
        """
        SLOW PATH: Periodic Model Inference
        """
        while self.running:
            # Check if we have enough frames for an inference cycle
            if len(self.detector.frame_buffer) < self.cfg["sequence_len"]:
                time.sleep(0.05)
                continue
                
            start_inf = time.perf_counter()
            
            # Get latest sequence (thread-safe copy?)
            # For simplicity, we assume detector.frame_buffer (deque) is thread-safe for reads
            seq = np.array(list(self.detector.frame_buffer), dtype=np.float32)
            
            # Transform and Infer
            seq_scaled = self.detector.scaler.transform(seq)
            prob = self.inference_engine.infer(seq_scaled)
            
            inf_ms = (time.perf_counter() - start_inf) * 1000
            self.metrics.update("avg_inference_ms", inf_ms)
            self.metrics.update("inference_fps", 1000.0 / max(inf_ms, 1e-6))
            
            # Update detector with AI result
            self.detector.update_ai_state(prob)
            
            # Throttling to target frequency
            time.sleep(0.001)

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=1.0)
        if self.extractor:
            self.extractor.close()
        log.info("Pipeline Engine stopped.")
