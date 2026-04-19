import time
import torch
import numpy as np
import logging
from src.config import CFG

log = logging.getLogger(__name__)

class PerformanceBenchmarker:
    """
    Benchmarks hardware and auto-selects performance mode.
    Runs dummy inference and measurement cycles.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg["device"]

    def run_benchmark(self, model=None, extractor=None):
        """
        Runs a quick benchmark to determine machine capability.
        Returns performance settings overrides.
        """
        log.info("Starting Startup Performance Benchmark...")
        
        samples = self.cfg.get("benchmark_samples", 15)
        inf_times = []
        ext_times = []

        # 1. Benchmark PyTorch Inference (if model provided)
        if model is not None:
            model.to(self.device).eval()
            dummy_input = torch.randn(1, self.cfg["sequence_len"], self.cfg["landmark_features"]).to(self.device)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(dummy_input)

            for _ in range(samples):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(dummy_input)
                inf_times.append(time.perf_counter() - start)

        # 2. Benchmark MediaPipe (if extractor provided)
        if extractor is not None:
            dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Warmup
            for _ in range(3):
                _ = extractor.extract(dummy_img)

            for _ in range(samples):
                start = time.perf_counter()
                _ = extractor.extract(dummy_img)
                ext_times.append(time.perf_counter() - start)

        avg_inf = np.mean(inf_times) * 1000 if inf_times else 10.0
        avg_ext = np.mean(ext_times) * 1000 if ext_times else 30.0
        total_cycle = avg_inf + avg_ext

        log.info(f"Benchmark Results: Inference={avg_inf:.1f}ms, Extraction={avg_ext:.1f}ms (Total={total_cycle:.1f}ms)")

        # Decision Logic
        mode = "BALANCED"
        overrides = {}

        if total_cycle < 22:  # Strong hardware (GPU or High-end CPU)
            mode = "HIGH_PERFORMANCE"
            overrides = {
                "inference_interval": 1,
                "landmark_interval": 1,
                "camera_res": (640, 480),
                "jpeg_quality": 90
            }
        elif total_cycle < 45: # Standard Laptop
            mode = "BALANCED"
            overrides = {
                "inference_interval": 2, # Every 2nd frame
                "landmark_interval": 1,
                "camera_res": (640, 480), 
                "jpeg_quality": 80
            }
        else: # Low-end / Budget CPU
            mode = "LOW_END"
            overrides = {
                "inference_interval": 3, # Every 3rd frame
                "landmark_interval": 2, # Skip landmarking every other frame
                "camera_res": (480, 360),
                "jpeg_quality": 70
            }

        log.info(f"Auto-selected mode: {mode}")
        return mode, overrides
