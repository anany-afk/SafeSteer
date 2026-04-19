import numpy as np
import logging

log = logging.getLogger(__name__)

class AccuracyValidator:
    """
    Validates that performance optimizations haven't degraded detection quality.
    Compares optimized outputs against known baselines if available.
    """

    def __init__(self):
        self.baseline_results = []
        self.optimized_results = []

    def log_comparison(self, baseline_prob, optimized_prob, frame_idx):
        """Logs a comparison for a specific frame."""
        diff = abs(baseline_prob - optimized_prob)
        if diff > 0.1:
            log.warning(f"Accuracy Drift at Frame {frame_idx}: diff={diff:.4f}")
        
    def generate_report(self):
        """Generates a summary of accuracy preservation."""
        if not self.baseline_results or not self.optimized_results:
            return "No data for accuracy reporting."
            
        b = np.array(self.baseline_results)
        o = np.array(self.optimized_results)
        
        mae = np.mean(np.abs(b - o))
        mse = np.mean((b - o) ** 2)
        
        report = (
            f"--- Accuracy Validation Report ---\n"
            f"Mean Absolute Error: {mae:.4f}\n"
            f"Mean Squared Error:  {mse:.4f}\n"
            f"Max Deviation:       {np.max(np.abs(b-o)):.4f}\n"
        )
        return report
