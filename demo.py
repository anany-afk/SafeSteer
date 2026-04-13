import os
import sys
import argparse
import logging
from src.config import CFG
from src.core.detector import RealTimeDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser(description="Run Real-time Driver Drowsiness Detection")
    p.add_argument("--model",  default=os.path.join(CFG["output_dir"], "drowsiness_model_full.pth"))
    p.add_argument("--scaler", default=os.path.join(CFG["output_dir"], "scaler.pkl"))
    return p.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.model):
        log.error(f"Model file not found at {args.model}")
        sys.exit(1)
        
    if not os.path.exists(args.scaler):
        log.error(f"Scaler file not found at {args.scaler}")
        sys.exit(1)

    log.info("Starting Real-time Detector...")
    detector = RealTimeDetector(args.model, args.scaler, CFG)
    detector.run()

if __name__ == "__main__":
    main()
