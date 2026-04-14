import sys
import time
import cv2
import torch
import joblib
import logging
import threading
import numpy as np
from collections import deque
from mediapipe.tasks.python import vision
from src.config import CFG
from src.data.extractor import FacialFeatureExtractor
from src.models.architecture import DrowsinessNet

# windows beep (wakeup alert)
try:
    import winsound
    BEEP_AVAILABLE = True
except ImportError:
    BEEP_AVAILABLE = False

log = logging.getLogger(__name__)

class RealTimeDetector:
    # --- Tuning constants ---
    ALERT_PROB_THRESHOLD = 0.93      # Higher certainty required
    ALERT_CONSECUTIVE    = 35      # Faster trigger (was 65)
    PERCLOS_ALERT        = 0.45      # Combined with shorter window for higher sensitivity
    SMOOTH_WINDOW        = 20      
    EAR_SMOOTH           = 12      
    ADAPT_RATE           = 0.001     
    
    # Head-Pose Thresholds
    PITCH_THRESHOLD         = -20.0   # Balanced (was -16.0, originally -25.0)
    PITCH_CONSECUTIVE_LIMIT = 60      # Balanced (was 45, originally 100)
    YAW_LOOK_AWAY_THRESHOLD = 22.0    
    
    # Yawn logic
    YAWN_COOLDOWN        = 45
    YAWN_MIN_FRAMES      = 15

    # Recovery Dynamics
    RECOVERY_MULTIPLIER  = 20      # Faster reset (was 12)
    FACE_LOST_RESET_TIME = 150     
    
    # Counter Caps (To prevent recovery lag)
    PITCH_MAX_CTR = 80             # Lower cap to prevent "winding up" state
    ALERT_MAX_CTR = 100

    # Window Name
    WINDOW_NAME = "SafeSteer Detector"

    def __init__(self, model_path, scaler_path, cfg):
        self.device = cfg["device"]
        self.cfg    = cfg

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = DrowsinessNet(
            feat_dim    = checkpoint["feat_dim"],
            seq_len     = cfg["sequence_len"],
            cnn_channels= cfg["cnn_channels"],
            lstm_hidden = cfg["lstm_hidden"],
            lstm_layers = cfg["lstm_layers"],
            fc_hidden   = cfg["fc_hidden"],
            dropout     = 0.0,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        self.extractor = FacialFeatureExtractor(running_mode=vision.RunningMode.VIDEO)

        self.frame_buffer   = deque(maxlen=cfg["sequence_len"])
        self.prob_buffer    = deque(maxlen=self.SMOOTH_WINDOW)
        self.ear_buffer     = deque(maxlen=self.EAR_SMOOTH)
        self.perclos_window = deque(maxlen=60) # Shorter window (was 150)

        # State Variables
        self.pitch_consec_ctr = 0
        self.current_pitch    = 0.0
        self.current_yaw      = 0.0
        self.face_lost_ctr    = 0     

        self.yawn_count        = 0
        self.yawn_cooldown_ctr = 0
        self.in_yawn           = False
        self.yawn_open_ctr     = 0
        
        self.consec_alert = 0
        self.alert_active = False
        self.session_active = True
        
        self.ear_baseline  = None
        self.mar_baseline  = None
        self.ear_thresh    = None
        self.mar_thresh    = None
        self.session_start = None
        self.beep_active = False

        # Web Mode State
        self.web_mode = False
        self.stop_flag = False
        self.latest_frame = None
        self.latest_stats = {}

    def _beep(self):
        if self.beep_active or self.web_mode:
            return   
        def _play():
            self.beep_active = True
            log.info("Alert loop started")
            while self.alert_active and not self.web_mode:
                if BEEP_AVAILABLE:
                    winsound.Beep(1000, 200) # Shorter beep (was 500ms)
                    time.sleep(0.05)
                else:
                    sys.stdout.write('\a')
                    sys.stdout.flush()
                    time.sleep(0.5)
            time.sleep(0.3)
            self.beep_active = False
            log.info("Alert loop stopped")
        threading.Thread(target=_play, daemon=True).start()

    def _calibrate(self, cap):
        ears, mars  = [], []
        calib_total = 90 
        collected   = 0
        if not self.web_mode:
            cv2.namedWindow(self.WINDOW_NAME)
        
        while collected < calib_total:
            if self.stop_flag: return 0.3, 0.6
            ret, frame = cap.read()
            if not ret: break
            ts = int(time.perf_counter() * 1000)
            feat, ok = self.extractor.extract(frame, timestamp_ms=ts)
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (40, 40, 40), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            pct = int((collected / calib_total) * (w - 80))
            cv2.rectangle(frame, (40, h-60), (w-40, h-40), (60, 60, 60), -1)
            cv2.rectangle(frame, (40, h-60), (40+pct, h-40), (0, 210, 100), -1)
            cv2.putText(frame, "CALIBRATING — LOOK STRAIGHT", (50, h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if self.web_mode:
                self.latest_frame = frame
                self.latest_stats = {"status": "CALIBRATING", "progress": int(collected/calib_total*100)}
                time.sleep(0.01)
            else:
                cv2.imshow(self.WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.session_active = False
                    return 0.3, 0.6
            if ok:
                ears.append(float(feat[2]))
                mars.append(float(feat[4]))
                collected += 1
        if not ears: return 0.25, 0.60
        ear_base = float(np.mean(ears))
        mar_base = float(np.mean(mars))
        self.ear_baseline = ear_base
        self.mar_baseline = mar_base
        return float(ear_base * 0.70), float(mar_base * 1.70)

    def _overlay(self, frame, smooth_ear, smooth_prob, perclos, fps, elapsed_s, ok):
        h, w = frame.shape[:2]
        status = "ALERT"
        colour = (0, 210, 0)
        show_wakeup = False
        if not ok:
            if self.pitch_consec_ctr > 30:
                status = "DROOPING (LOST)..."
                colour = (0, 80, 255)
                show_wakeup = self.alert_active
            else:
                status = "FACE LOST"
                colour = (160, 160, 160)
        else:
            if self.alert_active:
                status = "DROWSY DETECTED"
                colour = (0, 0, 220)
                show_wakeup = True
            elif self.pitch_consec_ctr > 30: 
                 status = "DROOPING..."
                 colour = (0, 140, 255) 
        cv2.rectangle(frame, (0, 0), (w-1, h-1), colour, 6 if show_wakeup else 2)
        cv2.putText(frame, f"STATUS: {status}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
        if show_wakeup:
            cv2.putText(frame, "WAKE UP!", (w//2 - 110, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 255), 3)
        lines = [
            f"Prob    : {smooth_prob:.2f}",
            f"EAR     : {smooth_ear:.3f}",
            f"Pitch   : {self.current_pitch:.1f} deg",
            f"Yawns   : {self.yawn_count}",
            f"Droop   : {int((self.pitch_consec_ctr/self.PITCH_CONSECUTIVE_LIMIT)*100)}%",
            f"FPS     : {fps:.1f}",
        ]
        dashboard_colour = (230, 230, 230)
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, 62 + i*27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, dashboard_colour, 2)
        return frame, status

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return
        self.ear_thresh, self.mar_thresh = self._calibrate(cap)
        if not self.session_active or self.stop_flag:
            cap.release()
            return

        self.session_start = time.time()
        prev_time = time.time()
        while not self.stop_flag:
            ret, frame = cap.read()
            if not ret: break
            ts = int(time.perf_counter() * 1000)
            feat, ok = self.extractor.extract(frame, timestamp_ms=ts)
            if ok:
                self.face_lost_ctr = 0 
                raw_ear = float(feat[2]); raw_mar = float(feat[4])
                self.current_pitch = float(feat[9] * 90.0); self.current_yaw = float(feat[10] * 90.0)
                if self.current_pitch < self.PITCH_THRESHOLD:
                    self.pitch_consec_ctr = min(self.PITCH_MAX_CTR, self.pitch_consec_ctr + 1)
                else:
                    if self.current_pitch > -8.0:
                        self.pitch_consec_ctr = max(0, self.pitch_consec_ctr - self.RECOVERY_MULTIPLIER)
                    else:
                        self.pitch_consec_ctr = max(0, self.pitch_consec_ctr - 3)
                self.ear_buffer.append(raw_ear)
                smooth_ear = float(np.mean(self.ear_buffer))
                if smooth_ear > self.ear_thresh * 1.15: 
                    self.ear_baseline = (self.ear_baseline * (1 - self.ADAPT_RATE) + smooth_ear * self.ADAPT_RATE)
                    self.ear_thresh = float(np.clip(self.ear_baseline * 0.70, 0.18, 0.35))
                is_looking_away = abs(self.current_yaw) > self.YAW_LOOK_AWAY_THRESHOLD
                is_yawning = raw_mar > self.mar_thresh
                if is_yawning:
                    self.yawn_open_ctr += 1; self.in_yawn = True
                else:
                    if self.in_yawn and self.yawn_open_ctr > self.YAWN_MIN_FRAMES:
                        self.yawn_count += 1
                    self.in_yawn = False; self.yawn_open_ctr = 0
                self.frame_buffer.append(feat)
                prob = 0.0
                if len(self.frame_buffer) == self.cfg["sequence_len"]:
                    seq = np.array(self.frame_buffer, dtype=np.float32)
                    seq_scaled = self.scaler.transform(seq)
                    X_t = torch.tensor(seq_scaled[None], dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        logits = self.model(X_t); prob = float(torch.softmax(logits, dim=1)[0, 1])
                self.prob_buffer.append(prob); smooth_prob = float(np.mean(self.prob_buffer))
                perclos = float(np.mean(self.perclos_window)) if self.perclos_window else 0.0
                ear_drowsy = smooth_ear < self.ear_thresh
                if is_looking_away:
                    ai_drowsy = False; self.perclos_window.append(0)
                else:
                    ai_drowsy = (smooth_prob > self.ALERT_PROB_THRESHOLD and ear_drowsy)
                    self.perclos_window.append(1 if ear_drowsy else 0)
                head_droop = (self.pitch_consec_ctr >= self.PITCH_CONSECUTIVE_LIMIT)
                drowsy_signal = ai_drowsy or perclos > self.PERCLOS_ALERT or head_droop
                if drowsy_signal:
                    self.consec_alert = min(self.ALERT_MAX_CTR, self.consec_alert + 1)
                else:
                    recovery = self.RECOVERY_MULTIPLIER if (self.current_pitch > -8.0 and not is_looking_away) else 2
                    self.consec_alert = max(0, self.consec_alert - recovery)
                prev_alert = self.alert_active
                self.alert_active = (self.consec_alert >= self.ALERT_CONSECUTIVE or head_droop)
                if self.alert_active and not prev_alert: self._beep()
            else:
                self.face_lost_ctr += 1
                if self.current_pitch < self.PITCH_THRESHOLD:
                    self.pitch_consec_ctr = min(self.PITCH_MAX_CTR, self.pitch_consec_ctr + 1)
                else:
                    self.pitch_consec_ctr = max(0, self.pitch_consec_ctr - 1)
                self.consec_alert = max(0, self.consec_alert - 1)
                self.alert_active = (self.consec_alert >= self.ALERT_CONSECUTIVE or (self.pitch_consec_ctr >= self.PITCH_CONSECUTIVE_LIMIT))
                if self.face_lost_ctr > self.FACE_LOST_RESET_TIME:
                    self.consec_alert = 0; self.pitch_consec_ctr = 0; self.alert_active = False
            
            now = time.time(); fps = 1.0 / max(now - prev_time, 1e-6); prev_time = now
            elapsed = now - self.session_start
            
            smooth_ear = float(np.mean(self.ear_buffer)) if self.ear_buffer else 0.0
            smooth_prob = float(np.mean(self.prob_buffer)) if self.prob_buffer else 0.0
            perclos = float(np.mean(self.perclos_window)) if self.perclos_window else 0.0
            
            display = self.extractor._preprocess(frame)
            display, status = self._overlay(display, smooth_ear, smooth_prob, perclos, fps, elapsed, ok)
            
            if self.web_mode:
                self.latest_frame = display
                self.latest_stats = {
                    "prob": round(smooth_prob, 2),
                    "ear": round(smooth_ear, 3),
                    "pitch": round(self.current_pitch, 1),
                    "yaw": round(self.current_yaw, 1),
                    "yawns": self.yawn_count,
                    "droop_pct": int((self.pitch_consec_ctr/self.PITCH_CONSECUTIVE_LIMIT)*100),
                    "perclos": round(perclos, 2),
                    "status": status,
                    "elapsed": int(elapsed),
                    "fps": round(fps, 1)
                }
                # Throttling to keep it around 30FPS for the MJPEG stream
                time.sleep(0.01)
            else:
                cv2.imshow(self.WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                
        cap.release()
        if not self.web_mode:
            cv2.destroyAllWindows()
        self.extractor.close()
