import sys
import time
import cv2
import logging
import threading
import numpy as np
from collections import deque
from enum import Enum
from src.config import CFG

# windows beep (wakeup alert)
try:
    import winsound
    BEEP_AVAILABLE = True
except ImportError:
    BEEP_AVAILABLE = False

log = logging.getLogger(__name__)

class CalibrationState(Enum):
    IDLE = 0
    COLLECTING = 1
    SUCCESS = 2
    FAIL = 3

class RealTimeDetector:
    # --- Tuning constants ---
    ALERT_PROB_THRESHOLD = 0.93      
    ALERT_CONSECUTIVE    = 35      
    PERCLOS_ALERT        = 0.45      
    SMOOTH_WINDOW        = 20      
    EAR_SMOOTH           = 12      
    ADAPT_RATE           = 0.001     
    
    YAWN_MIN_DURATION     = 1.2 
    YAWN_MAX_DURATION     = 4.5
    RECOVERY_MULTIPLIER  = 20      
    FACE_LOST_RESET_TIME = 150     
    ALERT_MAX_CTR = 100

    def __init__(self, model_path, scaler_path, cfg):
        import joblib
        
        self.cfg = cfg
        self.scaler = joblib.load(scaler_path)
        
        # Buffers
        self.frame_buffer   = deque(maxlen=cfg["sequence_len"])
        self.prob_buffer    = deque(maxlen=self.SMOOTH_WINDOW)
        self.ear_buffer     = deque(maxlen=self.EAR_SMOOTH)
        self.perclos_window = deque(maxlen=60) 
        self.pitch_buffer   = deque(maxlen=15)
        self.roll_buffer    = deque(maxlen=15)
        self.yaw_buffer     = deque(maxlen=15)

        # State Variables (Protected by lock)
        self.lock = threading.Lock()
        self.current_pitch    = 0.0
        self.current_yaw      = 0.0
        self.current_roll     = 0.0
        self.face_lost_ctr    = 0     
        
        # Calibration (Personal Bio-Profile)
        self.calib_state      = CalibrationState.COLLECTING
        self.pitch_baseline   = 0.0
        self.roll_baseline    = 0.0
        self.yaw_baseline     = 0.0
        self.ear_baseline     = 0.30
        
        self.pitch_samples    = []
        self.roll_samples     = []
        self.yaw_samples      = []
        self.ear_samples      = []
        
        self.droop_accumulator = 0.0
        self.droop_pct         = 0.0
        self.last_geom_ts      = time.perf_counter()
        
        self.head_droop_active = False
        self.eyes_closed_active = False
        self.eyes_closed_start_ts = None
        self.posture_state     = "CALIBRATING"
        self.calib_prompt      = "LOOK AT CAMERA"

        # Yawn state machine
        self.yawn_count        = 0
        self.mouth_is_open     = False
        self.yawn_start_time   = 0.0
        
        self.consec_alert = 0
        self.alert_active = False
        self.beep_active = False
        
        self.ear_thresh   = 0.15 
        self.mar_thresh   = cfg.get("mar_threshold", 0.50)
        
        self.session_start = time.time()
        self.last_ai_prob = 0.0

    def recalibrate(self):
        """Triggers a manual restart of the biometric calibration phase."""
        with self.lock:
            self.calib_state = CalibrationState.COLLECTING
            self.pitch_samples = []
            self.roll_samples = []
            self.yaw_samples = []
            self.ear_samples = []
            self.session_start = time.time()
            self.droop_accumulator = 0.0
            self.posture_state = "CALIBRATING"
            log.info("Manual recalibration triggered.")

    def update_geometric_state(self, feat):
        """FAST PATH: Update geometric counters (EAR, MAR, Pitch/Roll/Yaw)."""
        with self.lock:
            now_ts = time.perf_counter()
            delta_t = now_ts - self.last_geom_ts
            self.last_geom_ts = now_ts
            
            self.face_lost_ctr = 0 
            raw_ear = float(feat[2])
            raw_mar = float(feat[4])
            
            # 1. 3D Head Pose Smoothing
            self.pitch_buffer.append(float(feat[9] * 90.0))
            self.roll_buffer.append(float(feat[11] * 90.0))
            self.yaw_buffer.append(float(feat[10] * 90.0))
            
            self.current_pitch = float(np.mean(self.pitch_buffer))
            self.current_roll  = float(np.mean(self.roll_buffer))
            self.current_yaw   = float(np.mean(self.yaw_buffer))

            # 2. Simplified Human-Friendly Calibration (4s Median Capture)
            if self.calib_state == CalibrationState.COLLECTING:
                self.pitch_samples.append(self.current_pitch)
                self.roll_samples.append(self.current_roll)
                self.yaw_samples.append(self.current_yaw)
                self.ear_samples.append(raw_ear)
                
                elapsed = time.time() - self.session_start
                target_samples = self.cfg.get("calib_min_samples", 80)
                
                if elapsed > 4.0 and len(self.pitch_samples) >= target_samples:
                    self.pitch_baseline = float(np.median(self.pitch_samples))
                    self.roll_baseline  = float(np.median(self.roll_samples))
                    self.yaw_baseline   = float(np.median(self.yaw_samples))
                    self.ear_baseline   = float(np.median(self.ear_samples))
                    
                    self.ear_thresh = 0.15 
                    self.calib_state = CalibrationState.SUCCESS
                    self.posture_state = "UPRIGHT"
                return

            # 3. Vector-Based Robust Drooping Model (Fixed Diagonal Support)
            rate = 0.0
            p_delta = self.current_pitch - self.pitch_baseline # Positive = Nod Down
            r_delta = abs(self.current_roll - self.roll_baseline) # Magnitude of lean
            y_delta = abs(self.current_yaw - self.yaw_baseline)
            
            p_neutral = self.cfg.get("droop_neutral_zone", 15.0) 
            r_neutral = self.cfg.get("droop_roll_neutral", 8.0) 
            p_max = self.cfg.get("droop_max_offset", 35.0)
            r_max = self.cfg.get("droop_roll_max", 35.0)
            
            # Severity mapping (0.0 to 1.0 per axis)
            p_sev = np.clip((p_delta - p_neutral) / (p_max - p_neutral), 0, 1) if p_delta > p_neutral else 0
            r_sev = np.clip((r_delta - r_neutral) / (r_max - r_neutral), 0, 1) if r_delta > r_neutral else 0
            
            # Core SENSOR FUSION: Vector Magnitude of drooping
            # This captures diagonal slumping mathematically: sqrt(p^2 + r^2)
            total_sev = np.sqrt(p_sev**2 + r_sev**2) 
            
            is_drooping = total_sev > 0.35 # Lowered threshold for higher sensitivity
            self.posture_state = "DROOPING" if is_drooping else "UPRIGHT"

            # Yaw Looking-Away logic
            is_looking_away = y_delta > self.cfg.get("droop_yaw_limit", 25.0)
            
            if is_looking_away:
                rate = self.cfg.get("droop_dec_looking", -250.0)
            elif not is_drooping:
                rate = self.cfg.get("droop_dec_upright", -600.0)
            else:
                # Growing Accumulator
                # Apply extra diagonal bonus for concurrent multi-axis shifts
                growth_sev = total_sev
                if p_sev > 0.2 and r_sev > 0.2:
                    growth_sev *= self.cfg.get("droop_diagonal_bonus", 1.5)
                
                base_r = self.cfg.get("droop_inc_mild", 220.0)
                max_r  = self.cfg.get("droop_inc_heavy", 450.0)
                rate = base_r + min(1.3, growth_sev) * (max_r - base_r)
                
                if self.eyes_closed_active: rate *= 2.0
            
            self.droop_accumulator = np.clip(self.droop_accumulator + rate * delta_t, 0, 1000)
            self.droop_pct = float(self.droop_accumulator / 10.0)

            # 5. Sustained Eye Closure Timer
            self.ear_buffer.append(raw_ear)
            smooth_ear = float(np.mean(self.ear_buffer))
            
            is_closed = smooth_ear < self.ear_thresh 
            if is_closed and not is_looking_away:
                if self.eyes_closed_start_ts is None: self.eyes_closed_start_ts = now_ts
                closure_dur = now_ts - self.eyes_closed_start_ts
                if closure_dur > self.cfg.get("eye_closure_warning", 1.2):
                    self.eyes_closed_active = True
            else:
                self.eyes_closed_start_ts = None
                self.eyes_closed_active = False

            # Post-calibration adaptive baseline
            if smooth_ear > self.ear_baseline * 1.15: 
                self.ear_baseline = (self.ear_baseline * (1 - self.ADAPT_RATE) + smooth_ear * self.ADAPT_RATE)
                self.ear_thresh = float(np.clip(self.ear_baseline * 0.50, 0.12, 0.18))

            # 6. Yawn State Machine
            if raw_mar > self.mar_thresh:
                if not self.mouth_is_open:
                    self.mouth_is_open = True
                    self.yawn_start_time = now_ts
            else:
                if self.mouth_is_open:
                    y_dur = now_ts - self.yawn_start_time
                    if self.YAWN_MIN_DURATION < y_dur < self.YAWN_MAX_DURATION:
                        self.yawn_count += 1
                    self.mouth_is_open = False
            
            self.frame_buffer.append(feat)
            self.perclos_window.append(1 if (is_closed and not is_looking_away) else 0)
            self._update_alert_status(is_closed, is_looking_away)

    def update_ai_state(self, prob):
        with self.lock:
            self.last_ai_prob = float(prob)
            self.prob_buffer.append(prob)

    def _update_alert_status(self, is_closed, is_looking_away):
        smooth_prob = float(np.mean(self.prob_buffer)) if self.prob_buffer else 0.0
        perclos = float(np.mean(self.perclos_window)) if self.perclos_window else 0.0
        
        ai_drowsy = (smooth_prob > self.ALERT_PROB_THRESHOLD and is_closed and not is_looking_away)
        
        threshold = self.cfg.get("droop_alert_threshold", 800.0)
        recovery  = self.cfg.get("droop_recovery_level", 450.0)
        
        if self.droop_accumulator > threshold:
            self.head_droop_active = True
        elif self.droop_accumulator < recovery or self.posture_state == "UPRIGHT":
            self.head_droop_active = False

        drowsy_sig = ai_drowsy or perclos > self.PERCLOS_ALERT or self.head_droop_active or self.eyes_closed_active
        
        if drowsy_sig:
            self.consec_alert = min(self.ALERT_MAX_CTR, self.consec_alert + 1)
        else:
            rec_val = self.RECOVERY_MULTIPLIER if (self.droop_accumulator < 100 and not is_looking_away) else 2
            self.consec_alert = max(0, self.consec_alert - rec_val)
            
        prev_a = self.alert_active
        self.alert_active = (self.consec_alert >= self.ALERT_CONSECUTIVE or self.head_droop_active or (self.eyes_closed_active and not is_looking_away))
        if self.alert_active and not prev_a: self._trigger_beep()

    def _trigger_beep(self):
        if self.beep_active: return
        def _play():
            self.beep_active = True
            while self.alert_active:
                if BEEP_AVAILABLE:
                    winsound.Beep(1000, 250); time.sleep(0.05)
                else:
                    sys.stdout.write('\a'); sys.stdout.flush(); time.sleep(0.5)
            self.beep_active = False
        threading.Thread(target=_play, daemon=True).start()

    def create_display_frame(self, frame, ok, fps, metrics):
        with self.lock:
            h, w = frame.shape[:2]
            smooth_ear = float(np.mean(self.ear_buffer)) if self.ear_buffer else 0.0
            
            # --- Calibration Overlay Mode ---
            if self.calib_state == CalibrationState.COLLECTING:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 20), -1)
                frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
                
                cv2.putText(frame, "SAFESTEER INITIALIZING", (w//2-140, h//2-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Look naturally at the camera", (w//2-110, h//2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
                
                elapsed = time.time() - self.session_start
                cd = max(0, 4 - int(elapsed))
                if cd > 0:
                    cv2.putText(frame, str(cd), (w//2-15, h//2+35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, "DONE!", (w//2-30, h//2+35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                return frame, "CALIBRATING"

            # --- Live Dash Overlay Mode ---
            status = "HEALTHY"; colour = (0, 210, 0); show_wakeup = False
            
            if not ok:
                self.face_lost_ctr += 1
                if self.face_lost_ctr > self.FACE_LOST_RESET_TIME: self.alert_active = False 
                status = "FACE LOST"; colour = (160, 160, 160)
            else:
                if self.alert_active:
                    status = "DROWSY DETECTED"; colour = (0, 0, 220); show_wakeup = True
                elif self.droop_accumulator > 450.0 or self.eyes_closed_active: 
                    status = "WARNING: TIRED"; colour = (0, 140, 255) 

            # Drawing
            cv2.rectangle(frame, (0, 0), (w-1, h-1), colour, 6 if show_wakeup else 2)
            cv2.putText(frame, f"STATUS: {status}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)
            if show_wakeup:
                cv2.putText(frame, "WAKE UP!", (w//2 - 110, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 255), 3)
            
            lines = [
                f"Mode      : {self.cfg.get('performance_mode', 'AUTO')}",
                f"Drowsy Val: {int(self.last_ai_prob * 100)}%",
                f"EAR       : {smooth_ear:.3f} (BL: {self.ear_baseline:.3f})",
                f"Posture   : {self.posture_state}",
                f"Droopiness: {int(self.droop_pct)}%",
                f"Yawns     : {self.yawn_count}",
                f"Proc FPS  : {fps:.1f}",
            ]
            for i, txt in enumerate(lines):
                text_col = (230, 230, 230)
                if "Droopiness" in txt and self.droop_pct > 45: text_col = (0, 165, 255)
                cv2.putText(frame, txt, (10, 62 + i*27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_col, 2)
                
            return frame, status
