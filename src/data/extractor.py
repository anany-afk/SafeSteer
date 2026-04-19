import cv2
import math
import os
import time
import logging
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.config import CFG, RIGHT_EYE, LEFT_EYE, MOUTH, MOUTH_OUTER, HEAD_POSE_POINTS, NOSE_TIP, CHIN, MODEL_URL, MODEL_PATH
from src.utils.helpers import download_file

log = logging.getLogger(__name__)

class FacialFeatureExtractor:
    """
    extracts facial features from a single frame using mediapipe Tasks API.
    Optimized for performance and head-pose tracking.
    """

    def __init__(self, running_mode=vision.RunningMode.VIDEO):
        # 1. Download model if missing
        download_file(MODEL_URL, MODEL_PATH)

        # 2. Initialize Tasks API FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=running_mode
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.running_mode = running_mode
        self.last_ts = -1
        # Smaller tile grid for CLAHE to speed up
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    def _eye_aspect_ratio(self, lm_np, indices):
        """Vectorized EAR calculation (2D)."""
        pts = lm_np[indices, :2]
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        hz = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    def _mouth_aspect_ratio(self, lm_np):
        """Vectorized MAR calculation (2D)."""
        pts = lm_np[MOUTH, :2]
        v1 = np.linalg.norm(pts[2] - pts[6])
        v2 = np.linalg.norm(pts[3] - pts[5])
        hz = np.linalg.norm(pts[0] - pts[1])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    def _polygon_area(self, pts):
        """Vectorized shoelace formula."""
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _head_pose(self, lm_np, w, h):
        model_pts = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0],
        ], dtype=np.float64)

        img_pts = lm_np[HEAD_POSE_POINTS, :2].astype(np.float64)
        # Rescale normalized landmarks to pixel coordinates
        img_pts[:, 0] *= w
        img_pts[:, 1] *= h

        focal = w
        cam_mat = np.array([[focal, 0, w/2],
                            [0, focal, h/2],
                            [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            model_pts, img_pts, cam_mat, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        return float(euler[0, 0]), float(euler[1, 0]), float(euler[2, 0])

    def _preprocess(self, frame_bgr):
        """
        Optimized preprocessing: Skip slow steps if lighting is good.
        """
        # Step 1: Lightweight blur
        blurred = cv2.GaussianBlur(frame_bgr, (3, 3), 0)
        
        # Step 2: Check brightness on a subsampled frame
        gray = cv2.cvtColor(blurred[::4, ::4], cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray))
        
        # If lighting is "alright" (mean_lum > 100), skip expensive CLAHE/Gamma
        if mean_lum > 100:
            return blurred
            
        # Fallback for dark environments
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        # Using Lab color space for better illumination enhancement
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        
        if mean_lum < 70:
            gamma = 0.65 + (mean_lum / 70.0) * 0.55
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            l_clahe = cv2.LUT(l_clahe, table)
            
        enhanced = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
        return enhanced

    def extract(self, frame_bgr, timestamp_ms: int = None):
        """
        Extract features. In IMAGE mode, timestamp_ms is ignored.
        """
        raw_h, raw_w = frame_bgr.shape[:2]
        
        # Performance optimization: Resize frame for detection if too large
        process_w = 480
        if raw_w > process_w:
            scale = process_w / raw_w
            frame_detect = cv2.resize(frame_bgr, (process_w, int(raw_h * scale)))
        else:
            frame_detect = frame_bgr
            
        h, w = frame_detect.shape[:2]
        processed = self._preprocess(frame_detect)
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        if self.running_mode == vision.RunningMode.IMAGE:
            results = self.detector.detect(mp_image)
        elif self.running_mode == vision.RunningMode.VIDEO:
            if timestamp_ms is None:
                timestamp_ms = int(time.perf_counter() * 1000)
            
            # MediaPipe requires strictly increasing timestamps
            if timestamp_ms <= self.last_ts:
                timestamp_ms = self.last_ts + 1
            self.last_ts = timestamp_ms
            
            results = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            return np.zeros(20, dtype=np.float32), False

        if not results or not results.face_landmarks:
            rgb_raw = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image_raw = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_raw)
            if self.running_mode == vision.RunningMode.IMAGE:
                results = self.detector.detect(mp_image_raw)
            elif self.running_mode == vision.RunningMode.VIDEO:
                # Increment timestamp again for fallback call
                timestamp_ms += 1
                self.last_ts = timestamp_ms
                results = self.detector.detect_for_video(mp_image_raw, timestamp_ms)

        if not results or not results.face_landmarks:
            return np.zeros(20, dtype=np.float32), False

        lm = results.face_landmarks[0]
        # Convert landmarks to NumPy for vectorized math
        lm_np = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)

        # Geometric calculations (Vectorized 2D for robustness)
        ear_r = self._eye_aspect_ratio(lm_np, RIGHT_EYE)
        ear_l = self._eye_aspect_ratio(lm_np, LEFT_EYE)
        ear_m = (ear_r + ear_l) / 2.0
        ear_d = abs(ear_r - ear_l)
        mar = self._mouth_aspect_ratio(lm_np)
        
        eye_flag  = 1.0 if ear_m < CFG["ear_threshold"] else 0.0
        yawn_flag = 1.0 if mar  > CFG["mar_threshold"]  else 0.0
        
        mouth_top    = lm[13].y
        mouth_bottom = lm[14].y
        face_height  = abs(lm[10].y - lm[152].y) + 1e-6
        mouth_open   = abs(mouth_bottom - mouth_top) / face_height
        
        brow_l = lm_np[107, :2]
        brow_r = lm_np[336, :2]
        brow_furrow = np.linalg.norm(brow_l - brow_r)
        
        # Throttled head pose calculation
        if not hasattr(self, '_prev_pose') or (timestamp_ms % CFG.get('solve_pnp_interval', 1) == 0):
            self._prev_pose = self._head_pose(lm_np, w, h)
        pitch, yaw, roll = self._prev_pose
        
        nose  = lm_np[NOSE_TIP, :2]
        chin  = lm_np[CHIN, :2]
        n2c   = np.linalg.norm(nose - chin)
        
        r_pts = lm_np[RIGHT_EYE, :2]
        l_pts = lm_np[LEFT_EYE, :2]
        m_pts = lm_np[MOUTH_OUTER, :2]
        eye_r_area = self._polygon_area(r_pts)
        eye_l_area = self._polygon_area(l_pts)
        mouth_area = self._polygon_area(m_pts)
        
        try:
            p_l = lm_np[468, :2]
            p_r = lm_np[473, :2]
            pupil_dist = np.linalg.norm(p_l - p_r)
        except Exception:
            pupil_dist = 0.0
            
        l_upper  = lm_np[386, 1]
        l_corner = lm_np[362, 1]
        r_upper  = lm_np[159, 1]
        r_corner = lm_np[33, 1]
        l_droop  = max(0.0, l_upper - l_corner)
        r_droop  = max(0.0, r_upper - r_corner)
        
        features = np.array([
            ear_r, ear_l, ear_m, ear_d,
            mar, eye_flag, yawn_flag, mouth_open,
            brow_furrow, pitch/90.0, yaw/90.0, roll/90.0,
            n2c, eye_l_area*1000, eye_r_area*1000,
            mouth_area*1000, pupil_dist,
            l_droop*100, r_droop*100, 1.0
        ], dtype=np.float32)

        return features, True

    def close(self):
        self.detector.close()
