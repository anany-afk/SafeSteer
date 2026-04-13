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

    def _euclidean(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def _eye_aspect_ratio(self, lm, indices, w, h):
        pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
        v1 = self._euclidean(pts[1], pts[5])
        v2 = self._euclidean(pts[2], pts[4])
        hz = self._euclidean(pts[0], pts[3])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    def _mouth_aspect_ratio(self, lm, w, h):
        pts = [(lm[i].x * w, lm[i].y * h) for i in MOUTH]
        v1 = self._euclidean(pts[2], pts[6])
        v2 = self._euclidean(pts[3], pts[5])
        hz = self._euclidean(pts[0], pts[1])
        return (v1 + v2) / (2.0 * hz + 1e-6)

    def _polygon_area(self, pts):
        n = len(pts)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1]
            area -= pts[j][0] * pts[i][1]
        return abs(area) / 2.0

    def _head_pose(self, lm, w, h):
        model_pts = np.array([
            [0.0, 0.0, 0.0],
            [0.0, -330.0, -65.0],
            [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0],
            [-150.0, -150.0, -125.0],
            [150.0, -150.0, -125.0],
        ], dtype=np.float64)

        img_pts = np.array([
            (lm[HEAD_POSE_POINTS[i]].x * w,
             lm[HEAD_POSE_POINTS[i]].y * h)
            for i in range(6)
        ], dtype=np.float64)

        focal = w
        cam_mat = np.array([[focal, 0, w/2],
                            [0, focal, h/2],
                            [0, 0, 1]], dtype=np.float64)
        dist = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            model_pts, img_pts, cam_mat, dist,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
        # pitch (X), yaw (Y), roll (Z)
        pitch = float(euler[0, 0])
        yaw   = float(euler[1, 0])
        roll  = float(euler[2, 0])
        return pitch, yaw, roll

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
        h, w = frame_bgr.shape[:2]
        processed = self._preprocess(frame_bgr)
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

        # Geometric calculations
        ear_r = self._eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear_l = self._eye_aspect_ratio(lm, LEFT_EYE,  w, h)
        ear_m = (ear_r + ear_l) / 2.0
        ear_d = abs(ear_r - ear_l)
        mar = self._mouth_aspect_ratio(lm, w, h)
        
        eye_flag  = 1.0 if ear_m < CFG["ear_threshold"] else 0.0
        yawn_flag = 1.0 if mar  > CFG["mar_threshold"]  else 0.0
        
        mouth_top    = lm[13].y * h
        mouth_bottom = lm[14].y * h
        face_height  = abs(lm[10].y - lm[152].y) * h + 1e-6
        mouth_open   = abs(mouth_bottom - mouth_top) / face_height
        
        brow_l = (lm[107].x * w, lm[107].y * h)
        brow_r = (lm[336].x * w, lm[336].y * h)
        brow_furrow = self._euclidean(brow_l, brow_r) / (w + 1e-6)
        
        pitch, yaw, roll = self._head_pose(lm, w, h)
        
        nose  = (lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h)
        chin  = (lm[CHIN].x * w,     lm[CHIN].y * h)
        n2c   = self._euclidean(nose, chin) / (h + 1e-6)
        
        r_pts = [(lm[i].x*w, lm[i].y*h) for i in RIGHT_EYE]
        l_pts = [(lm[i].x*w, lm[i].y*h) for i in LEFT_EYE]
        m_pts = [(lm[i].x*w, lm[i].y*h) for i in MOUTH_OUTER]
        eye_r_area = self._polygon_area(r_pts) / (w*h + 1e-6)
        eye_l_area = self._polygon_area(l_pts) / (w*h + 1e-6)
        mouth_area = self._polygon_area(m_pts) / (w*h + 1e-6)
        
        try:
            p_l = (lm[468].x * w, lm[468].y * h)
            p_r = (lm[473].x * w, lm[473].y * h)
            pupil_dist = self._euclidean(p_l, p_r) / (w + 1e-6)
        except Exception:
            pupil_dist = 0.0
            
        l_upper  = lm[386].y * h
        l_corner = lm[362].y * h
        r_upper  = lm[159].y * h
        r_corner = lm[33].y  * h
        l_droop  = max(0.0, l_upper - l_corner) / (h + 1e-6)
        r_droop  = max(0.0, r_upper - r_corner) / (h + 1e-6)
        
        conf = 1.0

        features = np.array([
            ear_r, ear_l, ear_m, ear_d,
            mar, eye_flag, yawn_flag, mouth_open,
            brow_furrow, pitch/90.0, yaw/90.0, roll/90.0,
            n2c, eye_l_area*1000, eye_r_area*1000,
            mouth_area*1000, pupil_dist,
            l_droop*100, r_droop*100, conf
        ], dtype=np.float32)

        return features, True

    def close(self):
        self.detector.close()
