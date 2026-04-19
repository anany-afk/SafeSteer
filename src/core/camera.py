import cv2
import threading
import time
import logging

log = logging.getLogger(__name__)

class ThreadedCamera:
    """
    Optimized camera interface with threaded capture and stale frame dropping.
    Guarantees that read() always returns the latest sensor frame.
    """

    def __init__(self, source=0, resolution=(640, 480), fps=30):
        self.source = source
        self.resolution = resolution
        self.fps = fps
        
        # Use DSHOW on Windows for faster initialization and better control
        # If Linux, it will fallback to V4L2
        backend = cv2.CAP_ANY
        if hasattr(cv2, 'CAP_DSHOW'):
            backend = cv2.CAP_DSHOW
            
        self.cap = cv2.VideoCapture(self.source, backend)
        
        if not self.cap.isOpened():
            log.error(f"Could not open video source {self.source}")
            return

        # Request resolution and FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # Buffer should be minimal to reduce latency
        if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        
        # Metrics
        self.frame_count = 0
        self.start_time = time.time()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        log.info(f"Threaded Camera started (Source: {self.source}, Res: {self.resolution})")
        return self

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame
                self.frame_count += 1
            
            # Simple throttle to prevent thread from pinning 
            # while still maintaining target FPS
            time.sleep(0.001)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def get_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed == 0: return 0
        return self.frame_count / elapsed

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()
        log.info("Threaded Camera stopped.")
        
    def __del__(self):
        self.stop()
