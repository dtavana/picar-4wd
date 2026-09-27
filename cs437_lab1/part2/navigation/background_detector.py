'''
Runs the same YOLO-World detection as part2/detect.py in a background thread, so navigation
can ask whether something (e.g. a person) is in view while the car drives.

The camera and model settings match run() in detect.py. If those change there, update them here too.
'''
import os
import threading
import time

import cv2
import torch
from picamera2 import Picamera2
from ultralytics import YOLOWorld

# The model lives next to detect.py in the part2 folder, one level up from this file
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "yolov8s-worldv2-road-10.pt")
# Camera frame size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Number of CPU threads PyTorch uses for inference
NUM_THREADS = 1
# Image size YOLO-World uses for inference
INPUT_SIZE = 320
# Detections below this confidence are discarded by the model
MODEL_CONFIDENCE = 0.25


class BackgroundDetector:
    '''
    Runs YOLO-World detection continuously in a background thread.

    Other code can call seen_recently('person') at any time to ask whether a label was
    detected recently, without waiting on the camera.
    '''
    def __init__(self, min_score=0.5):
        # Only detections scoring at least min_score count as "seen"
        self.min_score = min_score
        self.fps = 0.0
        self.error = None
        self._detector = None
        self._picam2 = None
        self._device = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_seen = {}
        self._latest = []

    def start(self):
        '''
        Loads the model and camera, then starts detecting in the background
        '''
        # Loaded here rather than in the thread so loading errors are raised to the caller
        torch.set_num_threads(NUM_THREADS)
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._detector = YOLOWorld(MODEL_PATH)
        print("Model loaded successfully")
        print("Classes:", self._detector.names)

        self._picam2 = Picamera2()
        camera_config = self._picam2.create_preview_configuration(
            main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"}
        )
        self._picam2.configure(camera_config)
        self._picam2.start()

        self._thread = threading.Thread(target=self._run, name="object-detection", daemon=True)
        self._thread.start()
        return self

    def _detect_frame(self):
        '''
        Captures one frame and returns its detections, the same way run() in detect.py does
        '''
        image = cv2.flip(self._picam2.capture_array(), 1)
        # Picamera2 gives RGB frames, Ultralytics expects OpenCV-style BGR
        model_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self._detector.predict(
            source=model_image,
            imgsz=INPUT_SIZE,
            conf=MODEL_CONFIDENCE,
            device=self._device,
            verbose=False
        )
        detections = []
        for box in results[0].boxes:
            detections.append({
                "label": self._detector.names.get(int(box.cls[0]), "unknown"),
                "score": float(box.conf[0]),
                "box": tuple(box.xyxy[0].int().tolist()),
            })
        return detections

    def _run(self):
        fps_avg_frame_count = 10
        counter, start_time = 0, time.time()
        try:
            while not self._stop_event.is_set():
                detections = self._detect_frame()
                now = time.time()
                with self._lock:
                    self._latest = detections
                    for detection in detections:
                        if detection["score"] >= self.min_score:
                            self._last_seen[detection["label"]] = now

                # Calculate the FPS
                counter += 1
                if counter % fps_avg_frame_count == 0:
                    self.fps = fps_avg_frame_count / (time.time() - start_time)
                    start_time = time.time()
        except Exception as e:
            self.error = e
            print(f"OBJECT DETECTION STOPPED: {e!r}")

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def seen_recently(self, label, within=1.5):
        '''
        Returns whether `label` was detected in the last `within` seconds
        '''
        with self._lock:
            last_seen = self._last_seen.get(label)
        return last_seen is not None and time.time() - last_seen <= within

    def latest(self):
        '''
        Returns the detections from the most recent frame
        '''
        with self._lock:
            return list(self._latest)

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._picam2 is not None:
            self._picam2.stop()
