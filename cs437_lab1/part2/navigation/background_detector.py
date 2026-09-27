'''
Runs TensorFlow Lite object detection (EfficientDet-Lite0) in a background thread, so navigation
can ask whether something (e.g. a person) is in view while the car drives.

Uses the same model, labels and camera settings as the TensorFlow Lite version of part2/detect.py.
If those change there, update them here too.
'''
import os
import threading
import time

import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

# The model and labels live next to detect.py in the part2 folder, one level up from this file
PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PART2_DIR, "efficientdet_lite0.tflite")
LABELS_PATH = os.path.join(PART2_DIR, "labelmap.txt")
# Camera frame size
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# Number of CPU threads TensorFlow Lite uses for inference
NUM_THREADS = 4
# Detections below this confidence are discarded
MODEL_CONFIDENCE = 0.3


def load_labels(path=LABELS_PATH):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


class BackgroundDetector:
    '''
    Runs TensorFlow Lite detection continuously in a background thread.

    Other code can call seen_recently('person') at any time to ask whether a label was
    detected recently, without waiting on the camera.
    '''
    def __init__(self, min_score=0.5):
        # Only detections scoring at least min_score count as "seen"
        self.min_score = min_score
        self.fps = 0.0
        self.error = None
        self._interpreter = None
        self._input_details = None
        self._output_details = None
        self._labels = None
        self._picam2 = None
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
        self._labels = load_labels()
        self._interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=NUM_THREADS)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        print("Model loaded successfully")

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
        Captures one frame and returns its detections, the same way detect.py does
        '''
        image = cv2.flip(self._picam2.capture_array(), 1)

        # Resize the frame to the model's input size and add a batch dimension
        input_height = self._input_details[0]["shape"][1]
        input_width = self._input_details[0]["shape"][2]
        input_dtype = self._input_details[0]["dtype"]
        input_tensor = np.expand_dims(cv2.resize(image, (input_width, input_height)), axis=0)
        if input_dtype == np.float32:
            input_tensor = input_tensor.astype(np.float32) / 255.0
        else:
            input_tensor = input_tensor.astype(input_dtype)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()

        boxes = self._interpreter.get_tensor(self._output_details[0]["index"])[0]
        classes = self._interpreter.get_tensor(self._output_details[1]["index"])[0]
        scores = self._interpreter.get_tensor(self._output_details[2]["index"])[0]

        image_height, image_width, _ = image.shape
        detections = []
        for i in range(len(scores)):
            if scores[i] < MODEL_CONFIDENCE:
                continue
            ymin, xmin, ymax, xmax = boxes[i]
            class_id = int(classes[i])
            label = self._labels[class_id] if 0 <= class_id < len(self._labels) else "unknown"
            detections.append({
                "label": label,
                "score": float(scores[i]),
                "box": (int(xmin * image_width), int(ymin * image_height),
                        int(xmax * image_width), int(ymax * image_height)),
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
