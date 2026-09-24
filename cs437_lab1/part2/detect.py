# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main script to run the object detection routine."""

import argparse
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2

# from tflite_support.task import core
# from tflite_support.task import processor
# from tflite_support.task import vision

# import utils


def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  picam2 = Picamera2()

  camera_config = picam2.create_preview_configuration(
      main={"size": (width, height), "format": "RGB888"}
  )

  picam2.configure(camera_config)
  picam2.start()

  # Original OpenCV camera capture implementation:
  # cap = cv2.VideoCapture(camera_id)
  # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model

  # Original tflite_support implementation:
  # base_options = core.BaseOptions(
  #     file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  # detection_options = processor.DetectionOptions(
  #     max_results=3, score_threshold=0.3)
  # options = vision.ObjectDetectorOptions(
  #     base_options=base_options, detection_options=detection_options)
  # detector = vision.ObjectDetector.create_from_options(options)

  # TensorFlow Lite implementation
  interpreter = tf.lite.Interpreter(
      model_path=model,
      num_threads=num_threads
  )

  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  input_height = input_details[0]['shape'][1]
  input_width = input_details[0]['shape'][2]
  input_dtype = input_details[0]['dtype']

  with open("labelmap.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

  print("Model loaded successfully")

  # Continuously capture images from the camera and run inference
  while True:
    image = picam2.capture_array()

    # Original OpenCV camera capture implementation:
    # while cap.isOpened():
    #   success, image = cap.read()
    #   if not success:
    #     sys.exit(
    #         'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    #     )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    # Picamera2 is already configured to provide RGB888.
    rgb_image = image

    # Original OpenCV conversion:
    # rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    # TensorFlow Lite expects a resized numpy array instead of TensorImage.
    resized_image = cv2.resize(
        rgb_image,
        (input_width, input_height)
    )

    input_tensor = np.expand_dims(resized_image, axis=0)

    if input_dtype == np.float32:
      input_tensor = input_tensor.astype(np.float32) / 255.0
    else:
      input_tensor = input_tensor.astype(input_dtype)

    # Run object detection estimation using the model.
    interpreter.set_tensor(
        input_details[0]['index'],
        input_tensor
    )

    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    # Draw keypoints and edges on input image
    image_height, image_width, _ = image.shape

    for i in range(len(scores)):
      if scores[i] < 0.3:
        continue

      ymin, xmin, ymax, xmax = boxes[i]

      xmin = int(xmin * image_width)
      xmax = int(xmax * image_width)
      ymin = int(ymin * image_height)
      ymax = int(ymax * image_height)

      cv2.rectangle(
          image,
          (xmin, ymin),
          (xmax, ymax),
          (0, 255, 0),
          2
      )

      # label = 'class {}: {:.2f}'.format(
      #     int(classes[i]),
      #     scores[i]
      # )

      # extracting the label using the class id index in labels array
      class_id = int(classes[i])

      if 0 <= class_id < len(labels):
        class_name = labels[class_id]
      else:
        class_name = "unknown"

      label = '{}: {:.2f}'.format(
          class_name,
          scores[i]
      )

      cv2.putText(
          image,
          label,
          (xmin, max(20, ymin - 10)),
          cv2.FONT_HERSHEY_PLAIN,
          1,
          (0, 255, 0),
          1
      )

      # Headless output for SSH
      print(label)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

      # Headless output for SSH
      print("FPS =", fps)

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)

    cv2.putText(
        image,
        fps_text,
        text_location,
        cv2.FONT_HERSHEY_PLAIN,
        font_size,
        text_color,
        font_thickness
    )

    # Stop the program if the ESC key is pressed.
    # Nonheadless version:
    # if cv2.waitKey(1) == 27:
    #   break

    # Show live annotated camera feed.
    # Nonheadless version:
    # cv2.imshow('object_detector', image)

  picam2.stop()

  # Original OpenCV camera cleanup:
  # cap.release()

  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet_lite0.tflite')

  parser.add_argument(
      '--cameraId',
      help='Id of camera.',
      required=False,
      type=int,
      default=0)

  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=640)

  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=480)

  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)

  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=False)

  args = parser.parse_args()

  run(
      args.model,
      int(args.cameraId),
      args.frameWidth,
      args.frameHeight,
      int(args.numThreads),
      bool(args.enableEdgeTPU)
  )


if __name__ == '__main__':
  main()