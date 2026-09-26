"""Run road-object detection with YOLO-World and the Raspberry Pi camera."""

import argparse
import time

import cv2
import torch
from picamera2 import Picamera2
from ultralytics import YOLOWorld


def run(model: str, camera_id: int, width: int, height: int,
        num_threads: int, input_size: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Path to the customized YOLO-World checkpoint.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    input_size: The image size YOLO-World uses for inference.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0.0
  start_time = time.time()
  fps_avg_frame_count = 10

  # Limit PyTorch to the requested number of CPU threads for repeatable tests.
  torch.set_num_threads(num_threads)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # The class names and their text features were saved into this checkpoint on
  # Windows, so the Pi does not need to run the CLIP text encoder.
  detector = YOLOWorld(model)
  print("Model loaded successfully")
  print("Classes:", detector.names)
  print("Device:", device)

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
  row_size = 20
  left_margin = 24
  text_color = (0, 0, 255)

  # Continuously capture images from the camera and run inference
  try:
    while True:
      image = picam2.capture_array()

      # Original OpenCV camera capture implementation:
      # while cap.isOpened():
      #   success, image = cap.read()
      #   if not success:
      #     sys.exit(
      #         'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      #     )

      image = cv2.flip(image, 1)
      counter += 1

      # Picamera2 gives us RGB frames.
      # Ultralytics accepts an OpenCV-style BGR image. It resizes and pads the
      # frame internally before running the YOLO-World neural network.
      model_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      results = detector.predict(
          source=model_image,
          imgsz=input_size,
          conf=0.25,
          device=device,
          verbose=False
      )

      # Each result contains bounding boxes, confidence scores, and class IDs.
      for box in results[0].boxes:
        score = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = detector.names.get(class_id, "unknown")
        xmin, ymin, xmax, ymax = box.xyxy[0].int().tolist()

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            (0, 255, 0),
            2
        )

        label = "{}: {:.2f}".format(class_name, score)

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
        start_time = end_time

        # Headless output for SSH
        print("FPS =", fps)

      # Show the FPS
      fps_text = "FPS = {:.1f}".format(fps)
      cv2.putText(
          image,
          fps_text,
          (left_margin, row_size),
          cv2.FONT_HERSHEY_PLAIN,
          1,
          text_color,
          1
      )

      # Stop the program if the ESC key is pressed.
      # Nonheadless version:
      # if cv2.waitKey(1) == 27:
      #   break

      # Show live annotated camera feed.
      # Nonheadless version:
      # cv2.imshow('object_detector', image)

      # This runs through SSH without opening a desktop window. Press Ctrl+C
      # in the terminal when the test is complete.

  except KeyboardInterrupt:
    print("Stopping detector")
  finally:
    picam2.stop()

    # Original OpenCV camera cleanup:
    # cap.release()

    cv2.destroyAllWindows()


def main() -> None:
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
      "--model",
      help="Path to the customized YOLO-World model.",
      default="yolov8s-worldv2-road-10.pt"
  )

  parser.add_argument(
      "--cameraId",
      help="Camera ID retained for compatibility with the earlier script.",
      type=int,
      default=0
  )

  parser.add_argument(
      "--frameWidth",
      help="Width of the camera frame.",
      type=int,
      default=640
  )

  parser.add_argument(
      "--frameHeight",
      help="Height of the camera frame.",
      type=int,
      default=480
  )

  parser.add_argument(
      "--numThreads",
      help="Number of CPU threads used for inference.",
      type=int,
      default=1
  )

  parser.add_argument(
      "--inputSize",
      help="YOLO-World inference image size.",
      type=int,
      default=320
  )

  args = parser.parse_args()

  run(
      args.model,
      args.cameraId,
      args.frameWidth,
      args.frameHeight,
      args.numThreads,
      args.inputSize
  )


if __name__ == "__main__":
  main()