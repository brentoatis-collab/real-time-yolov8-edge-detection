"""
=========================================================
Week 7 Assignment
CST 544 – Artificial Intelligence in Computer Vision

Real-Time Computer Vision Application
Using YOLOv8 for Smart Surveillance

Student:
Brent Oatis

Description:
This application performs real-time object detection using
a pre-trained YOLOv8 model on webcam or video input.
Bounding boxes, class labels, confidence scores, FPS,
object counts, screenshots, and optional video recording
are supported.

Controls:
Q = Quit application
S = Save screenshot
=========================================================
"""

import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


def select_device() -> str:
    """Select the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time object detection using YOLOv8."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source. Use 0 for webcam or provide a video file path.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model file. Default is yolov8n.pt.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.40,
        help="Minimum confidence threshold for detections.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save annotated output video to assets/output.",
    )

    return parser.parse_args()


def open_video_source(source: str):
    """Open webcam or video file."""
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video source: {source}")

    return cap


def create_video_writer(cap, output_path: Path):
    """Create a video writer for saving annotated output."""
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or fps != fps:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )


def get_class_color(class_name: str):
    """
    Return consistent colors for detected classes.

    Colors are BGR because OpenCV uses BGR instead of RGB.
    """
    color_map = {
        "person": (0, 255, 0),
        "cell phone": (0, 165, 255),
        "laptop": (255, 0, 0),
        "keyboard": (255, 255, 0),
        "mouse": (255, 0, 255),
        "chair": (0, 255, 255),
        "book": (180, 105, 255),
        "cup": (255, 255, 255),
        "bottle": (255, 140, 0),
        "backpack": (128, 0, 255),
        "banana": (0, 255, 255),
        "apple": (0, 0, 255),
    }

    return color_map.get(class_name, (200, 200, 200))


def draw_detection_box(frame, box, label: str, confidence: float):
    """Draw bounding box and label on the video frame."""
    x1, y1, x2, y2 = map(int, box)
    color = get_class_color(label)

    display_text = f"{label} {confidence:.2f}"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    text_size, _ = cv2.getTextSize(
        display_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        2,
    )

    text_width, text_height = text_size

    cv2.rectangle(
        frame,
        (x1, max(y1 - text_height - 12, 0)),
        (x1 + text_width + 8, y1),
        color,
        -1,
    )

    cv2.putText(
        frame,
        display_text,
        (x1 + 4, max(y1 - 7, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


def draw_panel_background(frame, top_left, bottom_right, alpha=0.60):
    """Draw transparent panel background."""
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        top_left,
        bottom_right,
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame,
    )


def draw_dashboard(frame, fps: float, object_count: int, device: str, class_counts):
    """Draw professional dashboard overlay."""
    height, width = frame.shape[:2]

    draw_panel_background(
        frame,
        (10, 10),
        (min(width - 10, 560), 180),
        alpha=0.62,
    )

    title = "SMART SURVEILLANCE AI | YOLOv8 REAL-TIME DETECTION"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dashboard_lines = [
        title,
        f"FPS: {fps:.2f}",
        f"Objects Detected: {object_count}",
        f"Device: {device.upper()}",
        f"Time: {timestamp}",
        "Controls: Q = Quit | S = Screenshot",
    ]

    x = 24
    y = 38
    line_height = 25

    for index, text in enumerate(dashboard_lines):
        color = (0, 255, 255) if index == 0 else (255, 255, 255)

        cv2.putText(
            frame,
            text,
            (x, y + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
        )

    if class_counts:
        panel_width = 260
        panel_height = min(220, 40 + len(class_counts) * 28)

        draw_panel_background(
            frame,
            (width - panel_width - 10, 10),
            (width - 10, panel_height),
            alpha=0.62,
        )

        cv2.putText(
            frame,
            "Detected Classes",
            (width - panel_width + 5, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        sorted_counts = class_counts.most_common(6)

        for index, (class_name, count) in enumerate(sorted_counts):
            text = f"{class_name}: {count}"
            color = get_class_color(class_name)

            cv2.putText(
                frame,
                text,
                (width - panel_width + 5, 70 + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
            )


def save_screenshot(frame):
    """Save screenshot to assets/screenshots."""
    screenshot_dir = Path("assets/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = screenshot_dir / f"smart_surveillance_{timestamp}.jpg"

    cv2.imwrite(str(screenshot_path), frame)

    print(f"Screenshot saved: {screenshot_path}")


def run_detection():
    """Main real-time object detection loop."""
    args = parse_arguments()
    device = select_device()

    print("=" * 70)
    print("Real-Time Computer Vision Application Using YOLOv8")
    print("Smart Surveillance Local Deployment")
    print("=" * 70)
    print(f"Using device: {device.upper()}")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence threshold: {args.confidence}")
    print("=" * 70)

    print("\nLoading YOLO model...")
    model = YOLO(args.model)
    model.to(device)
    print("Model loaded successfully.\n")

    cap = open_video_source(args.source)

    output_writer = None

    if args.save:
        output_dir = Path("assets/output")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"smart_surveillance_output_{timestamp}.mp4"

        output_writer = create_video_writer(cap, output_path)

        print(f"Saving output video to: {output_path}")

    previous_time = time.time()
    total_frames = 0

    print("Starting real-time detection...")
    print("Press Q in the video window to quit.")
    print("Press S to save a screenshot.\n")

    while True:
        success, frame = cap.read()

        if not success:
            print("End of video stream or unable to read frame.")
            break

        total_frames += 1

        current_time = time.time()
        elapsed_time = current_time - previous_time
        previous_time = current_time

        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        results = model.predict(
            source=frame,
            conf=args.confidence,
            device=device,
            verbose=False,
        )

        detected_classes = []

        for result in results:
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                coordinates = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = model.names[class_id]

                draw_detection_box(
                    frame,
                    coordinates,
                    label,
                    confidence,
                )

                detected_classes.append(label)

        class_counts = Counter(detected_classes)
        object_count = len(detected_classes)

        draw_dashboard(
            frame,
            fps,
            object_count,
            device,
            class_counts,
        )

        cv2.imshow(
            "Smart Surveillance AI - YOLOv8 Real-Time Detection",
            frame,
        )

        if output_writer is not None:
            output_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            save_screenshot(frame)

        if key == ord("q"):
            print("Application closed by user.")
            break

    cap.release()

    if output_writer is not None:
        output_writer.release()

    cv2.destroyAllWindows()

    print("=" * 70)
    print("Application finished successfully.")
    print(f"Total frames processed: {total_frames}")
    print("Resources released.")
    print("=" * 70)


if __name__ == "__main__":
    run_detection()