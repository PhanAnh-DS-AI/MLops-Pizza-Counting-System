# src/detection/tracking.py
import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torchreid

# Test tracking function from video
def track_pizzas_from_video(video_path, output_path, conf_thres=0.5):
    model = YOLO("models/yolov8l.pt")
    tracker = DeepSort(
        max_age=120,         # Number of missed frames before a track is deleted
        n_init=3,           # Number of consecutive detections before track is confirmed
        nms_max_overlap=1.0,# Allow full overlap for pizzas on trays
        max_cosine_distance=0.3, # Appearance feature matching threshold
        nn_budget=100,      # Max number of appearance features to store per track
        embedder="mobilenet", # Lightweight and fast
        half=True           # Use half precision for speed
    )

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Tracking")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.model.names[cls_id]

            if label != "pizza" or conf < conf_thres:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, 'pizza'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            color = (0, 165, 255)
            print(f"Tracking pizza ID {track_id}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"pizza ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()
    print(f"Tracking video saved to: {output_path}")

# ============== Tracker ==============  
def pizza_tracker(video_path, model_path="models/yolov8l.pt", conf_thres=0.5, tracker_params=None):
    if tracker_params is None:
        tracker_params = dict(
            max_age=90,
            n_init=2,
            nms_max_overlap=0.7,
            max_cosine_distance=0.5,
            nn_budget=100,
            embedder="mobilenet",
            half=True
        )
    model = YOLO(model_path)
    tracker = DeepSort(**tracker_params)

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.model.names[cls_id]

            if label != "pizza" or conf < conf_thres:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, 'pizza'))

        tracks = []
        ds_tracks = tracker.update_tracks(detections, frame=frame)
        for track in ds_tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            tracks.append([x1, y1, x2, y2, 1.0, track_id])  # 1.0 as dummy score

        yield frame, tracks

    cap.release()
    yield None, None 


if __name__ == "__main__":
    track_pizzas_from_video(
        video_path="data/raw_videos/cut_video_test/1465_CH02_20250607170555_172408 - Trim.mp4", 
        output_path="data/results/tracked_1465_CH02 - Trim.mp4",
        conf_thres=0.5
    )