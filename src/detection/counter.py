import cv2
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import csv
from src.config.camera_zones import CAMERA_ZONES 
from src.detection.utils import point_in_polygon, draw_polygon
from src.detection.tracking import pizza_tracker

def get_camera_key(video_path):
    """Extract camera key from video filename, e.g. '1461_CH01' from '1461_CH01_20250607193711_203711.mp4'."""
    base = os.path.basename(video_path)
    parts = base.split("_")
    return f"{parts[0]}_{parts[1]}"

def track_and_count_pizzas(
    video_path, 
    output_path, 
    conf_thres=0.5, 
    count_polygon=None,
    stop_flag=lambda: False
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    pizza_tracks = {}
    counted_ids = set()
    pizza_count = 0
    sale_events = []

    last_positions = {}  # track_id: (cx, cy, frame_idx)
    recently_lost = []   # [{'cx':..., 'cy':..., 'frame':...}]

    try:
        frame_idx = 0
        for frame, tracks in pizza_tracker(video_path, conf_thres=conf_thres):
            if stop_flag():
                print("Counting stopped by user.")
                break
            if frame is None:
                break
            frame_idx += 1
            active_ids = set()
            for track in tracks or []:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[5])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if track_id not in pizza_tracks:
                    pizza_tracks[track_id] = []
                pizza_tracks[track_id].append((cx, cy))
                active_ids.add(track_id)

                # --- Count if first detection is inside the polygon ---
                if (
                    count_polygon and
                    len(pizza_tracks[track_id]) == 1 and
                    track_id not in counted_ids and
                    point_in_polygon((cx, cy), count_polygon)
                ):
                    # Proximity check: skip if close to a recently lost track
                    skip = False
                    for lost in recently_lost:
                        if abs(lost['cx'] - cx) < 40 and abs(lost['cy'] - cy) < 40 and (frame_idx - lost['frame']) < 20:
                            skip = True
                            break
                    if not skip:
                        pizza_count += 1
                        counted_ids.add(track_id)
                        sale_events.append({
                            "frame": frame_idx,
                            "pizza_id": track_id,
                            "cx": cx,
                            "cy": cy
                        })

                # --- Count if crossing into the polygon ---
                if (
                    count_polygon and
                    len(pizza_tracks[track_id]) >= 2 and
                    track_id not in counted_ids
                ):
                    prev = pizza_tracks[track_id][-2]
                    curr = pizza_tracks[track_id][-1]
                    if not point_in_polygon(prev, count_polygon) and point_in_polygon(curr, count_polygon):
                        pizza_count += 1
                        counted_ids.add(track_id)
                        sale_events.append({
                            "frame": frame_idx,
                            "pizza_id": track_id,
                            "cx": curr[0],
                            "cy": curr[1]
                        })

                last_positions[track_id] = (cx, cy, frame_idx)

                color = (0, 165, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"pizza ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

            # --- Proximity check: update recently lost tracks ---
            lost_ids = set(last_positions.keys()) - active_ids
            for lost_id in list(lost_ids):
                cx, cy, lost_frame = last_positions[lost_id]
                recently_lost.append({'cx': cx, 'cy': cy, 'frame': lost_frame})
                # Keep only recent lost tracks (last 30 frames)
                recently_lost = [l for l in recently_lost if frame_idx - l['frame'] < 30]
                del last_positions[lost_id]

            # Draw counting polygon and count
            if count_polygon:
                draw_polygon(frame, count_polygon, color=(0,0,255), thickness=2)
            cv2.putText(frame, f"Pizzas Sold: {pizza_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            out.write(frame)
    finally:
        out.release()
        cap.release()
        # Save sale events to CSV even if interrupted
        csv_path = output_path.replace(".mp4", "_sales.csv")
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = ["frame", "pizza_id", "cx", "cy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for event in sale_events:
                writer.writerow(event)
        print(f"Counting video saved to: {output_path}")
        print(f"Total pizzas sold: {pizza_count}")
        print(f"Pizza sale events saved to: {csv_path}")

if __name__ == "__main__":
    video_path = "data/raw_videos/cut_video_test/1464_CH02_20250607180000_190000 - Trim.mp4"
    camera_key = get_camera_key(video_path)
    # Use "count_polygon" or fallback to "count_box" for backward compatibility
    zone = CAMERA_ZONES[camera_key]
    count_polygon = zone.get("count_polygon") or zone.get("count_box")

    track_and_count_pizzas(
        video_path=video_path,
        output_path=f"data/results/counted_{camera_key}.mp4",
        conf_thres=0.5,
        count_polygon=count_polygon
    )

