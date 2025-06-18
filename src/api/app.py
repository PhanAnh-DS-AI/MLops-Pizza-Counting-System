import sys
import os
import threading
import json
import cv2
from fastapi import FastAPI, Request
from starlette.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
import uvicorn

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.detection.counter import track_and_count_pizzas, get_camera_key, draw_polygon
from src.detection.tracking import pizza_tracker
from src.config.camera_zones import CAMERA_ZONES

app = FastAPI(title="Pizza Sales Counting System")

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def abs_path(relative_path):
    return os.path.abspath(os.path.join(project_root, relative_path))

feedback_dir = abs_path("data/feedback")
os.makedirs(feedback_dir, exist_ok=True)

process_threads = {}
stop_flags = {}
stream_stop_flags = {}

class ProcessRequest(BaseModel):
    video_path: str

@app.post("/process")
async def process_video(req: ProcessRequest):
    video_path = abs_path(req.video_path)
    if not os.path.exists(video_path):
        return {"error": f"Video file not found: {video_path}"}
    camera_key = get_camera_key(video_path)
    zone = CAMERA_ZONES[camera_key]
    count_polygon = zone.get("count_polygon") or zone.get("count_box")
    output_path = abs_path(f"data/results/counted_{camera_key}.mp4")
    stop_flags[camera_key] = False

    def run_process():
        track_and_count_pizzas(
            video_path=video_path,
            output_path=output_path,
            conf_thres=0.5,
            count_polygon=count_polygon,
            stop_flag=lambda: stop_flags[camera_key]
        )

    t = threading.Thread(target=run_process)
    t.start()
    process_threads[camera_key] = t

    return {
        "status": "processing",
        "video_id": camera_key,
        "output_video": output_path,
        "output_csv": output_path.replace(".mp4", "_sales.csv")
    }

@app.post("/stop/{video_id}")
def stop_process(video_id: str):
    stop_flags[video_id] = True
    stream_stop_flags[video_id] = True
    return {"status": "stopping"}

@app.get("/stream/{video_name}")
def stream_video(video_name: str):
    video_path = abs_path(f"data/raw_videos/cut_video_test/{video_name}")
    camera_key = get_camera_key(video_path)
    zone = CAMERA_ZONES[camera_key]
    count_polygon = zone.get("count_polygon") or zone.get("count_box")

    def gen():
        frame_count = 0
        try:
            for frame, tracks in pizza_tracker(video_path, conf_thres=0.5):
                frame_count += 1
                if frame_count % 45 == 0:
                    continue  
                if frame is None:
                    break
                # Draw polygon overlay
                if count_polygon:
                    draw_polygon(frame, count_polygon, color=(0, 0, 255), thickness=2)
                # Draw tracks and center points
                for track in tracks or []:
                    x1, y1, x2, y2 = map(int, track[:4])
                    track_id = int(track[5])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
                # Encode frame to JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                try:
                    _, jpeg = cv2.imencode('.jpg', frame, encode_param)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                except Exception as e:
                    print("Client disconnected or yield error:", e)
                    break
        except GeneratorExit:
            print("Stream generator exit (client disconnected)")
        except Exception as e:
            print("Stream error:", e)
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.post("/feedback")
async def receive_feedback(request: Request):
    data = await request.json()
    video_id = data.get("video_id")
    feedback_path = os.path.join(feedback_dir, f"{video_id}_feedback.json")
    if os.path.exists(feedback_path):
        with open(feedback_path, "r") as f:
            feedbacks = json.load(f)
    else:
        feedbacks = []
    feedbacks.append(data)
    with open(feedback_path, "w") as f:
        json.dump(feedbacks, f, indent=2)
    return {"status": "received"}

@app.get("/results/{video_id}")
def get_results(video_id: str):
    csv_path = abs_path(f"data/results/counted_{video_id}_sales.csv")
    if not os.path.exists(csv_path):
        return {"error": "Result not found"}
    return FileResponse(csv_path, media_type="text/csv", filename=os.path.basename(csv_path))
    
@app.get("/video/{video_id}")
def get_video(video_id: str):
    video_path = abs_path(f"data/results/counted_{video_id}.mp4")
    if not os.path.exists(video_path):
        return {"error": "Video not found"}
    return FileResponse(video_path, media_type="video/mp4", filename=os.path.basename(video_path))

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
