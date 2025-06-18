from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm
from detection.utils import apply_clahe

def extract_and_classify_frames(video_path, output_dir, frame_interval=60, conf_thres=0.5, model_path="models/yolov8m.pt", show_preview=True):
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure your model weights (e.g., yolov8m.pt) are in the 'models/' directory.")
        return

    all_dir = os.path.join(output_dir, "all_frames")
    pizza_dir = os.path.join(output_dir, "contains_pizza")
    no_pizza_dir = os.path.join(output_dir, "no_pizza")
    os.makedirs(all_dir, exist_ok=True)
    os.makedirs(pizza_dir, exist_ok=True)
    os.makedirs(no_pizza_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from path '{video_path}'")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps}")

    frame_count, saved_count = 0, 0
    pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")

    TARGET_DISPLAY_CLASSES = ["pizza", "person"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{frame_count:05d}.jpg"
            
            processed_frame = apply_clahe(frame)

            frame_with_detections = processed_frame.copy() 

            results = model(processed_frame, verbose=False)[0] 

            has_pizza_for_dir_classification = False 

            for box in results.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.model.names[cls_id]

                if label == "pizza" and conf >= conf_thres:
                    has_pizza_for_dir_classification = True

                if label in TARGET_DISPLAY_CLASSES and conf >= conf_thres:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    
                    color = (0, 255, 0) 
                    if label == "person":
                        color = (255, 0, 0) 
                    elif label == "pizza":
                        color = (0, 165, 255) 
                        
                    cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_with_detections, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(all_dir, frame_name), frame) # Lưu frame gốc vào all_dir
            
            save_path = os.path.join(pizza_dir if has_pizza_for_dir_classification else no_pizza_dir, frame_name)
            cv2.imwrite(save_path, frame_with_detections) # Lưu frame đã xử lý và vẽ box
            saved_count += 1

            if show_preview:
                scale = 0.5  
                h, w = frame_with_detections.shape[:2]
                display_frame = cv2.resize(frame_with_detections, (int(w * scale), int(h * scale)))
                cv2.imshow("Detection Preview (Pizza & Person)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
    print(f"✅ Done. Saved {saved_count} frames to '{output_dir}' (preview mode = {show_preview}).")


if __name__ == "__main__":
    extract_and_classify_frames(
        video_path="data/raw_videos/1461_CH01_20250607193711_203711.mp4",  
        output_dir="data/frames/train_video_1461_CH01", 
        frame_interval=60,
        show_preview=True,
        model_path="models/yolov8l.pt"
    )