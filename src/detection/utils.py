import cv2
import numpy as np

# =================== HELPER FUNCTIONS ===================
def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    limg = cv2.merge((cl, a, b))
    
    final_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final_image

# =================== DRAWING FUNCTIONS ===================

def draw_box_on_frame(video_path, display_scale=0.5):
    """
    Allows user to draw a rectangle on the first frame of the video.
    Returns the coordinates as a dict: {'x1': int, 'y1': int, 'x2': int, 'y2': int}
    display_scale: float, scale for display window (default 0.5)
    """
    drawing = False
    ix, iy = -1, -1
    box = {}

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video.")
        return None

    # Resize for display
    h, w = frame.shape[:2]
    disp_w, disp_h = int(w * display_scale), int(h * display_scale)
    disp_frame = cv2.resize(frame, (disp_w, disp_h))
    frame_copy = disp_frame.copy()

    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing, ix, iy, box, frame_copy

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            box['x1'], box['y1'] = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                frame_copy = disp_frame.copy()
                cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 0, 255), 2)
                cv2.imshow('Draw Box', frame_copy)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            box['x2'], box['y2'] = x, y
            cv2.rectangle(disp_frame, (ix, iy), (x, y), (0, 0, 255), 2)
            cv2.imshow('Draw Box', disp_frame)
            print(f"Box coordinates (display): x1={box['x1']}, y1={box['y1']}, x2={box['x2']}, y2={box['y2']}")

    cv2.namedWindow('Draw Box')
    cv2.setMouseCallback('Draw Box', draw_rectangle)

    while True:
        cv2.imshow('Draw Box', frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break

    cv2.destroyAllWindows()
    if all(k in box for k in ('x1', 'y1', 'x2', 'y2')):
        # Convert display coordinates back to original image scale
        x1, x2 = sorted([box['x1'], box['x2']])
        y1, y2 = sorted([box['y1'], box['y2']])
        x1 = int(x1 / display_scale)
        x2 = int(x2 / display_scale)
        y1 = int(y1 / display_scale)
        y2 = int(y2 / display_scale)
        print(f"Box coordinates (original): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    else:
        print("Box not drawn")
        return None

def draw_polygon_on_frame(video_path, display_scale=0.5):
    """
    Allows user to click 4 points on the first frame of the video to define a quadrilateral (custom box).
    Returns the coordinates as a dict: {'x1': int, 'y1': int, 'x2': int, 'y2': int, ...} in original image scale.
    """
    points = []

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video.")
        return None

    # Resize for display
    h, w = frame.shape[:2]
    disp_w, disp_h = int(w * display_scale), int(h * display_scale)
    disp_frame = cv2.resize(frame, (disp_w, disp_h))
    frame_copy = disp_frame.copy()

    def click_event(event, x, y, flags, param):
        nonlocal points, frame_copy
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(frame_copy, points[-2], points[-1], (255, 0, 0), 2)
            cv2.imshow('Draw Polygon', frame_copy)
            if len(points) == 4:
                cv2.line(frame_copy, points[-1], points[0], (255, 0, 0), 2)
                cv2.imshow('Draw Polygon', frame_copy)
                print("Polygon points (display):", points)

    cv2.namedWindow('Draw Polygon')
    cv2.setMouseCallback('Draw Polygon', click_event)

    while True:
        cv2.imshow('Draw Polygon', frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or len(points) == 4:  # ESC or 4 points
            break

    cv2.destroyAllWindows()
    if len(points) == 4:
        # Convert display coordinates back to original image scale
        coords = {}
        for idx, (x, y) in enumerate(points, 1):
            coords[f'x{idx}'] = int(x / display_scale)
            coords[f'y{idx}'] = int(y / display_scale)
        print("Polygon points (original):", coords)
        return coords
    else:
        print("Polygon not completed.")
        return None

def point_in_polygon(pt, polygon_dict):
    pts = np.array([
        [polygon_dict['x1'], polygon_dict['y1']],
        [polygon_dict['x2'], polygon_dict['y2']],
        [polygon_dict['x3'], polygon_dict['y3']],
        [polygon_dict['x4'], polygon_dict['y4']]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, pt, False) >= 0

def draw_polygon(frame, polygon_dict, color=(0,0,255), thickness=2):
    pts = np.array([
        [polygon_dict['x1'], polygon_dict['y1']],
        [polygon_dict['x2'], polygon_dict['y2']],
        [polygon_dict['x3'], polygon_dict['y3']],
        [polygon_dict['x4'], polygon_dict['y4']]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


if __name__ == "__main__":
    video_path = "data/raw_videos/1464_CH02_20250607180000_190000.mp4"
    draw_polygon_on_frame(video_path)

