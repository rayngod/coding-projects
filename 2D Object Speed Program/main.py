import numpy as np
import math
import subprocess
import cv2
import os
import shutil
import pandas as pd
from ultralytics import YOLO
import supervision as sv

def getVideo(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
        
    return cap

def loadFrames(cap):
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    output_folder = 'frames'

    if os.path.exists(output_folder):
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(output_folder, exist_ok=True)

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = os.path.join(output_folder, f"frame_{frame_num}.jpg")
        cv2.imwrite(filename, frame)
        frame_num += 1

    print(f"Saved {frame_num} frames to '{output_folder}'")

def getTime(cap):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_time = frame_count / fps
    print("Total Time: ", total_time)
    # get time frames
    time1 = -1
    time2 = -1
    while(not(0 <= time1 <= time2 < total_time)):
        time1 = float(input("Enter time 1: "))
        time2 = float(input("Enter time 2: "))
        

    #get frames numbers
    t1 = math.floor(time1 * fps)
    t2 = math.floor(time2 * fps)

    duration = (t2 - t1) / fps
    print(f"Frame {t1} to {t2}")

    return (t1, t2, duration)

def process(t1, t2):
    #run both frames through DepthAnything v2
    subprocess.run([
    "python", "run.py",
    "--encoder", "vitl",
    "--load-from", "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth",
    "--max-depth", "80",
    "--img-path", f"frames/frame_{t1}.jpg",
    "--outdir", "output",
    "--save-numpy"
    ])

    subprocess.run([
    "python", "run.py",
    "--encoder", "vitl",
    "--load-from", "checkpoints/depth_anything_v2_metric_vkitti_vitl.pth",
    "--max-depth", "80",
    "--img-path", f"frames/frame_{t2}.jpg",
    "--outdir", "output",
    "--save-numpy"
    ])

def load(t1, t2):
    # Load the .npy file
    f1 = np.load(f'output/frame_{t1}_raw_depth_meter.npy')
    f2 = np.load(f'output/frame_{t2}_raw_depth_meter.npy')
    return (f1, f2)

def getVP(img_path):
    # Fix the vanishing point horizontally at the image center; user selects its height.
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print(f"Error: Could not load image {img_path} for vanishing point estimation.")
        return (0, 0)

    image_height, image_width, _ = img_orig.shape
    center_x = image_width // 2
    img = img_orig.copy()
    vp_coordinates = []
    window_name = 'Select Vanishing Point Height (Click & Press Any Key)'

    def mouse_callback(event, x, y, flags, param):
        nonlocal img
        if event == cv2.EVENT_LBUTTONDOWN:
            vp_coordinates.clear()
            vp_coordinates.append((center_x, y))
            img = img_orig.copy()
            cv2.line(img, (0, y), (image_width - 1, y), (0, 255, 0), 2)
            cv2.circle(img, (center_x, y), 6, (0, 255, 0), -1)
            print(f"Vanishing Point height selected: y={y}")
            cv2.imshow(window_name, img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, img)
    while not vp_coordinates:
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return vp_coordinates[0]



def objectDetection(video_path):
    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")

    # Set up the tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=0.05,
        minimum_matching_threshold=0.95,
        lost_track_buffer=60,
        frame_rate=30,
    )

    # Annotators
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    position_data = []

    # Expand bounding boxes
    def expand_box(xyxy):
        x1, y1, x2, y2 = xyxy
        w = (x2 - x1)
        h = (y2 - y1)
        return [x1 - 2 * w, y1 - h, x2 + 2 * w, y1 + h]

    # Callback per frame
    def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
        #run frame through YOLOv8 
        #larger imgsz = more accurate detections, but slower
        results = model(frame, imgsz=2140)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter out tiny objects
        detections = detections[detections.area > 50]

        if len(detections) == 0:
            return frame 

        detections.xyxy = np.array([
            expand_box(box) for box in detections.xyxy
        ])

        detections = tracker.update_with_detections(detections)
        labels = [
            f"#{tid} {results.names[cid]}"
            for cid, tid in zip(detections.class_id, detections.tracker_id)
            if tid is not None
        ]

        # Annotate
        annotated = box_annotator.annotate(frame.copy(), detections=detections)
        annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)
        annotated = trace_annotator.annotate(annotated, detections=detections)

        # Save position data
        for box_xyxy, tracker_id in zip(detections.xyxy, detections.tracker_id):
            if tracker_id is None:
                continue
            x1, y1, x2, y2 = box_xyxy
            x_center = (x1 + x2) / 2
            y_bottom = y2 + (y2 - y1) / 3
            width = x2 - x1
            position_data.append([
                frame_index, tracker_id, int(x_center), int(y_bottom), float(width),
            ])

        return annotated

    sv.process_video(
        source_path=video_path,
        target_path="output/pp.mp4",
        callback=callback
    )

    df = pd.DataFrame(
        position_data,
        columns=["frame", "tracker_id", "x_center", "y_bottom", "width"],
    )
    df.to_csv("output/positions.csv", index=False)

def calculate(t1, t2, f1, f2, vpx, vpy, depth_scale, duration, img_path, target_id):
    df = pd.read_csv("output/positions.csv")
    filtered = df[df["tracker_id"] == target_id]
    max_frame = df["frame"].max()

    # Initialize with -1 for all frames
    x = [-1] * (max_frame + 1)
    y_bottom = [-1] * (max_frame + 1)
    pix_width = [0] * (max_frame + 1)
    
    for _, row in filtered.iterrows():
        frame = int(row["frame"])
        x[frame] = row["x_center"]
        y_bottom[frame] = row["y_bottom"]
        pix_width[frame] = row["width"]
        
    x1 = int(x[t1])
    y1 = int(y_bottom[t1])
    x2 = int(x[t2])
    y2 = int(y_bottom[t2])

    if(x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1):
        print("Unable to detect object")
    else:
        depth_meters = (f2[y2][x2] - f1[y1][x1]) * depth_scale
        if math.isclose(vpx, x1):
            #avoid divide by 0
            x_projected = x1
        else:
            m = (vpy - y1) / (vpx - x1)
            x_projected = ((y2 - y1) / m) + x1

        width_scale = get_width_scale(img_path, vpy, y2)
        width_meters = (x2 - x_projected) * width_scale
        dist = math.sqrt(depth_meters ** 2 + width_meters ** 2)

        MPS_TO_MPH = 2.23694
        print(f"Depth component (meters): {depth_meters}")
        print(f"Width component (meters): {width_meters}")
        print("Raw Depth outputs: "f"{f1[y1][x1]}, "f"{f2[y2][x2]}")
        print(f"X coordinates (Actual vs Projected): {x2}, {x_projected}")
        print(f"Vanishing Point: ({vpx}, {vpy})")
        print(f"Tracking box pixel width: {pix_width[t2] / 5}")
        print(f"velocity vector(mph): <{width_meters/duration * MPS_TO_MPH},{depth_meters/duration * MPS_TO_MPH}>")
        print("speed(mph): ", dist/duration * MPS_TO_MPH)

def get_width_scale(img_path, vpy, target_y):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image {img_path} for width calibration.")
    
    clicked_points = []
    window_name = "Width Calibration (Select Reference Ends)"

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicked_points) >= 2:
            return
        if clicked_points:
            y = clicked_points[0][1]
        clicked_points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(clicked_points) == 1:
            cv2.line(img, (0, y), (img.shape[1] - 1, y), (255, 255, 0), 1)
        else:
            cv2.line(img, clicked_points[0], clicked_points[1], (0, 255, 0), 2)
        cv2.imshow(window_name, img)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, img)
    while len(clicked_points) < 2:
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pixel_width = abs(clicked_points[1][0] - clicked_points[0][0])
    known_width = float(input("Real-world width of reference (meters): "))

    cur_y = (clicked_points[0][1] + clicked_points[1][1]) / 2
    dist = abs(cur_y - vpy)
    target = abs(target_y - vpy)
    return (known_width / pixel_width) * (dist / target)


def get_depth_scale(img_path, depth):             
    img = cv2.imread(img_path)
    clicked_points = []

    def mouse_callback(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or len(clicked_points) >= 2:
            return
        if clicked_points:
            x = clicked_points[0][0]
        depth_value = depth[y, x]
        print(f"Clicked at pixel: ({x}, {y}), Depth: {depth_value:.4f} meters")
        clicked_points.append((x, y, depth_value))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        if len(clicked_points) == 1:
            cv2.line(img, (x, 0), (x, img.shape[0] - 1), (255, 255, 0), 1)
        else:
            cv2.line(img, (x, clicked_points[0][1]), (x, y), (0, 255, 0), 2)
        cv2.imshow('Depth Calibration (Select Reference Ends)', img)

    cv2.namedWindow('Depth Calibration (Select Reference Ends)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Depth Calibration (Select Reference Ends)', mouse_callback)
    cv2.imshow('Depth Calibration (Select Reference Ends)', img)
    while len(clicked_points) < 2:
        cv2.waitKey(1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    known_real_length = float(input("Real World Depth of Reference Object (Meters): "))
    x1, y1, d1 = clicked_points[0]
    x2, y2, d2 = clicked_points[1]

    # Calculate scale factor
    scale_factor = known_real_length / abs(d2 - d1)
    return scale_factor

if __name__ == "__main__":
    video_path = input("Video Path: ")

    # Detect object
    objectDetection(video_path)

    cap = getVideo(video_path)
    loadFrames(cap)

    while(True):
        # get tracked object ID
        target_id = int(input("Target ID: "))

        # get vanishing point
        vpx, vpy = getVP("frames/frame_0.jpg")
        tup = getTime(cap)
        t1 = tup[0]
        t2 = tup[1]
        duration = tup[2]

        # run DepthAnything v2
        process(t1, t2)
        f1, f2 = load(t1, t2)

        depth_scale = get_depth_scale(f"frames/frame_{t2}.jpg", f1)
        print("Depth Scale:",depth_scale)
        calculate(t1, t2, f1, f2, vpx, vpy, depth_scale, duration, f"frames/frame_{t2}.jpg", target_id)
        
        cont = input("\nRun another calculation loop? (y/n): ")
        if cont.lower() != 'y':
            break



















