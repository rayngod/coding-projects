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
    # --- Open the video ---
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
        
    return cap

def loadFrames(cap):
    # --- Get total frame count ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames: {total_frames}")

    output_folder = 'frames'       # Folder to save frames

    # --- Clear the output folder if it exists ---
    if os.path.exists(output_folder):
        # Remove all contents inside the folder
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove folder recursively
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    # --- Read and save frames ---
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

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
        

    #get frames
    t1 = math.floor(time1 * fps)
    t2 = math.floor(time2 * fps)

    duration = (t2 - t1) / fps
    print(f"Frame {t1} to {t2}")

    return (t1, t2, duration)

def process(t1, t2):
    #run both frames through depthanything
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
    # Load the frame image to visually locate the vanishing point
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        print(f"Error: Could not load image {img_path} for vanishing point estimation.")
        return (0, 0)

    # Keep a working copy that we can overwrite to erase old circles
    img = img_orig.copy()
    vp_coordinates = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal img
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Vanishing Point updated to pixel: ({x}, {y})")
            
            # Reset the coordinate list so only the LAST click is saved
            vp_coordinates.clear()
            vp_coordinates.append((x, y))

            # Reset the image back to original clean copy to erase previous dots
            img = img_orig.copy()

            # Draw a green dot at the *newest* vanishing point location
            cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
            cv2.imshow('Select Vanishing Point (Click & Press Any Key)', img)

    print("\n--> Action Required: Click on the image where parallel ground lines converge (Vanishing Point), then press any key to confirm.")
    cv2.namedWindow('Select Vanishing Point (Click & Press Any Key)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Select Vanishing Point (Click & Press Any Key)', mouse_callback)
    cv2.imshow('Select Vanishing Point (Click & Press Any Key)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(vp_coordinates) > 0:
        return vp_coordinates[0]
    else:
        # Fallback to visual center if user doesn't click anything
        image_height, image_width, _ = img_orig.shape
        print(f"No selection recorded. Falling back to image center: ({image_width / 2}, {image_height / 2})")
        return (image_width / 2, image_height / 2)

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

    # Constants
    MIN_AREA = 50
    position_data = []

    # Expand bounding boxes
    def expand_box(xyxy, img_shape):
        x1, y1, x2, y2 = xyxy
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = (x2 - x1) * 5
        h = (y2 - y1) * 3
        new_x1 = cx - w / 2
        new_y1 = cy - h / 2
        new_x2 = cx + w / 2
        new_y2 = cy + h / 2
        return [new_x1, new_y1, new_x2, new_y2]

    # Callback per frame
    def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
        results = model(frame, imgsz=2140)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter small objects
        detections = detections[detections.area > MIN_AREA]

        if len(detections) == 0:
            return frame 
    
        # Expand boxes
        expanded_boxes = [
            expand_box(box, frame.shape)
            for box in detections.xyxy
        ]
        detections.xyxy = np.array(expanded_boxes)

        # Track
        detections = tracker.update_with_detections(detections)

        # Labels
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
            y_center = (y1 + y2) / 2
            width = x2 - x1
            position_data.append([frame_index, tracker_id, float(x_center), float(y_center), float(width)])

        return annotated

    # Process video
    sv.process_video(
        source_path=video_path,
        target_path="output/pp.mp4",
        callback=callback
    )

    # Save CSV
    df = pd.DataFrame(position_data, columns=["frame", "tracker_id", "x_center", "y_center", "width"])
    df.to_csv("output/positions.csv", index=False)

def calculate(t1, t2, f1, f2, vpx, vpy, depth_scale, duration):
    # Load CSV
    df = pd.read_csv("output/positions.csv")

    # Get target tracker ID from user
    target_id = int(input("Target ID: "))

    width_scale = float(input("Width of Object(Meters): "))

    # Filter for the target ID only
    filtered = df[df["tracker_id"] == target_id]

    # Get total number of frames in the dataset
    max_frame = df["frame"].max()

    # Initialize with -1 for all frames
    x = [-1] * (max_frame + 1)
    y = [-1] * (max_frame + 1)

    # Fill in x and y values for frames where the target was detected
    pix_width = [0] * (max_frame + 1)
    for _, row in filtered.iterrows():
        frame = int(row["frame"])
        x[frame] = row["x_center"]
        y[frame] = row["y_center"]
        pix_width[frame] = row["width"]
        
    x1 = int(x[t1])
    y1 = int(y[t1])
    x2 = int(x[t2])
    y2 = int(y[t2])
    
    if(x1 == -1 or y1 == -1 or x2 == -1 or y2 == -1):
        print("Unable to detect object")
    else:
        width_scale = width_scale / (pix_width[t2] / 5)
        depth_meters = (f1[y1][x1] - f2[y2][x2]) * depth_scale
        m = (vpy - y1) / (vpx - x1)  # slope of the line
        x_projected = ((y2 - y1) / m) + x1
        width_meters = (x2 - x_projected) * width_scale
        dist = math.sqrt(depth_meters ** 2 + width_meters ** 2)
        
        print(f"Depth component (meters): {depth_meters}")
        print(f"Raw Depth outputs (F1 vs F2): {f1[y1][x1]}, {f2[y2][x2]}")
        print(f"X coordinates (Actual vs Projected): {x2}, {x_projected}")
        print(f"Vanishing Point Used: ({vpx}, {vpy})")
        print(f"Normalized tracking box pixel width: {pix_width[t2] / 5}")
        
        MPS_TO_MPH = 2.23694
        print(f"velocity vector(mph): <{width_meters/duration * MPS_TO_MPH},{depth_meters/duration * MPS_TO_MPH}>")
        print("speed(mph): ", dist/duration * MPS_TO_MPH)

def get_depth_scale(img_path, depth):             
    img = cv2.imread(img_path)
    clicked_points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            depth_value = depth[y, x]
            print(f"Clicked at pixel: ({x}, {y}), Depth: {depth_value:.4f} meters")
            clicked_points.append((x, y, depth_value))

            # Draw a red dot on the clicked location
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Image Calibration (Select Reference Ends)', img)

            if len(clicked_points) == 2:
                cv2.destroyAllWindows()

    print("\n--> Action Required: Click on the two endpoints of your known reference object, then press any key to confirm.")
    cv2.namedWindow('Image Calibration (Select Reference Ends)', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Image Calibration (Select Reference Ends)', mouse_callback)
    cv2.imshow('Image Calibration (Select Reference Ends)', img)
    cv2.waitKey(0)
    
    known_real_length = float(input("Real Life Depth of Reference Object (Meters): "))
    # Extract point info
    x1, y1, d1 = clicked_points[0]
    x2, y2, d2 = clicked_points[1]

    # Calculate scale factor
    scale_factor = known_real_length / (d2 - d1)
    return scale_factor

# --- Main Runtime Sequence ---
if __name__ == "__main__":
    video_path = input("Video Path: ")

    # Step 1: Detect and build position tracking matrix
    objectDetection(video_path)

    # Step 2: Unpack frames
    cap = getVideo(video_path)
    loadFrames(cap)

    # Step 3: Calibrate Projective Constraint (Manual Vanishing Point)
    vpx, vpy = getVP("frames/frame_0.jpg")

    while(True):
        tup = getTime(cap)
        t1 = tup[0]
        t2 = tup[1]
        duration = tup[2]

        # Step 4: Extract MDE map tensors 
        process(t1, t2)
        f1, f2 = load(t1, t2)

        # Step 5: Metric Calibration Constraint (Manual Reference Scaling)
        depth_scale = get_depth_scale(f"frames/frame_{t2}.jpg", f1)
        
        # Step 6: Compute final vector kinematics
        print("Depth Scale:",depth_scale)
        calculate(t1, t2, f1, f2, vpx, vpy, depth_scale, duration)
        
        cont = input("\nRun another calculation loop? (y/n): ")
        if cont.lower() != 'y':
            break
