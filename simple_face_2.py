# correct_face_detection.py
import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# ------------ Configuration ------------
MODEL_PATH = os.path.expanduser("~/.insightface/models/light/scrfd_500m_bnkps.onnx")
CAMERA_WIDTH = 1640
CAMERA_HEIGHT = 1232
DETECTION_THRESHOLD = 0.5  # Higher threshold = fewer false positives

print("Using model:", MODEL_PATH)
assert os.path.exists(MODEL_PATH), "Model not found!"

# ------------ Initialize Model ------------
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2
session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

print("Model input:", input_name)
print("Model outputs:", [output.name for output in session.get_outputs()])

# ------------ Initialize Camera ------------
pipeline = (
    f"libcamerasrc ! "
    f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate=30/1 ! "
    f"videoconvert ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    # Try simpler pipeline
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open camera")
        exit(1)

print(f"Camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
print(f"Detection threshold: {DETECTION_THRESHOLD}")
print("\nPress 'q' to quit")
print("Press 's' to save current frame")
print("Press '+' to increase threshold")
print("Press '-' to decrease threshold")

# ------------ Preprocessing ------------
def preprocess_image(img, input_size=(640, 640)):
    """Preprocess image for SCRFD"""
    img_resized = cv2.resize(img, input_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - 127.5) / 128.0  # SCRFD normalization
    img_resized = img_resized.transpose(2, 0, 1)  # HWC to CHW
    return np.expand_dims(img_resized, axis=0)

# ------------ CORRECTED Detection Function ------------
def decode_predictions(outputs, img_shape, threshold=0.5):
    """
    Properly decode SCRFD predictions
    Returns list of [x1, y1, x2, y2, score]
    """
    h, w = img_shape[:2]
    input_size = 640
    scale_x = w / input_size
    scale_y = h / input_size
    
    # SCRFD has 3 feature maps with strides [8, 16, 32]
    # Outputs: [score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32, bbox_32, kps_32]
    
    all_detections = []
    
    # Process each feature map
    for i in range(0, len(outputs), 3):
        if i+1 >= len(outputs):  # Ensure we have both score and bbox
            continue
            
        scores = outputs[i][0]  # Remove batch dimension
        bboxes = outputs[i+1][0]
        
        # Get indices where score > threshold
        high_score_indices = np.where(scores > threshold)
        
        if len(high_score_indices[0]) == 0:
            continue
            
        # For each high score
        for idx in range(len(high_score_indices[0])):
            score = scores[high_score_indices[0][idx], high_score_indices[1][idx]]
            
            # Get corresponding bbox (l, t, r, b format)
            bbox = bboxes[high_score_indices[0][idx], high_score_indices[1][idx], :]
            
            # The bbox represents distances from anchor center
            # We need anchor position - simplified calculation
            stride = 8 if i == 0 else 16 if i == 3 else 32
            grid_x = high_score_indices[1][idx] % (input_size // stride)
            grid_y = high_score_indices[0][idx] % (input_size // stride)
            
            anchor_center_x = (grid_x + 0.5) * stride
            anchor_center_y = (grid_y + 0.5) * stride
            
            # Decode bbox
            x1 = anchor_center_x - bbox[0]
            y1 = anchor_center_y - bbox[1]
            x2 = anchor_center_x + bbox[2]
            y2 = anchor_center_y + bbox[3]
            
            # Scale to original image
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            # Clip to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # Ensure valid box size
            if (x2 - x1) > 20 and (y2 - y1) > 20:
                all_detections.append([x1, y1, x2, y2, float(score)])
    
    return all_detections

def non_max_suppression(detections, iou_threshold=0.3):
    """Remove overlapping boxes"""
    if not detections:
        return []
    
    # Convert to numpy array
    boxes = np.array([d[:4] for d in detections])
    scores = np.array([d[4] for d in detections])
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
            
        # Calculate IoU with remaining boxes
        box_i = boxes[i]
        other_boxes = boxes[indices[1:]]
        
        # Intersection
        x1 = np.maximum(box_i[0], other_boxes[:, 0])
        y1 = np.maximum(box_i[1], other_boxes[:, 1])
        x2 = np.minimum(box_i[2], other_boxes[:, 2])
        y2 = np.minimum(box_i[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Areas
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        
        union = area_i + area_others - intersection
        iou = intersection / union
        
        # Keep boxes with IoU < threshold
        indices = indices[1:][iou < iou_threshold]
    
    return [detections[i] for i in keep]

# ------------ Main Loop ------------
frame_count = 0
save_count = 0
current_threshold = DETECTION_THRESHOLD

# For FPS calculation
fps_start_time = time.time()
fps_frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        fps_frame_count += 1
        display_frame = frame.copy()
        
        # Process every 3rd frame for better performance
        if frame_count % 3 == 0:
            try:
                # Run detection
                input_data = preprocess_image(frame)
                outputs = session.run(None, {input_name: input_data})
                
                # Decode predictions
                raw_detections = decode_predictions(outputs, frame.shape, current_threshold)
                
                # Apply NMS to remove overlapping boxes
                detections = non_max_suppression(raw_detections)
                
                if detections:
                    status = f"Faces: {len(detections)}"
                    
                    # Draw bounding boxes
                    for i, (x1, y1, x2, y2, score) in enumerate(detections):
                        # Color based on score
                        green = int(255 * min(1.0, score))
                        color = (0, green, 255 - green)
                        
                        # Draw box
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw score with background
                        score_text = f"{score:.3f}"
                        (text_w, text_h), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(display_frame, 
                                    (x1, y1 - text_h - 10), 
                                    (x1 + text_w, y1), 
                                    color, -1)
                        cv2.putText(display_frame, score_text, 
                                  (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Draw face number
                        cv2.putText(display_frame, f"Face {i+1}", 
                                  (x1, y2 + 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Draw center dot
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        cv2.circle(display_frame, (center_x, center_y), 5, (255, 255, 0), -1)
                    
                    # Print to console (only when detection changes)
                    if frame_count % 30 == 0:
                        print(f"\r{status} (thresh: {current_threshold:.2f})", end="")
                else:
                    status = "No faces"
                    if frame_count % 30 == 0:
                        print(f"\r{status} (thresh: {current_threshold:.2f})", end="")
                    
            except Exception as e:
                status = f"Error: {str(e)[:30]}"
                print(f"\nDetection error: {e}")
        
        # Calculate and display FPS
        if fps_frame_count >= 30:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
            fps_frame_count = 0
        else:
            fps = 0
        
        # Draw info on frame
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Thresh: {current_threshold:.2f}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, "Q:Quit  S:Save  +/-:Adjust thresh", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Face Detection", display_frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            filename = f"detection_{save_count:03d}.jpg"
            cv2.imwrite(filename, display_frame)
            print(f"\nSaved: {filename}")
            save_count += 1
        elif key == ord('+') or key == ord('='):
            current_threshold = min(0.95, current_threshold + 0.05)
            print(f"\nThreshold increased to: {current_threshold:.2f}")
        elif key == ord('-') or key == ord('_'):
            current_threshold = max(0.1, current_threshold - 0.05)
            print(f"\nThreshold decreased to: {current_threshold:.2f}")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera released. Goodbye!")
