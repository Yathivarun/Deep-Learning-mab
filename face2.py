import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# ---------------- Paths ----------------
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE = os.path.join(BASE, "glintr100.onnx")

print("Detector :", SCRFD)
print("Recognizer:", ARCFACE)
assert os.path.exists(SCRFD), "SCRFD model not found"
assert os.path.exists(ARCFACE), "GlintR100 model not found"

# ---------------- ONNX Sessions ----------------
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2

det_sess = ort.InferenceSession(SCRFD, opts, providers=["CPUExecutionProvider"])
rec_sess = ort.InferenceSession(ARCFACE, opts, providers=["CPUExecutionProvider"])

det_input = det_sess.get_inputs()[0].name
rec_input = rec_sess.get_inputs()[0].name

# Print model output info for debugging
print("\n=== SCRFD Outputs ===")
for i, output in enumerate(det_sess.get_outputs()):
    print(f"Output {i}: {output.name}, shape: {output.shape}")

# ---------------- Camera (IMX Full FOV) ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

# ---------------- Preprocess ----------------
def preprocess_det(img):
    """Preprocess image for SCRFD detection"""
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (640, 640))
    img_resized = img_resized.astype(np.float32)
    # Normalize (SCRFD typically uses mean=127.5, std=128.0)
    img_resized = (img_resized - 127.5) / 128.0
    img_resized = img_resized.transpose(2, 0, 1)  # HWC to CHW
    return np.expand_dims(img_resized, axis=0)  # Add batch dimension

def preprocess_rec(face):
    """Preprocess face for ArcFace recognition"""
    if face.size == 0:
        return None
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    # ArcFace normalization
    face = (face - 127.5) / 128.0
    face = face.transpose(2, 0, 1)  # HWC to CHW
    return np.expand_dims(face, axis=0)  # Add batch dimension

# ---------------- SCRFD Proper Decode ----------------
def scrfd_postprocess(outputs, orig_shape, input_size=640, thresh=0.5):
    """
    SCRFD outputs multiple feature maps with scores and bboxes.
    This function decodes them properly.
    """
    h, w = orig_shape[:2]
    scale_x = w / input_size
    scale_y = h / input_size
    
    all_boxes = []
    all_scores = []
    
    # SCRFD typically outputs 3 feature levels
    # Each level has: scores (batch, num_anchors, 1) and bboxes (batch, num_anchors, 4)
    num_levels = len(outputs) // 2
    
    for i in range(num_levels):
        try:
            scores_idx = i * 2
            boxes_idx = i * 2 + 1
            
            if scores_idx >= len(outputs) or boxes_idx >= len(outputs):
                break
                
            scores = outputs[scores_idx]
            boxes = outputs[boxes_idx]
            
            # Handle different output shapes
            if len(scores.shape) == 3:
                scores = scores[0]  # Remove batch dimension
            if len(boxes.shape) == 3:
                boxes = boxes[0]  # Remove batch dimension
            
            # Flatten if needed
            if len(scores.shape) > 1:
                scores = scores.reshape(-1)
            if len(boxes.shape) > 1 and boxes.shape[-1] == 4:
                boxes = boxes.reshape(-1, 4)
            
            # Filter by threshold
            valid_mask = scores > thresh
            if not np.any(valid_mask):
                continue
                
            valid_scores = scores[valid_mask]
            valid_boxes = boxes[valid_mask]
            
            all_scores.extend(valid_scores)
            all_boxes.extend(valid_boxes)
            
        except Exception as e:
            print(f"Warning: Error processing level {i}: {e}")
            continue
    
    if len(all_boxes) == 0:
        return None
    
    # Find best detection
    all_scores = np.array(all_scores)
    all_boxes = np.array(all_boxes)
    best_idx = np.argmax(all_scores)
    best_box = all_boxes[best_idx]
    best_score = all_scores[best_idx]
    
    # Scale back to original image size
    x1, y1, x2, y2 = best_box
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    
    # Clamp to image boundaries
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    # Ensure valid box
    if x2 <= x1 or y2 <= y1:
        return None
    
    return np.array([x1, y1, x2, y2]), best_score

# ---------------- Main Loop ----------------
frame_id = 0
print("\n=== Starting face detection loop ===")
print("Press ESC to exit\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Skip frames for Pi speed
        frame_id += 1
        if frame_id % 2 != 0:
            cv2.imshow("Sensor", frame)
            if cv2.waitKey(1) == 27:
                break
            continue
        
        t0 = time.time()
        
        # Run detection
        try:
            det_input_data = preprocess_det(frame)
            det_out = det_sess.run(None, {det_input: det_input_data})
            
            # Debug: print output shapes on first frame
            if frame_id == 2:
                print("Detection output shapes:")
                for i, out in enumerate(det_out):
                    print(f"  Output {i}: {out.shape}")
            
            result = scrfd_postprocess(det_out, frame.shape)
            
            if result is not None:
                box, score = result
                x1, y1, x2, y2 = box
                
                # Extract face region
                face = frame[y1:y2, x1:x2]
                
                if face.size > 0:
                    # Run recognition
                    face_input = preprocess_rec(face)
                    if face_input is not None:
                        emb = rec_sess.run(None, {rec_input: face_input})[0]
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw score
                        label = f"Face: {score:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        if frame_id % 30 == 0:  # Print every 30 frames
                            print(f"Face detected! Score: {score:.3f}, Embedding shape: {emb.shape}")
            
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
        
        # Calculate and display FPS
        fps = 1.0 / (time.time() - t0 + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Sensor", frame)
        
        if cv2.waitKey(1) == 27:  # ESC key
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Cleanup complete")
