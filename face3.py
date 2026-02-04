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

# Print model info for debugging
print("\n=== SCRFD Model Info ===")
print(f"Input: {det_sess.get_inputs()[0].name}, shape: {det_sess.get_inputs()[0].shape}")
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
def preprocess_det(img, input_size=(640, 640)):
    """Preprocess image for SCRFD detection"""
    img_resized = cv2.resize(img, input_size)
    img_resized = img_resized.astype(np.float32)
    # Mean and std normalization for SCRFD
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
    return np.expand_dims(face, axis=0)

# ---------------- Distance calculation for anchors ----------------
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def generate_anchors(height, width, stride, num_anchors=2):
    """Generate anchor points for a feature map."""
    anchors = []
    for y in range(height):
        for x in range(width):
            for _ in range(num_anchors):
                anchors.append([x * stride + stride // 2, y * stride + stride // 2])
    return np.array(anchors, dtype=np.float32)

# ---------------- SCRFD Decode with Robust Size Handling ----------------
def scrfd_postprocess(outputs, orig_shape, input_size=640, thresh=0.4, nms_thresh=0.4):
    """
    Properly decode SCRFD outputs with robust size matching.
    SCRFD has 3 detection heads for different scales with strides [8, 16, 32]
    """
    h, w = orig_shape[:2]
    
    # SCRFD strides
    fmc = 3  # number of feature maps
    feat_stride_fpn = [8, 16, 32]
    num_anchors = 2  # anchors per location
    
    all_boxes = []
    all_scores = []
    
    # Parse outputs - for scrfd_500m_bnkps: 9 outputs (3 scales × 3 outputs per scale)
    # Format: [score_8, bbox_8, kps_8, score_16, bbox_16, kps_16, score_32, bbox_32, kps_32]
    outputs_per_scale = len(outputs) // fmc
    
    for idx in range(fmc):
        stride = feat_stride_fpn[idx]
        
        # Calculate feature map size
        fm_height = input_size // stride
        fm_width = input_size // stride
        
        # Get outputs for this scale
        score_idx = idx * outputs_per_scale
        bbox_idx = score_idx + 1
        
        if score_idx >= len(outputs) or bbox_idx >= len(outputs):
            continue
        
        scores = outputs[score_idx]
        bboxes = outputs[bbox_idx]
        
        # Remove batch dimension if present
        if len(scores.shape) == 3:
            scores = scores[0]
        if len(bboxes.shape) == 3:
            bboxes = bboxes[0]
        
        # Now handle the actual shape mismatch
        # scores might be (800,) or (12800,) etc.
        # bboxes might be (800, 4) or (12800, 4) etc.
        
        # Flatten scores if needed
        scores_flat = scores.flatten()
        
        # Reshape bboxes to (N, 4)
        if len(bboxes.shape) == 1:
            # If bboxes is flat, reshape to (N, 4)
            num_boxes = len(bboxes) // 4
            bboxes_reshaped = bboxes.reshape(num_boxes, 4)
        elif len(bboxes.shape) == 2 and bboxes.shape[1] == 4:
            bboxes_reshaped = bboxes
        else:
            # Try to reshape to (N, 4)
            total_elements = bboxes.size
            num_boxes = total_elements // 4
            bboxes_reshaped = bboxes.reshape(num_boxes, 4)
        
        # Now match sizes between scores and boxes
        num_scores = len(scores_flat)
        num_boxes = len(bboxes_reshaped)
        
        # Take the minimum to avoid index errors
        num_valid = min(num_scores, num_boxes)
        
        if num_valid == 0:
            continue
        
        scores_matched = scores_flat[:num_valid]
        bboxes_matched = bboxes_reshaped[:num_valid]
        
        # Generate anchor centers - but match the actual number we have
        # Calculate how many we actually need
        total_positions = fm_height * fm_width * num_anchors
        
        if num_valid != total_positions:
            # Size mismatch - adjust anchor generation
            # Use actual number of predictions
            actual_positions = num_valid // num_anchors
            if actual_positions * num_anchors < num_valid:
                actual_positions += 1
            
            # Generate anchors based on actual feature map that produced this many outputs
            actual_fm_size = int(np.sqrt(actual_positions / num_anchors))
            if actual_fm_size == 0:
                actual_fm_size = 1
            
            # Regenerate with corrected size
            anchors_temp = []
            count = 0
            for y in range(actual_fm_size):
                for x in range(actual_fm_size):
                    for _ in range(num_anchors):
                        if count >= num_valid:
                            break
                        anchors_temp.append([x * stride + stride // 2, y * stride + stride // 2])
                        count += 1
                    if count >= num_valid:
                        break
                if count >= num_valid:
                    break
            
            # Pad if needed
            while len(anchors_temp) < num_valid:
                anchors_temp.append(anchors_temp[-1] if anchors_temp else [stride // 2, stride // 2])
            
            anchor_centers = np.array(anchors_temp[:num_valid], dtype=np.float32)
        else:
            # Normal case
            anchor_centers = generate_anchors(fm_height, fm_width, stride, num_anchors)
            anchor_centers = anchor_centers[:num_valid]
        
        # Double-check sizes match
        if len(anchor_centers) != len(scores_matched):
            min_len = min(len(anchor_centers), len(scores_matched), len(bboxes_matched))
            anchor_centers = anchor_centers[:min_len]
            scores_matched = scores_matched[:min_len]
            bboxes_matched = bboxes_matched[:min_len]
        
        # Filter by threshold
        try:
            valid_mask = scores_matched > thresh
            
            if not np.any(valid_mask):
                continue
            
            valid_scores = scores_matched[valid_mask]
            valid_bboxes = bboxes_matched[valid_mask]
            valid_anchors = anchor_centers[valid_mask]
            
            # Decode boxes (distance format: l, t, r, b)
            decoded_boxes = distance2bbox(valid_anchors, valid_bboxes)
            
            all_scores.extend(valid_scores)
            all_boxes.extend(decoded_boxes)
            
        except Exception as e:
            print(f"Warning at scale {idx} (stride={stride}): {e}")
            print(f"  scores: {scores_matched.shape}, boxes: {bboxes_matched.shape}, anchors: {anchor_centers.shape}")
            continue
    
    if len(all_boxes) == 0:
        return None
    
    # Convert to numpy arrays
    all_scores = np.array(all_scores)
    all_boxes = np.array(all_boxes)
    
    # Apply NMS (simple version - just pick best for now)
    best_idx = np.argmax(all_scores)
    best_box = all_boxes[best_idx]
    best_score = all_scores[best_idx]
    
    # Scale back to original image coordinates
    scale_x = w / input_size
    scale_y = h / input_size
    
    x1, y1, x2, y2 = best_box
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    
    # Clamp to image boundaries
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    # Ensure valid box
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    
    return np.array([x1, y1, x2, y2]), float(best_score)

# ---------------- Main Loop ----------------
frame_id = 0
detect_count = 0
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
            
            # Debug: print output shapes on first detection frame
            if frame_id == 2:
                print("\nDetection output details:")
                for i, out in enumerate(det_out):
                    print(f"  Output {i}: shape={out.shape}, dtype={out.dtype}, "
                          f"size={out.size}, min={out.min():.4f}, max={out.max():.4f}")
                print()
            
            result = scrfd_postprocess(det_out, frame.shape, thresh=0.3)
            
            if result is not None:
                box, score = result
                x1, y1, x2, y2 = box
                
                # Extract face region
                face = frame[y1:y2, x1:x2]
                
                if face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0:
                    # Run recognition
                    face_input = preprocess_rec(face)
                    if face_input is not None:
                        try:
                            emb = rec_sess.run(None, {rec_input: face_input})[0]
                            
                            # Draw bounding box (THICK and BRIGHT)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                            
                            # Draw corners for emphasis
                            corner_len = 20
                            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 255), 3)
                            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 255), 3)
                            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 255), 3)
                            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 255), 3)
                            
                            # Draw score with background
                            label = f"FACE: {score:.2f}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
                            cv2.putText(frame, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                            
                            detect_count += 1
                            if detect_count % 10 == 1:  # Print occasionally
                                print(f"✓ FACE DETECTED! Score: {score:.3f}, Box: ({x1},{y1})-({x2},{y2})")
                        except Exception as e:
                            print(f"Recognition error: {e}")
            
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
        
        # Calculate and display FPS
        fps = 1.0 / (time.time() - t0 + 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Show detection count
        cv2.putText(frame, f"Faces: {detect_count}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
        
        # Display frame
        cv2.imshow("Sensor", frame)
        
        if cv2.waitKey(1) == 27:  # ESC key
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nCleanup complete. Total faces detected: {detect_count}")
