import cv2
import numpy as np
import onnxruntime as ort
import os
import json
import time
from datetime import datetime

# ---------------- Paths ----------------
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE = os.path.join(BASE, "glintr100.onnx")

# Database directory
DB_DIR = os.path.expanduser("~/face_db")
os.makedirs(DB_DIR, exist_ok=True)

print("Detector :", SCRFD)
print("Recognizer:", ARCFACE)
print("Database  :", DB_DIR)

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

# ---------------- Camera ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

# ---------------- Preprocess Functions ----------------
def preprocess_det(img, input_size=(640, 640)):
    img_resized = cv2.resize(img, input_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - 127.5) / 128.0
    img_resized = img_resized.transpose(2, 0, 1)
    return np.expand_dims(img_resized, axis=0)

def preprocess_rec(face):
    if face.size == 0:
        return None
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = face.transpose(2, 0, 1)
    return np.expand_dims(face, axis=0)

def distance2bbox(points, distance, max_shape=None):
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
    anchors = []
    for y in range(height):
        for x in range(width):
            for _ in range(num_anchors):
                anchors.append([x * stride + stride // 2, y * stride + stride // 2])
    return np.array(anchors, dtype=np.float32)

def scrfd_postprocess(outputs, orig_shape, input_size=640, thresh=0.3):
    h, w = orig_shape[:2]
    fmc = 3
    feat_stride_fpn = [8, 16, 32]
    num_anchors = 2
    
    all_boxes = []
    all_scores = []
    outputs_per_scale = len(outputs) // fmc
    
    for idx in range(fmc):
        stride = feat_stride_fpn[idx]
        fm_height = input_size // stride
        fm_width = input_size // stride
        
        score_idx = idx * outputs_per_scale
        bbox_idx = score_idx + 1
        
        if score_idx >= len(outputs) or bbox_idx >= len(outputs):
            continue
        
        scores = outputs[score_idx]
        bboxes = outputs[bbox_idx]
        
        if len(scores.shape) == 3:
            scores = scores[0]
        if len(bboxes.shape) == 3:
            bboxes = bboxes[0]
        
        scores_flat = scores.flatten()
        
        if len(bboxes.shape) == 1:
            num_boxes = len(bboxes) // 4
            bboxes_reshaped = bboxes.reshape(num_boxes, 4)
        elif len(bboxes.shape) == 2 and bboxes.shape[1] == 4:
            bboxes_reshaped = bboxes
        else:
            total_elements = bboxes.size
            num_boxes = total_elements // 4
            bboxes_reshaped = bboxes.reshape(num_boxes, 4)
        
        num_valid = min(len(scores_flat), len(bboxes_reshaped))
        if num_valid == 0:
            continue
        
        scores_matched = scores_flat[:num_valid]
        bboxes_matched = bboxes_reshaped[:num_valid]
        
        total_positions = fm_height * fm_width * num_anchors
        
        if num_valid != total_positions:
            actual_positions = num_valid // num_anchors
            if actual_positions * num_anchors < num_valid:
                actual_positions += 1
            actual_fm_size = int(np.sqrt(actual_positions / num_anchors))
            if actual_fm_size == 0:
                actual_fm_size = 1
            
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
            
            while len(anchors_temp) < num_valid:
                anchors_temp.append(anchors_temp[-1] if anchors_temp else [stride // 2, stride // 2])
            
            anchor_centers = np.array(anchors_temp[:num_valid], dtype=np.float32)
        else:
            anchor_centers = generate_anchors(fm_height, fm_width, stride, num_anchors)
            anchor_centers = anchor_centers[:num_valid]
        
        if len(anchor_centers) != len(scores_matched):
            min_len = min(len(anchor_centers), len(scores_matched), len(bboxes_matched))
            anchor_centers = anchor_centers[:min_len]
            scores_matched = scores_matched[:min_len]
            bboxes_matched = bboxes_matched[:min_len]
        
        try:
            valid_mask = scores_matched > thresh
            if not np.any(valid_mask):
                continue
            
            valid_scores = scores_matched[valid_mask]
            valid_bboxes = bboxes_matched[valid_mask]
            valid_anchors = anchor_centers[valid_mask]
            
            decoded_boxes = distance2bbox(valid_anchors, valid_bboxes)
            all_scores.extend(valid_scores)
            all_boxes.extend(decoded_boxes)
        except Exception as e:
            continue
    
    if len(all_boxes) == 0:
        return None
    
    all_scores = np.array(all_scores)
    all_boxes = np.array(all_boxes)
    
    best_idx = np.argmax(all_scores)
    best_box = all_boxes[best_idx]
    best_score = all_scores[best_idx]
    
    scale_x = w / input_size
    scale_y = h / input_size
    
    x1, y1, x2, y2 = best_box
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    
    return np.array([x1, y1, x2, y2]), float(best_score)

# ---------------- Save Embedding ----------------
def save_embedding(name, embedding, face_image):
    """Save face embedding and image to database"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_dir = os.path.join(DB_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Save embedding
    embedding_file = os.path.join(person_dir, f"{timestamp}.npy")
    np.save(embedding_file, embedding)
    
    # Save face image
    image_file = os.path.join(person_dir, f"{timestamp}.jpg")
    cv2.imwrite(image_file, face_image)
    
    # Save metadata
    metadata = {
        "name": name,
        "timestamp": timestamp,
        "embedding_shape": embedding.shape,
        "embedding_file": embedding_file,
        "image_file": image_file
    }
    
    metadata_file = os.path.join(person_dir, f"{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved: {name} ({timestamp})")
    print(f"  Embedding: {embedding_file}")
    print(f"  Image: {image_file}")
    
    return metadata

# ---------------- Main Enrollment Loop ----------------
print("\n=== FACE ENROLLMENT MODE ===")
print("Controls:")
print("  SPACE - Capture face and enroll")
print("  ESC   - Exit")
print("\nWaiting for face...\n")

captured_face = None
captured_box = None
captured_embedding = None

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        try:
            det_input_data = preprocess_det(frame)
            det_out = det_sess.run(None, {det_input: det_input_data})
            result = scrfd_postprocess(det_out, frame.shape, thresh=0.3)
            
            if result is not None:
                box, score = result
                x1, y1, x2, y2 = box
                
                face = frame[y1:y2, x1:x2]
                
                if face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0:
                    # Generate embedding
                    face_input = preprocess_rec(face)
                    if face_input is not None:
                        emb = rec_sess.run(None, {rec_input: face_input})[0]
                        
                        # Store for capture
                        captured_face = face.copy()
                        captured_box = box
                        captured_embedding = emb
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, f"Score: {score:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Ready to capture indicator
                        cv2.putText(frame, "PRESS SPACE TO CAPTURE", (20, frame.shape[0] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        except Exception as e:
            print(f"Detection error: {e}")
        
        # Display
        cv2.imshow("Enrollment", frame)
        
        key = cv2.waitKey(1)
        
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if captured_face is not None and captured_embedding is not None:
                print("\n" + "="*50)
                name = input("Enter person's name: ").strip()
                
                if name:
                    save_embedding(name, captured_embedding, captured_face)
                    print(f"✓ Enrollment complete for: {name}")
                    print("="*50 + "\n")
                    
                    # Reset
                    captured_face = None
                    captured_embedding = None
                    
                    time.sleep(1)
                else:
                    print("✗ Name cannot be empty. Try again.")
            else:
                print("✗ No face detected. Please position your face in frame.")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nEnrollment session ended")
