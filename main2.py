# face_snapshot.py
import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from datetime import datetime
import json
import glob

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================
CONFIG = {
    "camera": {
        "width": 1640,
        "height": 1232,
        "framerate": 30,
        "capture_interval": 1.0  # seconds between captures
    },
    "detection": {
        "input_size": (640, 640),
        "threshold": 0.3
    },
    "recognition": {
        "face_size": (112, 112),
        "similarity_threshold": 0.5  # Adjust this value (0.3-0.7)
    },
    "paths": {
        "models": os.path.expanduser("~/.insightface/models/light"),
        "database": os.path.expanduser("~/face_db")  # Where enroll.py saves embeddings
    },
    # ===== MODE SELECTION =====
    "mode": {
        "headless": False,  # True = no display, False = show camera preview
        "skip_frames": 2    # Process every N frames (for performance)
    }
}

# ============================================================================
# MODEL PATHS
# ============================================================================
SCRFD_MODEL = os.path.join(CONFIG["paths"]["models"], "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(CONFIG["paths"]["models"], "glintr100.onnx")

# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_system():
    """Initialize all system components"""
    mode = "HEADLESS" if CONFIG["mode"]["headless"] else "PREVIEW"
    print("=" * 60)
    print(f"INITIALIZING FACE RECOGNITION SYSTEM ({mode} MODE)")
    print("=" * 60)
    
    # Check models exist
    assert os.path.exists(SCRFD_MODEL), f"SCRFD model not found: {SCRFD_MODEL}"
    assert os.path.exists(ARCFACE_MODEL), f"ArcFace model not found: {ARCFACE_MODEL}"
    print(f"✓ Models loaded: {os.path.basename(SCRFD_MODEL)}, {os.path.basename(ARCFACE_MODEL)}")
    
    # Initialize ONNX sessions
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 2
    
    det_sess = ort.InferenceSession(
        SCRFD_MODEL, 
        opts, 
        providers=["CPUExecutionProvider"]
    )
    
    rec_sess = ort.InferenceSession(
        ARCFACE_MODEL, 
        opts, 
        providers=["CPUExecutionProvider"]
    )
    
    # Get input names
    det_input_name = det_sess.get_inputs()[0].name
    rec_input_name = rec_sess.get_inputs()[0].name
    
    print(f"✓ ONNX sessions initialized")
    print(f"  Detector input: {det_input_name}")
    print(f"  Recognizer input: {rec_input_name}")
    
    return det_sess, rec_sess, det_input_name, rec_input_name

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================
def load_database_embeddings():
    """Load embeddings from the same database that enroll.py uses"""
    db_dir = CONFIG["paths"]["database"]
    embeddings = {}
    
    print(f"\nLoading embeddings from: {db_dir}")
    
    if not os.path.exists(db_dir):
        print(f"✗ Database directory not found: {db_dir}")
        print("  Please run enroll.py first to create embeddings")
        return embeddings
    
    # Look for person directories
    person_dirs = [d for d in os.listdir(db_dir) 
                   if os.path.isdir(os.path.join(db_dir, d))]
    
    if not person_dirs:
        print("✗ No enrolled persons found in database")
        return embeddings
    
    for person_name in person_dirs:
        person_dir = os.path.join(db_dir, person_name)
        
        # Find all .npy files for this person
        npy_files = glob.glob(os.path.join(person_dir, "*.npy"))
        
        if not npy_files:
            continue
        
        # Load all embeddings for this person
        person_embeddings = []
        for npy_file in npy_files:
            try:
                emb = np.load(npy_file)
                person_embeddings.append(emb)
            except:
                continue
        
        if person_embeddings:
            # Average all embeddings for this person (more robust)
            avg_embedding = np.mean(person_embeddings, axis=0)
            embeddings[person_name] = avg_embedding
            print(f"  ✓ {person_name}: {len(person_embeddings)} embeddings averaged")
    
    print(f"\n✓ Database loaded: {len(embeddings)} persons")
    for name in embeddings.keys():
        print(f"    - {name}")
    
    return embeddings

# ============================================================================
# CAMERA FUNCTIONS
# ============================================================================
def initialize_camera():
    """Initialize Pi camera with GStreamer pipeline"""
    pipeline = (
        f"libcamerasrc ! "
        f"video/x-raw,width={CONFIG['camera']['width']},"
        f"height={CONFIG['camera']['height']},"
        f"framerate={CONFIG['camera']['framerate']}/1 ! "
        f"videoconvert ! appsink"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open camera")
    
    print(f"✓ Camera initialized: {CONFIG['camera']['width']}x{CONFIG['camera']['height']}")
    return cap

def capture_frame(cap, frame_id):
    """Capture and optionally display frame"""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    
    # Only print timestamp occasionally in headless mode
    if CONFIG["mode"]["headless"] and frame_id % 30 == 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Frame {frame_id} captured")
    
    return frame

# ============================================================================
# PREPROCESSING FUNCTIONS (Same as enroll.py)
# ============================================================================
def preprocess_detection(image):
    """Preprocess image for face detection (same as enroll.py)"""
    input_size = CONFIG["detection"]["input_size"]
    img_resized = cv2.resize(image, input_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - 127.5) / 128.0
    img_resized = img_resized.transpose(2, 0, 1)
    return np.expand_dims(img_resized, axis=0)

def preprocess_face(face_image):
    """Preprocess face image for recognition (same as enroll.py)"""
    if face_image.size == 0:
        return None
    
    face = cv2.resize(face_image, CONFIG["recognition"]["face_size"])
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = face.transpose(2, 0, 1)
    return np.expand_dims(face, axis=0)

# ============================================================================
# DETECTION POSTPROCESSING (Same as enroll.py)
# ============================================================================
def distance2bbox(points, distance, max_shape=None):
    """Same as enroll.py"""
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
    """Same as enroll.py"""
    anchors = []
    for y in range(height):
        for x in range(width):
            for _ in range(num_anchors):
                anchors.append([x * stride + stride // 2, y * stride + stride // 2])
    return np.array(anchors, dtype=np.float32)

def scrfd_postprocess(outputs, orig_shape, input_size=640, thresh=0.3):
    """Same as enroll.py"""
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
    
    return (x1, y1, x2, y2), float(best_score)

# ============================================================================
# EMBEDDING & RECOGNITION FUNCTIONS
# ============================================================================
def get_embedding(face_image, rec_sess, rec_input_name):
    """Generate face embedding using ArcFace"""
    face_input = preprocess_face(face_image)
    if face_input is None:
        return None
    
    embedding = rec_sess.run(None, {rec_input_name: face_input})[0]
    return embedding

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings"""
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def recognize_face(embedding, database, threshold=None):
    """
    Compare embedding with database from enroll.py
    Returns: ("HIT"/"MISS", matched_name, similarity_score)
    """
    if threshold is None:
        threshold = CONFIG["recognition"]["similarity_threshold"]
    
    if not database:
        return "MISS", None, 0.0
    
    best_match = None
    best_similarity = -1
    
    for name, db_embedding in database.items():
        similarity = cosine_similarity(embedding, db_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    
    if best_similarity >= threshold:
        return "HIT", best_match, best_similarity
    else:
        return "MISS", None, best_similarity

# ============================================================================
# DISPLAY FUNCTIONS (Only used in preview mode)
# ============================================================================
def draw_detection(frame, box, score, result, match_name, similarity):
    """Draw detection and recognition results on frame"""
    x1, y1, x2, y2 = box
    
    # Choose color based on result
    if result == "HIT":
        color = (0, 255, 0)  # Green
        label = f"{match_name}: {similarity:.3f}"
    else:
        color = (0, 0, 255)  # Red
        label = f"Unknown: {similarity:.3f}"
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw label with background
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Draw detection score
    score_text = f"Det: {score:.2f}"
    cv2.putText(frame, score_text, (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame

def draw_status(frame, fps, db_count, mode):
    """Draw status information on frame"""
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Database info
    cv2.putText(frame, f"DB: {db_count} persons", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # Mode indicator
    mode_color = (0, 255, 255) if mode == "PREVIEW" else (255, 165, 0)
    cv2.putText(frame, f"Mode: {mode}", (frame.shape[1] - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    # Instructions
    cv2.putText(frame, "ESC to exit", (frame.shape[1] - 150, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    """Main face recognition pipeline with mode selection"""
    # Determine mode
    mode = "HEADLESS" if CONFIG["mode"]["headless"] else "PREVIEW"
    
    print("\n" + "=" * 60)
    print(f"STARTING FACE RECOGNITION ({mode} MODE)")
    print("=" * 60)
    
    # Initialize components
    det_sess, rec_sess, det_input_name, rec_input_name = initialize_system()
    
    # Load database from enroll.py
    database = load_database_embeddings()
    
    if not database:
        print("\n⚠️  WARNING: No embeddings found!")
        print("   Run enroll.py first to enroll faces")
        if mode == "HEADLESS":
            print("   Exiting...")
            return
    
    # Initialize camera
    cap = initialize_camera()
    
    print("\n" + "=" * 60)
    print("READY FOR FACE RECOGNITION")
    print(f"Mode: {mode}")
    print(f"Capture interval: {CONFIG['camera']['capture_interval']}s")
    print(f"Similarity threshold: {CONFIG['recognition']['similarity_threshold']}")
    print(f"Database: {len(database)} persons")
    print("=" * 60 + "\n")
    
    if mode == "PREVIEW":
        print("Controls:")
        print("  ESC - Exit")
        print("  s   - Toggle display of detection box")
        print("  d   - Toggle debug info")
        print()
    
    frame_id = 0
    last_capture_time = 0
    show_detection = True
    show_debug = False
    
    try:
        while True:
            frame_start = time.time()
            frame_id += 1
            
            # Capture frame
            frame = capture_frame(cap, frame_id)
            
            # Skip frames for performance
            if frame_id % CONFIG["mode"]["skip_frames"] != 0:
                if mode == "PREVIEW":
                    display_frame = frame.copy()
                    if show_debug:
                        display_frame = draw_status(display_frame, 0, len(database), mode)
                    cv2.imshow("Face Recognition", display_frame)
                    
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC
                        break
                    elif key == ord('s'):
                        show_detection = not show_detection
                        print(f"Detection display: {'ON' if show_detection else 'OFF'}")
                    elif key == ord('d'):
                        show_debug = not show_debug
                        print(f"Debug info: {'ON' if show_debug else 'OFF'}")
                continue
            
            # Run face detection
            try:
                det_input = preprocess_detection(frame)
                det_outputs = det_sess.run(None, {det_input_name: det_input})
                result = scrfd_postprocess(det_outputs, frame.shape)
                
                if result is not None:
                    box, score = result
                    x1, y1, x2, y2 = box
                    
                    # Extract face
                    face = frame[y1:y2, x1:x2]
                    
                    if face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0:
                        # Generate embedding
                        embedding = get_embedding(face, rec_sess, rec_input_name)
                        
                        if embedding is not None:
                            # Recognize face
                            result, match_name, similarity = recognize_face(embedding, database)
                            
                            # Log result (always in both modes)
                            if result == "HIT":
                                print(f"✅ HIT - {match_name} (score: {similarity:.3f})")
                            else:
                                if similarity > 0.1:  # Only log if somewhat similar
                                    print(f"❌ MISS - No match (best: {similarity:.3f})")
                            
                            # Draw on frame in preview mode
                            if mode == "PREVIEW" and show_detection:
                                frame = draw_detection(frame, box, score, result, match_name, similarity)
                
            except Exception as e:
                if show_debug:
                    print(f"Detection error: {e}")
            
            # Calculate FPS
            fps = 1.0 / (time.time() - frame_start + 1e-6)
            
            # Display in preview mode
            if mode == "PREVIEW":
                display_frame = frame.copy()
                if show_debug:
                    display_frame = draw_status(display_frame, fps, len(database), mode)
                
                cv2.imshow("Face Recognition", display_frame)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
                elif key == ord('s'):
                    show_detection = not show_detection
                    print(f"Detection display: {'ON' if show_detection else 'OFF'}")
                elif key == ord('d'):
                    show_debug = not show_debug
                    print(f"Debug info: {'ON' if show_debug else 'OFF'}")
            
            # Sleep in headless mode
            elif mode == "HEADLESS":
                elapsed = time.time() - frame_start
                sleep_time = max(0, CONFIG["camera"]["capture_interval"] - elapsed)
                time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("SHUTTING DOWN SYSTEM")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        if mode == "PREVIEW":
            cv2.destroyAllWindows()
        print("✓ Camera released")
        print("System shutdown complete")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Quick configuration check
    print("Face Snapshot Configuration:")
    print(f"  Mode: {'HEADLESS' if CONFIG['mode']['headless'] else 'PREVIEW'}")
    print(f"  Database: {CONFIG['paths']['database']}")
    print(f"  Similarity threshold: {CONFIG['recognition']['similarity_threshold']}")
    
    # Optional: Let user confirm
    confirm = input("\nPress Enter to start or 'q' to quit: ")
    if confirm.lower() != 'q':
        main()
    else:
        print("Exiting...")
