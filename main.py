# face_snapshot.py
import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "camera": {
        "width": 1640,
        "height": 1232,
        "framerate": 30,
        "capture_interval": 1.0  # seconds
    },
    "detection": {
        "input_size": (640, 640),
        "threshold": 0.3
    },
    "recognition": {
        "face_size": (112, 112),
        "similarity_threshold": 0.5
    },
    "paths": {
        "models": os.path.expanduser("~/.insightface/models/light"),
        "database": os.path.expanduser("~/face_db"),
        "test_embeddings": "test_embeddings"  # Directory for test embeddings
    }
}

# ============================================================================
# MODEL PATHS
# ============================================================================
SCRFD_MODEL = os.path.join(CONFIG["paths"]["models"], "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(CONFIG["paths"]["models"], "glintr100.onnx")

# ============================================================================
# TEST EMBEDDINGS (For local testing)
# ============================================================================
# These can be loaded from .npy files or hardcoded for testing
# Structure: {"name": embedding_array}
TEST_EMBEDDINGS = {
    "john": None,  # Will be loaded from file
    "jane": None,
    "unknown": None
}

# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_system():
    """Initialize all system components"""
    print("=" * 60)
    print("INITIALIZING HEADLESS FACE RECOGNITION SYSTEM")
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

def capture_frame(cap):
    """Capture a single frame from camera"""
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] Frame captured: {frame.shape[1]}x{frame.shape[0]}")
    return frame

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================
def preprocess_detection(image):
    """Preprocess image for face detection"""
    input_size = CONFIG["detection"]["input_size"]
    img_resized = cv2.resize(image, input_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - 127.5) / 128.0  # SCRFD normalization
    img_resized = img_resized.transpose(2, 0, 1)  # HWC to CHW
    return np.expand_dims(img_resized, axis=0)  # Add batch dimension

def preprocess_face(face_image):
    """Preprocess face image for recognition"""
    if face_image.size == 0:
        return None
    
    face = cv2.resize(face_image, CONFIG["recognition"]["face_size"])
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0  # ArcFace normalization
    face = face.transpose(2, 0, 1)  # HWC to CHW
    return np.expand_dims(face, axis=0)  # Add batch dimension

# ============================================================================
# DETECTION POSTPROCESSING
# ============================================================================
def distance_to_bbox(points, distance, max_shape=None):
    """Convert distance predictions to bounding boxes"""
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
    """Generate anchor points for SCRFD"""
    anchors = []
    for y in range(height):
        for x in range(width):
            for _ in range(num_anchors):
                anchors.append([x * stride + stride // 2, y * stride + stride // 2])
    return np.array(anchors, dtype=np.float32)

def scrfd_postprocess(detection_outputs, original_shape, threshold=0.3):
    """
    Postprocess SCRFD outputs to get face bounding boxes
    Returns: (x1, y1, x2, y2) or None if no face detected
    """
    h, w = original_shape[:2]
    input_size = CONFIG["detection"]["input_size"][0]
    
    # SCRFD configuration
    fmc = 3  # Number of feature maps
    feat_stride_fpn = [8, 16, 32]
    num_anchors = 2
    outputs_per_scale = len(detection_outputs) // fmc
    
    all_boxes = []
    all_scores = []
    
    # Process each scale
    for idx in range(fmc):
        stride = feat_stride_fpn[idx]
        fm_height = input_size // stride
        fm_width = input_size // stride
        
        score_idx = idx * outputs_per_scale
        bbox_idx = score_idx + 1
        
        if score_idx >= len(detection_outputs) or bbox_idx >= len(detection_outputs):
            continue
        
        scores = detection_outputs[score_idx]
        bboxes = detection_outputs[bbox_idx]
        
        # Remove batch dimension if present
        if len(scores.shape) == 3:
            scores = scores[0]
        if len(bboxes.shape) == 3:
            bboxes = bboxes[0]
        
        scores_flat = scores.flatten()
        
        # Reshape bboxes to (N, 4)
        if len(bboxes.shape) == 1:
            num_boxes = len(bboxes) // 4
            bboxes_reshaped = bboxes.reshape(num_boxes, 4)
        elif len(bboxes.shape) == 2 and bboxes.shape[1] == 4:
            bboxes_reshaped = bboxes
        else:
            total_elements = bboxes.size
            num_boxes = total_elements // 4
            bboxes_reshaped = bboxes.reshape(num_boxes, 4)
        
        # Match sizes
        num_valid = min(len(scores_flat), len(bboxes_reshaped))
        if num_valid == 0:
            continue
        
        scores_matched = scores_flat[:num_valid]
        bboxes_matched = bboxes_reshaped[:num_valid]
        
        # Generate anchors
        total_positions = fm_height * fm_width * num_anchors
        
        if num_valid != total_positions:
            # Adjust for size mismatch
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
        
        # Ensure size match
        if len(anchor_centers) != len(scores_matched):
            min_len = min(len(anchor_centers), len(scores_matched), len(bboxes_matched))
            anchor_centers = anchor_centers[:min_len]
            scores_matched = scores_matched[:min_len]
            bboxes_matched = bboxes_matched[:min_len]
        
        try:
            # Filter by threshold
            valid_mask = scores_matched > threshold
            if not np.any(valid_mask):
                continue
            
            valid_scores = scores_matched[valid_mask]
            valid_bboxes = bboxes_matched[valid_mask]
            valid_anchors = anchor_centers[valid_mask]
            
            # Decode boxes
            decoded_boxes = distance_to_bbox(valid_anchors, valid_bboxes)
            all_scores.extend(valid_scores)
            all_boxes.extend(decoded_boxes)
        except Exception as e:
            continue
    
    if len(all_boxes) == 0:
        return None
    
    # Get best box
    all_scores = np.array(all_scores)
    all_boxes = np.array(all_boxes)
    best_idx = np.argmax(all_scores)
    best_box = all_boxes[best_idx]
    
    # Scale to original image
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
    
    # Validate box
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    
    return (x1, y1, x2, y2)

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

def load_test_embeddings():
    """Load test embeddings from files"""
    test_dir = CONFIG["paths"]["test_embeddings"]
    embeddings = {}
    
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.endswith('.npy'):
                name = os.path.splitext(file)[0]
                path = os.path.join(test_dir, file)
                embeddings[name] = np.load(path)
                print(f"  Loaded test embedding: {name}")
    
    # If no files, create dummy embeddings for testing
    if not embeddings:
        print("  No test embeddings found, using dummy data")
        dummy_emb = np.random.randn(1, 512).astype(np.float32)
        embeddings["john"] = dummy_emb * 0.9
        embeddings["jane"] = dummy_emb * 1.1
    
    return embeddings

def recognize_face(embedding, database, threshold=None):
    """
    Compare embedding with database
    
    FUTURE FASTAPI INTEGRATION POINT:
    Here is where you would send the embedding to a remote server
    instead of local comparison
    """
    if threshold is None:
        threshold = CONFIG["recognition"]["similarity_threshold"]
    
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
# MAIN PIPELINE
# ============================================================================
def main():
    """Main headless face recognition pipeline"""
    print("\n" + "=" * 60)
    print("STARTING HEADLESS SNAPSHOT PIPELINE")
    print("=" * 60)
    
    # Initialize components
    det_sess, rec_sess, det_input_name, rec_input_name = initialize_system()
    
    # Load test embeddings
    print("\nLoading test embeddings...")
    test_database = load_test_embeddings()
    print(f"✓ Test database loaded: {len(test_database)} embeddings")
    
    # Initialize camera
    cap = initialize_camera()
    
    print("\n" + "=" * 60)
    print("READY FOR FACE RECOGNITION")
    print(f"Capture interval: {CONFIG['camera']['capture_interval']}s")
    print(f"Similarity threshold: {CONFIG['recognition']['similarity_threshold']}")
    print("=" * 60 + "\n")
    
    try:
        while True:
            # 1. CAPTURE FRAME
            frame = capture_frame(cap)
            
            # 2. FACE DETECTION
            det_input = preprocess_detection(frame)
            det_outputs = det_sess.run(None, {det_input_name: det_input})
            face_box = scrfd_postprocess(det_outputs, frame.shape)
            
            if face_box is None:
                print("  No face detected, waiting for next capture...")
                time.sleep(CONFIG['camera']['capture_interval'])
                continue
            
            x1, y1, x2, y2 = face_box
            print(f"  ✓ Face detected at [{x1},{y1}]-[{x2},{y2}]")
            
            # 3. EXTRACT AND PROCESS FACE
            face_image = frame[y1:y2, x1:x2]
            
            if face_image.size == 0:
                print("  ✗ Invalid face region")
                time.sleep(CONFIG['camera']['capture_interval'])
                continue
            
            embedding = get_embedding(face_image, rec_sess, rec_input_name)
            
            if embedding is None:
                print("  ✗ Failed to generate embedding")
                time.sleep(CONFIG['camera']['capture_interval'])
                continue
            
            print(f"  ✓ Embedding generated: {embedding.shape}")
            
            # 4. FACE RECOGNITION
            result, match_name, similarity = recognize_face(embedding, test_database)
            
            if result == "HIT":
                print(f"  ✅ {result} - Known Face: {match_name} (similarity: {similarity:.3f})")
                
                # FUTURE FASTAPI INTEGRATION POINT:
                # Here you would send the result to your FastAPI server
                # Example: requests.post("http://your-server/api/recognize", 
                #                      json={"result": result, "name": match_name, 
                #                            "similarity": float(similarity)})
                
            else:
                print(f"  ❌ {result} - Unknown Face (best similarity: {similarity:.3f})")
            
            print("-" * 40)
            
            # 5. WAIT FOR NEXT CAPTURE
            time.sleep(CONFIG['camera']['capture_interval'])
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("SHUTTING DOWN SYSTEM")
        print("=" * 60)
    except Exception as e:
        print(f"\nERROR: {e}")
    finally:
        cap.release()
        print("✓ Camera released")
        print("System shutdown complete")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    main()
