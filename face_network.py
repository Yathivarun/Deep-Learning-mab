"""
Face Detection System with Network Communication
Raspberry Pi - Sensor Node

Detects faces, generates embeddings, sends to laptop for matching.
Receives and displays person's images when match found.

Changes from original:
- Added network client (PiClient from network_protocol)
- Sends real embeddings to laptop (not local matching)
- Receives images from laptop to display
- Maintains 4-second detection cycle
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from datetime import datetime
import json
import glob
import threading

# Import network protocol
from network_protocol import PiClient

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================
CONFIG = {
    "camera": {
        "width": 1640,
        "height": 1232,
        "framerate": 30,
        "capture_interval": 4.0  # 4 seconds between detections (as required)
    },
    "detection": {
        "input_size": (640, 640),
        "threshold": 0.3
    },
    "recognition": {
        "face_size": (112, 112),
    },
    "paths": {
        "models": os.path.expanduser("~/.insightface/models/light"),
    },
    "network": {
        "laptop_ip": "192.168.137.1",
        "port": 5000,
        "enabled": True,  # Set to False to disable network (local mode)
        "auto_reconnect": True,
        "reconnect_delay": 5  # seconds
    },
    "display": {
        "show_preview": True,  # Show camera feed + detections
        "window_name": "Pi Sensor - Face Detection"
    }
}

# ============================================================================
# MODEL PATHS
# ============================================================================
SCRFD_MODEL = os.path.join(CONFIG["paths"]["models"], "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(CONFIG["paths"]["models"], "glintr100.onnx")

# ============================================================================
# GLOBAL STATE
# ============================================================================
class SensorState:
    def __init__(self):
        self.network_client = None
        self.connected = False
        self.last_detection_time = 0
        self.detection_count = 0
        self.match_count = 0
        self.current_display_images = []  # Images to show from laptop
        self.display_mode = "idle"  # "idle" or "person"
        self.last_person_id = None

state = SensorState()

# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_models():
    """Initialize ONNX models for detection and recognition"""
    print("=" * 60)
    print("INITIALIZING FACE DETECTION MODELS")
    print("=" * 60)
    
    # Check models exist
    assert os.path.exists(SCRFD_MODEL), f"SCRFD model not found: {SCRFD_MODEL}"
    assert os.path.exists(ARCFACE_MODEL), f"ArcFace model not found: {ARCFACE_MODEL}"
    print(f"✓ Models found")
    
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
    
    return det_sess, rec_sess, det_input_name, rec_input_name

def initialize_network():
    """Initialize network client to connect to laptop"""
    if not CONFIG["network"]["enabled"]:
        print("[NETWORK] Network disabled in config")
        return None
    
    print("\n" + "=" * 60)
    print("INITIALIZING NETWORK CONNECTION")
    print("=" * 60)
    
    client = PiClient(
        laptop_ip=CONFIG["network"]["laptop_ip"],
        port=CONFIG["network"]["port"]
    )
    
    # Set callbacks
    client.on_match_result = handle_match_result
    client.on_images_received = handle_images_received
    client.on_disconnected = handle_disconnection
    
    # Try to connect
    if client.connect(timeout=10):
        state.network_client = client
        state.connected = True
        print("✓ Connected to laptop!")
        return client
    else:
        print("✗ Failed to connect to laptop")
        if CONFIG["network"]["auto_reconnect"]:
            print(f"  Will retry in {CONFIG['network']['reconnect_delay']}s...")
        return None

# ============================================================================
# NETWORK CALLBACKS
# ============================================================================
def handle_match_result(msg):
    """Called when laptop sends match result"""
    hit = msg.get("hit")
    person_id = msg.get("person_id")
    score = msg.get("score")
    
    if hit:
        state.match_count += 1
        state.last_person_id = person_id
        print(f"\n✅ MATCH! Person #{person_id} (score: {score:.3f})")
        print("   Waiting for images from laptop...")
    else:
        print(f"\n❌ No match (best score: {score:.3f})")

def handle_images_received(image_list):
    """Called when laptop sends images to display"""
    if not image_list:
        # Empty list = return to slideshow/idle
        state.display_mode = "idle"
        state.current_display_images = []
        print("← Laptop: Return to idle mode")
    else:
        state.display_mode = "person"
        state.current_display_images = image_list
        print(f"← Received {len(image_list)} images from laptop")
        print(f"   Now displaying Person #{state.last_person_id}")
        
        # In Phase 2B, we'll show these on a separate display window
        # For now, just log that we received them

def handle_disconnection():
    """Called when connection to laptop is lost"""
    state.connected = False
    state.network_client = None
    print("\n✗ Lost connection to laptop!")
    
    if CONFIG["network"]["auto_reconnect"]:
        # Start reconnection in background thread
        threading.Thread(target=attempt_reconnect, daemon=True).start()

def attempt_reconnect():
    """Background thread to reconnect to laptop"""
    while CONFIG["network"]["auto_reconnect"] and not state.connected:
        print(f"[NETWORK] Reconnecting in {CONFIG['network']['reconnect_delay']}s...")
        time.sleep(CONFIG["network"]["reconnect_delay"])
        
        client = initialize_network()
        if client:
            state.network_client = client
            state.connected = True
            print("[NETWORK] ✓ Reconnected!")
            break

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

# ============================================================================
# PREPROCESSING FUNCTIONS (Same as original)
# ============================================================================
def preprocess_detection(image):
    """Preprocess image for face detection"""
    input_size = CONFIG["detection"]["input_size"]
    img_resized = cv2.resize(image, input_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - 127.5) / 128.0
    img_resized = img_resized.transpose(2, 0, 1)
    return np.expand_dims(img_resized, axis=0)

def preprocess_face(face_image):
    """Preprocess face image for recognition"""
    if face_image.size == 0:
        return None
    
    face = cv2.resize(face_image, CONFIG["recognition"]["face_size"])
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = face.transpose(2, 0, 1)
    return np.expand_dims(face, axis=0)

# ============================================================================
# DETECTION POSTPROCESSING (Same as original - keeping all helper functions)
# ============================================================================
def distance2bbox(points, distance, max_shape=None):
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
    """Generate anchor points for detection"""
    anchors = []
    for y in range(height):
        for x in range(width):
            for _ in range(num_anchors):
                anchors.append([x * stride + stride // 2, y * stride + stride // 2])
    return np.array(anchors, dtype=np.float32)

def scrfd_postprocess(outputs, orig_shape, input_size=640, thresh=0.3):
    """Postprocess SCRFD detection outputs (same as original)"""
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
# EMBEDDING GENERATION
# ============================================================================
def get_embedding(face_image, rec_sess, rec_input_name):
    """Generate face embedding using ArcFace"""
    face_input = preprocess_face(face_image)
    if face_input is None:
        return None
    
    embedding = rec_sess.run(None, {rec_input_name: face_input})[0]
    return embedding

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================
def draw_detection(frame, box, score):
    """Draw detection box on frame"""
    x1, y1, x2, y2 = box
    
    # Green box for detection
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Detection score
    label = f"Face: {score:.2f}"
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def draw_status(frame):
    """Draw status information on frame"""
    # Network status
    if state.connected:
        net_color = (0, 255, 0)
        net_text = "CONNECTED"
    else:
        net_color = (0, 0, 255)
        net_text = "DISCONNECTED"
    
    cv2.putText(frame, f"Network: {net_text}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, net_color, 2)
    
    # Detection count
    cv2.putText(frame, f"Detections: {state.detection_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Match count
    cv2.putText(frame, f"Matches: {state.match_count}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display mode
    mode_text = f"Mode: {state.display_mode.upper()}"
    if state.display_mode == "person" and state.last_person_id:
        mode_text += f" (#{state.last_person_id})"
    cv2.putText(frame, mode_text, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
    
    return frame

# ============================================================================
# MAIN DETECTION LOOP
# ============================================================================
def main():
    """Main detection and network communication loop"""
    print("\n" + "=" * 60)
    print("STARTING PI SENSOR NODE")
    print("=" * 60)
    
    # Initialize models
    det_sess, rec_sess, det_input_name, rec_input_name = initialize_models()
    
    # Initialize network
    initialize_network()
    
    # Initialize camera
    cap = initialize_camera()
    
    print("\n" + "=" * 60)
    print("PI SENSOR READY")
    print("=" * 60)
    print(f"Detection interval: {CONFIG['camera']['capture_interval']}s")
    print(f"Network: {'ENABLED' if CONFIG['network']['enabled'] else 'DISABLED'}")
    print(f"Preview: {'ENABLED' if CONFIG['display']['show_preview'] else 'DISABLED'}")
    print("=" * 60 + "\n")
    
    if CONFIG['display']['show_preview']:
        print("Press ESC to exit\n")
    else:
        print("Press Ctrl+C to exit\n")
    
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # Check if it's time to detect (4-second cycle)
            time_since_last = current_time - state.last_detection_time
            
            if time_since_last >= CONFIG['camera']['capture_interval']:
                state.last_detection_time = current_time
                
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
                            state.detection_count += 1
                            
                            # Generate embedding
                            embedding = get_embedding(face, rec_sess, rec_input_name)
                            
                            if embedding is not None:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                print(f"[{state.detection_count}] Face detected at {timestamp}")
                                
                                # Send to laptop if connected
                                if state.connected and state.network_client:
                                    if state.network_client.send_embedding(embedding, timestamp):
                                        print(f"    ✓ Sent embedding to laptop")
                                    else:
                                        print(f"    ✗ Failed to send (disconnected?)")
                                else:
                                    print(f"    ⚠️ Not connected to laptop")
                            
                            # Draw detection on preview
                            if CONFIG['display']['show_preview']:
                                frame = draw_detection(frame, box, score)
                    
                    else:
                        # No face detected this cycle
                        if frame_count % 30 == 0:  # Print occasionally
                            print(f"[Scan] No face detected...")
                
                except Exception as e:
                    print(f"Detection error: {e}")
            
            # Show preview if enabled
            if CONFIG['display']['show_preview']:
                display_frame = frame.copy()
                display_frame = draw_status(display_frame)
                cv2.imshow(CONFIG['display']['window_name'], display_frame)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
            else:
                # Small sleep in headless mode
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("SHUTTING DOWN PI SENSOR")
        print("=" * 60)
    
    finally:
        # Cleanup
        cap.release()
        if CONFIG['display']['show_preview']:
            cv2.destroyAllWindows()
        
        if state.network_client:
            state.network_client.disconnect()
        
        print("✓ Shutdown complete")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    print("\nPi Sensor Node Configuration:")
    print(f"  Network: {CONFIG['network']['enabled']}")
    print(f"  Laptop IP: {CONFIG['network']['laptop_ip']}")
    print(f"  Detection interval: {CONFIG['camera']['capture_interval']}s")
    print(f"  Preview: {CONFIG['display']['show_preview']}")
    
    input("\nPress Enter to start or Ctrl+C to cancel: ")
    main()
