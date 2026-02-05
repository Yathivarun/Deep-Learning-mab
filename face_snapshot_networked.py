"""
Face Detection System with Network Communication
Raspberry Pi - Sensor Node with Integrated Display

Detects faces, generates embeddings, sends to laptop for matching.
Receives and displays person's images when match found.
Shows stock slideshow when idle.

Phase 2B Features:
- Integrated pi_display.py for showing images
- Dual window mode (sensor preview + display)
- Automatic slideshow/POI switching
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

# Import network protocol and display
from network_protocol import PiClient
from pi_display import PiDisplay

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
        "stock_images": "stock_images"  # Directory for slideshow images
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
        "preview_window": "Pi Sensor - Face Detection",
        "enable_display": True,  # Enable display window for images
        "fullscreen": False  # Set True for HDMI fullscreen, False for windowed test
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
        self.display_window = None  # PiDisplay instance

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

def initialize_display():
    """Initialize Pi display window"""
    if not CONFIG["display"]["enable_display"]:
        print("[DISPLAY] Display disabled in config")
        return None
    
    print("\n" + "=" * 60)
    print("INITIALIZING DISPLAY WINDOW")
    print("=" * 60)
    
    display = PiDisplay(
        fullscreen=CONFIG["display"]["fullscreen"],
        stock_dir=CONFIG["paths"]["stock_images"]
    )
    
    # Start display thread
    display.start()
    
    state.display_window = display
    print("✓ Display window started")
    
    return display

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
        print("← Laptop: Return to slideshow")
        
        # Update display window
        if state.display_window:
            state.display_window.return_to_slideshow()
    else:
        state.display_mode = "person"
        state.current_display_images = image_list
        print(f"← Received {len(image_list)} images from laptop")
        print(f"   Now displaying Person #{state.last_person_id}")
        
        # Update display window
        if state.display_window:
            state.display_window.show_poi_images(image_list)

def handle_disconnection():
    """Called when connection to laptop is lost"""
    state.connected = False
    state.network_client = None
    print("\n✗ Lost connection to laptop!")
    
    # Return display to slideshow
    if state.display_window:
        state.display_window.return_to_slideshow()
    
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
            print("[NETWORK] ✓ Reconnected successfully!")
            break

# ============================================================================
# CAMERA INITIALIZATION
# ============================================================================
def initialize_camera():
    """Initialize Pi Camera"""
    print("\n" + "=" * 60)
    print("INITIALIZING CAMERA")
    print("=" * 60)
    
    try:
        from picamera2 import Picamera2
        
        picam = Picamera2()
        
        # Configure camera
        config = picam.create_preview_configuration(
            main={"size": (CONFIG["camera"]["width"], CONFIG["camera"]["height"])},
            controls={"FrameRate": CONFIG["camera"]["framerate"]}
        )
        picam.configure(config)
        
        # Start camera
        picam.start()
        time.sleep(1)  # Warmup
        
        print(f"✓ Camera initialized ({CONFIG['camera']['width']}x{CONFIG['camera']['height']})")
        return picam
        
    except ImportError:
        print("✗ picamera2 not found, using fallback (webcam)")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera"]["height"])
        time.sleep(1)
        return cap

# ============================================================================
# DETECTION & RECOGNITION
# ============================================================================
def preprocess_detection(image, input_size):
    """Prepare image for SCRFD model"""
    resized = cv2.resize(image, input_size)
    blob = cv2.dnn.blobFromImage(
        resized, 
        1.0/128.0, 
        input_size, 
        (127.5, 127.5, 127.5), 
        swapRB=True
    )
    return blob

def detect_faces(det_sess, det_input_name, image):
    """Run face detection"""
    input_size = CONFIG["detection"]["input_size"]
    blob = preprocess_detection(image, input_size)
    
    outputs = det_sess.run(None, {det_input_name: blob})
    
    # Parse outputs (simplified version)
    # SCRFD outputs: [scores, bboxes, kpss]
    scores = outputs[0][0]
    bboxes = outputs[1][0]
    
    # Filter by threshold
    threshold = CONFIG["detection"]["threshold"]
    valid = scores > threshold
    
    if not np.any(valid):
        return []
    
    # Scale bboxes back to original image size
    h, w = image.shape[:2]
    scale_x = w / input_size[0]
    scale_y = h / input_size[1]
    
    faces = []
    for i, is_valid in enumerate(valid):
        if is_valid:
            bbox = bboxes[i]
            x1 = int(bbox[0] * scale_x)
            y1 = int(bbox[1] * scale_y)
            x2 = int(bbox[2] * scale_x)
            y2 = int(bbox[3] * scale_y)
            score = float(scores[i])
            faces.append({
                "bbox": [x1, y1, x2, y2],
                "score": score
            })
    
    return faces

def extract_face_aligned(image, bbox):
    """Extract and align face for recognition"""
    x1, y1, x2, y2 = bbox
    
    # Add margin
    margin = 20
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.shape[1], x2 + margin)
    y2 = min(image.shape[0], y2 + margin)
    
    # Crop face
    face_img = image[y1:y2, x1:x2]
    
    # Resize to model input size
    face_size = CONFIG["recognition"]["face_size"]
    face_resized = cv2.resize(face_img, face_size)
    
    return face_resized

def preprocess_recognition(face_img):
    """Prepare face for ArcFace model"""
    # Normalize to [-1, 1]
    face_blob = cv2.dnn.blobFromImage(
        face_img,
        1.0/127.5,
        CONFIG["recognition"]["face_size"],
        (127.5, 127.5, 127.5),
        swapRB=True
    )
    return face_blob

def generate_embedding(rec_sess, rec_input_name, face_img):
    """Generate face embedding using ArcFace"""
    blob = preprocess_recognition(face_img)
    embedding = rec_sess.run(None, {rec_input_name: blob})[0]
    
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding.flatten().astype(np.float32)

# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "RASPBERRY PI FACE DETECTION SENSOR")
    print(" " * 30 + "Phase 2B - With Display")
    print("=" * 80 + "\n")
    
    # Initialize components
    det_sess, rec_sess, det_input_name, rec_input_name = initialize_models()
    display = initialize_display()
    camera = initialize_camera()
    network_client = initialize_network()
    
    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print(f"Detection cycle: {CONFIG['camera']['capture_interval']}s")
    print(f"Network: {'✓ Connected' if state.connected else '✗ Disconnected'}")
    print(f"Display: {'✓ Enabled' if display else '✗ Disabled'}")
    print(f"Preview: {'✓ Enabled' if CONFIG['display']['show_preview'] else '✗ Disabled'}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    last_capture_time = 0
    frame_count = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Capture frame
            try:
                # Try picamera2 first
                if hasattr(camera, 'capture_array'):
                    frame = camera.capture_array()
                else:
                    # Fallback to OpenCV
                    ret, frame = camera.read()
                    if not ret:
                        print("Failed to capture frame")
                        time.sleep(0.1)
                        continue
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Check if it's time to detect (4-second cycle)
            if current_time - last_capture_time >= CONFIG["camera"]["capture_interval"]:
                last_capture_time = current_time
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Detection #{state.detection_count + 1}")
                
                # Detect faces
                faces = detect_faces(det_sess, det_input_name, frame)
                
                if faces:
                    # Take the first/largest face
                    face = faces[0]
                    bbox = face["bbox"]
                    score = face["score"]
                    
                    print(f"  ✓ Face detected (confidence: {score:.2f})")
                    print(f"    BBox: {bbox}")
                    
                    # Extract face and generate embedding
                    face_img = extract_face_aligned(frame, bbox)
                    embedding = generate_embedding(rec_sess, rec_input_name, face_img)
                    
                    print(f"  ✓ Embedding generated (shape: {embedding.shape})")
                    
                    state.detection_count += 1
                    
                    # Send to laptop if connected
                    if state.connected and state.network_client:
                        timestamp = datetime.now().isoformat()
                        
                        if state.network_client.send_embedding(embedding, timestamp):
                            print(f"  ✓ Sent to laptop")
                        else:
                            print(f"  ✗ Failed to send (disconnected?)")
                            state.connected = False
                    else:
                        print(f"  ⚠️ Not connected to laptop")
                    
                    # Draw on preview if enabled
                    if CONFIG["display"]["show_preview"]:
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Face: {score:.2f}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print(f"  ✗ No face detected")
            
            # Show preview window
            if CONFIG["display"]["show_preview"]:
                # Resize for display
                display_frame = cv2.resize(frame, (640, 480))
                
                # Add status overlay
                status_text = f"Status: {'Connected' if state.connected else 'Disconnected'}"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if state.connected else (0, 0, 255), 2)
                
                cv2.putText(display_frame, f"Detections: {state.detection_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f"Matches: {state.match_count}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(CONFIG["display"]["preview_window"], display_frame)
                
                # Check for ESC key
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    print("\nESC pressed, exiting...")
                    break
            else:
                # Small sleep to prevent CPU spinning
                time.sleep(0.03)
    
    except KeyboardInterrupt:
        print("\n\nCtrl+C detected, shutting down...")
    
    finally:
        # Cleanup
        print("\n[CLEANUP] Stopping camera...")
        if hasattr(camera, 'stop'):
            camera.stop()
        elif hasattr(camera, 'release'):
            camera.release()
        
        print("[CLEANUP] Closing display...")
        if state.display_window:
            state.display_window.stop()
        
        print("[CLEANUP] Disconnecting network...")
        if state.network_client:
            state.network_client.disconnect()
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("SHUTDOWN COMPLETE")
        print(f"Total detections: {state.detection_count}")
        print(f"Total matches: {state.match_count}")
        print("=" * 60)

if __name__ == "__main__":
    main()
