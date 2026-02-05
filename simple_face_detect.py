# simple_detect.py
import cv2
import numpy as np
import onnxruntime as ort
import os

# Use your existing scrfd_postprocess function
from enroll import scrfd_postprocess, preprocess_det

# Initialize
MODEL_PATH = os.path.expanduser("~/.insightface/models/light/scrfd_500m_bnkps.onnx")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Camera
cap = cv2.VideoCapture("libcamerasrc ! video/x-raw,width=1640,height=1232 ! videoconvert ! appsink", 
                      cv2.CAP_GSTREAMER)

print("Simple Face Detection")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect
    det_input = preprocess_det(frame)
    det_outputs = session.run(None, {input_name: det_input})
    result = scrfd_postprocess(det_outputs, frame.shape, thresh=0.3)
    
    # Draw if face found
    if result:
        (x1, y1, x2, y2), score = result
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"Face: {score:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"\rFace detected! Score: {score:.3f}", end="")
    else:
        print("\rNo face detected", end="")
    
    cv2.imshow("Face Detect", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
