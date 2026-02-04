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

# ---------------- Camera (IMX Full FOV) ----------------

pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# ---------------- Preprocess ----------------

def preprocess_det(img):
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32)
    img = img.transpose(2, 0, 1)
    return img[None]

def preprocess_rec(face):
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = face.transpose(2, 0, 1)
    return face[None]

# ---------------- SCRFD Proper Decode ----------------

def scrfd_postprocess(outputs, frame_shape, thresh=0.4):
    h, w, _ = frame_shape

    best_score = 0
    best_box = None

    # SCRFD outputs come as: score0, box0, score1, box1, score2, box2
    for i in range(0, len(outputs), 2):
        scores = outputs[i][0]      # (N,)
        boxes  = outputs[i+1][0]    # (N,4)

        for j in range(scores.shape[0]):
            s = float(scores[j])
            if s > thresh and s > best_score:
                best_score = s
                best_box = boxes[j]

    if best_box is None:
        return None

    # best_box is now [x1,y1,x2,y2] in 640x640 coords
    x1, y1, x2, y2 = best_box

    scale_x = w / 640.0
    scale_y = h / 640.0

    return np.array([
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    ])



# ---------------- Main Loop ----------------

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames for Pi speed
    frame_id += 1
    if frame_id % 2 != 0:
        continue

    t0 = time.time()

    det_out = det_sess.run(None, {det_input: preprocess_det(frame)})
    box = scrfd_postprocess(det_out, frame.shape)

    if box is not None:
        x1, y1, x2, y2 = box
        h, w, _ = frame.shape

        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)

        face = frame[y1:y2, x1:x2]

        if face.size > 0:
            emb = rec_sess.run(None, {rec_input: preprocess_rec(face)})[0]
            print("Embedding shape:", emb.shape)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    fps = 1.0 / (time.time() - t0 + 1e-6)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sensor", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
