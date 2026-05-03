import os
import tempfile
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2

# load model once at startup
MODEL_PATH = os.path.join(os.getcwd(), "traffic_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

# label map (same as in recognition.py)
label_cod = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}


def detect_signs(frame):
    """Fast OpenCV sign detector using red/blue color blobs and contour filters."""
    height, width = frame.shape[:2]
    area_limit = height * width * 0.35

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    blue_lower = np.array([90, 60, 50])
    blue_upper = np.array([135, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    mask = red_mask | blue_mask

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 250 or area > area_limit:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue

        aspect_ratio = w / float(h)
        if aspect_ratio < 0.45 or aspect_ratio > 1.9:
            continue

        pad = max(4, int(0.08 * max(w, h)))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)

        boxes.append((x1, y1, x2 - x1, y2 - y1, area))

    boxes.sort(key=lambda b: b[4], reverse=True)
    return boxes


def preprocess_frame(img):
    """Convert decoded OpenCV frame to model input."""
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img / 255.0
    return img


def preprocess_image(img_bytes):
    """Convert raw image bytes from upload to model input."""
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return preprocess_frame(img)


def classify_crop(crop):
    """Classify one cropped sign region and return class metadata."""
    inp = preprocess_frame(crop)
    if inp is None:
        return None

    pred = model.predict(np.expand_dims(inp, 0), verbose=0)[0]
    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id])
    label = label_cod.get(class_id, f"Unknown ({class_id})")
    return class_id, label, confidence


def process_uploaded_video(video_bytes):
    """Extract timestamped sign classifications from an uploaded MP4 file."""
    sample_fps = max(1.0, float(os.environ.get("VIDEO_SAMPLE_FPS", "4")))
    max_boxes_per_frame = max(1, int(os.environ.get("VIDEO_MAX_BOXES_PER_FRAME", "2")))
    min_class_conf = float(os.environ.get("VIDEO_MIN_CLASS_CONF", "0.45"))
    dedupe_seconds = float(os.environ.get("VIDEO_DEDUPE_SECONDS", "0.75"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name

    detections = []
    last_seen = {}

    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("could not open uploaded video")

        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 30.0
        frame_step = max(1, int(round(fps / sample_fps)))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps
            boxes = detect_signs(frame)[:max_boxes_per_frame]
            for (x, y, w, h, _) in boxes:
                crop = frame[y:y + h, x:x + w]
                result = classify_crop(crop)
                if result is None:
                    continue

                class_id, label, confidence = result
                if confidence < min_class_conf:
                    continue

                last_ts = last_seen.get(label)
                if last_ts is not None and (timestamp - last_ts) < dedupe_seconds:
                    continue

                last_seen[label] = timestamp
                detections.append({
                    "timestamp": round(timestamp, 2),
                    "class_id": class_id,
                    "label": label,
                    "confidence": round(confidence, 3),
                })

            frame_idx += 1

        cap.release()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    return detections


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify(error="no file part"), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="no selected file"), 400

    raw_bytes = file.read()
    filename = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    is_mp4 = filename.endswith(".mp4") or content_type in {"video/mp4", "application/mp4"}

    if is_mp4:
        try:
            detections = process_uploaded_video(raw_bytes)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(file_type="video", detections=detections, total_detections=len(detections))

    arr = np.frombuffer(raw_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(error="could not decode image or mp4"), 400

    boxes = detect_signs(frame)
    if boxes:
        x, y, w, h, _ = boxes[0]
        frame = frame[y:y + h, x:x + w]

    result = classify_crop(frame)
    if result is None:
        return jsonify(error="could not preprocess image"), 400

    class_id, label, confidence = result
    return jsonify(file_type="image", class_id=class_id, label=label, confidence=round(confidence, 3))


if __name__ == "__main__":
    # ensure template folder is correct when run from workspace root
    app.run(debug=True)
