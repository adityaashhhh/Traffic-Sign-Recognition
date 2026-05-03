import cv2
import numpy as np
import tensorflow as tf


# Load trained classifier.
model = tf.keras.models.load_model("traffic_model.keras")

# Label mapping for the classifier.
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


def preprocess(img):
    """Resize and normalize for model input."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img / 255.0
    return img


def detect_signs(frame, min_area=250, max_area_ratio=0.35):
    """Detect signs using contrast enhancement, HSV filtering, LoG binarization, and shape checks."""
    height, width = frame.shape[:2]
    area_limit = height * width * max_area_ratio

    # Increase dynamic range and local contrast on the value channel.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_ch = cv2.normalize(v_ch, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    v_ch = clahe.apply(v_ch)
    hsv_enhanced = cv2.merge([h_ch, s_ch, v_ch])
    _, sat_enhanced, val_enhanced = cv2.split(hsv_enhanced)
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)

    red_lower_1 = np.array([0, 70, 50])
    red_upper_1 = np.array([10, 255, 255])
    red_lower_2 = np.array([170, 70, 50])
    red_upper_2 = np.array([180, 255, 255])
    blue_lower = np.array([90, 60, 50])
    blue_upper = np.array([135, 255, 255])
    green_lower = np.array([35, 50, 50])
    green_upper = np.array([85, 255, 255])

    red_mask = cv2.inRange(hsv_enhanced, red_lower_1, red_upper_1) | cv2.inRange(hsv_enhanced, red_lower_2, red_upper_2)
    blue_mask = cv2.inRange(hsv_enhanced, blue_lower, blue_upper)
    green_mask = cv2.inRange(hsv_enhanced, green_lower, green_upper)
    color_mask = (red_mask | blue_mask) & (~green_mask)

    # Laplacian of Gaussian (LoG): blur first, then Laplacian for border emphasis.
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    log_abs = cv2.convertScaleAbs(log)
    _, edge_binary = cv2.threshold(log_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Build contour map through binarization and color gating.
    primary_mask = cv2.bitwise_and(edge_binary, color_mask)

    kernel = np.ones((5, 5), np.uint8)
    primary_mask = cv2.morphologyEx(primary_mask, cv2.MORPH_OPEN, kernel)
    primary_mask = cv2.morphologyEx(primary_mask, cv2.MORPH_CLOSE, kernel)
    primary_mask = cv2.dilate(primary_mask, np.ones((3, 3), np.uint8), iterations=1)

    # Keep a fallback based on pure color contours if LoG gating becomes too strict.
    color_only_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_only_mask = cv2.morphologyEx(color_only_mask, cv2.MORPH_CLOSE, kernel)

    def collect_boxes(contours, circularity_thresh, axis_ratio_thresh):
        collected = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > area_limit:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)

            ellipse_like = False
            if len(contour) >= 5:
                (_, _), (major, minor), _ = cv2.fitEllipse(contour)
                if major > 0 and minor > 0:
                    axis_ratio = min(major, minor) / max(major, minor)
                    ellipse_like = axis_ratio >= axis_ratio_thresh

            circle_like = circularity >= circularity_thresh
            if not (ellipse_like or circle_like):
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 0.45 or aspect_ratio > 1.9:
                continue

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / hull_area
            if solidity < 0.62:
                continue

            # Reject dark/low-saturation regions (typical shadow artifacts).
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, -1)
            contour_pixels = cv2.countNonZero(contour_mask)
            if contour_pixels == 0:
                continue

            mean_sat = cv2.mean(sat_enhanced, mask=contour_mask)[0]
            mean_val = cv2.mean(val_enhanced, mask=contour_mask)[0]
            if mean_sat < 55 or mean_val < 50:
                continue

            color_pixels = cv2.countNonZero(cv2.bitwise_and(color_mask, contour_mask))
            color_ratio = color_pixels / float(contour_pixels)
            if color_ratio < 0.22:
                continue

            pad = max(4, int(0.08 * max(w, h)))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(width, x + w + pad)
            y2 = min(height, y + h + pad)
            collected.append((x1, y1, x2 - x1, y2 - y1, area, contour))
        return collected

    contours, _ = cv2.findContours(primary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = collect_boxes(contours, circularity_thresh=0.40, axis_ratio_thresh=0.50)
    if not boxes:
        fallback_contours, _ = cv2.findContours(color_only_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = collect_boxes(fallback_contours, circularity_thresh=0.25, axis_ratio_thresh=0.35)

    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detect_signs(frame)
        for (x, y, w, h, _, contour) in boxes[:3]:
            crop = frame[y:y+h, x:x+w]
            try:
                inp = preprocess(crop)
                pred = model.predict(np.expand_dims(inp, 0), verbose=0)
                class_id = int(np.argmax(pred, axis=1)[0])
                label = label_cod.get(class_id, str(class_id))
                display = f"{label}"
                cv2.putText(frame, display, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
            except Exception:
                pass
            cv2.drawContours(frame, [contour], -1, (0, 180, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow('Traffic sign recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
