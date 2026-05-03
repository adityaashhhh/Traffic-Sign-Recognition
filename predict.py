import argparse
import cv2
import numpy as np
import tensorflow as tf

# label code mapping (same as in recognition.py)
label_cod = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Give Way', 14: 'Stop', 15: 'No vehicles', 16: 'Veh > 3.5 tons prohibited',
    17: 'No entry', 18: 'General caution', 19: 'Dangerous curve left', 20: 'Dangerous curve right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}


def preprocess_image(path: str) -> np.ndarray:
    """Load an image file and preprocess it to the shape expected by the model."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Unable to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (30, 30))
    img = img / 255.0
    return img


def main():
    parser = argparse.ArgumentParser(description="Predict traffic sign class for one or more images.")
    parser.add_argument("paths", nargs="+", help="Path(s) to image file(s) to classify")
    parser.add_argument("--model", default="traffic_model.keras", help="Path to the trained Keras model file")
    args = parser.parse_args()

    # load the model
    model = tf.keras.models.load_model(args.model)

    for p in args.paths:
        try:
            img_arr = preprocess_image(p)
        except FileNotFoundError as e:
            print(e)
            continue
        # model expects batch dimension
        pred = model.predict(np.expand_dims(img_arr, 0))
        class_id = int(np.argmax(pred, axis=1)[0])
        label = label_cod.get(class_id, f"Unknown ({class_id})")
        print(f"{p} -> class {class_id}: {label}")


if __name__ == "__main__":
    main()
