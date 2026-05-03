# Traffic Sign Recognition System

## Overview

This project implements a Traffic Sign Recognition System using a combination of Computer Vision (OpenCV) and Deep Learning (CNN). The system is capable of detecting and classifying traffic signs from images, video files, and real-time webcam streams.

The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset and supports classification of 43 traffic sign classes.

---

## Features

* Real-time traffic sign detection using webcam
* Image classification from static input
* Video processing using uploaded MP4 files
* Flask-based web interface
* Hybrid system combining OpenCV detection and CNN classification
* Support for multiple model formats (.h5, .keras, .tflite)

---

## Technology Stack

* Programming Language: Python
* Libraries:

  * TensorFlow / Keras
  * OpenCV
  * NumPy, Pandas
  * Flask
* Dataset: GTSRB

---

## System Architecture

The system follows a structured pipeline:

Input (Image / Video / Webcam)
→ Detection (OpenCV - HSV filtering and contour detection)
→ Region of Interest (ROI extraction)
→ Preprocessing (Resize to 30×30 and normalization)
→ CNN Model (43-class classification)
→ Prediction Output

---

## Detection Methodology

The system uses classical computer vision techniques to detect traffic signs before classification:

* HSV color filtering for red and blue regions
* Edge detection using Laplacian operator
* Morphological operations (opening and closing)
* Contour detection and filtering
* Shape validation using circularity and aspect ratio

---

## Model Details

* Input Size: 30 × 30 × 3
* Number of Classes: 43
* Model Type: Convolutional Neural Network (CNN)
* Output Layer: Softmax for multi-class classification

---

## Project Structure

```
project/
│
├── app.py               # Flask web application
├── realtime.py          # Real-time detection using webcam
├── predict.py           # Image prediction script
├── ModelTrain.py        # Model training and evaluation
│
├── traffic_model.keras  # Trained model
├── traffic_sign.h5      # Alternate model
├── labels.pkl           # Label mapping
│
├── templates/           # HTML templates for UI
├── README.md
└── .gitignore
```

---

## Installation and Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/traffic-sign-recognition.git
cd traffic-sign-recognition
```

---

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

### 3. Run the Flask application

```
python app.py
```

Open the application in a browser:

```
http://127.0.0.1:5000/
```

---

### 4. Run real-time detection

```
python realtime.py
```

---

### 5. Run image prediction

```
python predict.py path_to_image.jpg
```

---

## Results

* Accuracy: __________
* The system performs efficient real-time detection with low latency
* Model performance is evaluated using a confusion matrix

---

## Applications

* Autonomous driving systems
* Driver assistance systems
* Traffic monitoring and surveillance
* Smart transportation systems

---

## Limitations

* Performance may degrade under poor lighting conditions
* Detection relies on color-based filtering
* Limited to the trained set of 43 traffic sign classes

---

## Future Scope

* Integration of advanced detection models such as YOLO
* Deployment as a mobile application using TensorFlow Lite
* Improved robustness for real-world environments
* Support for multiple object detection

---

## Author

Aditya Sharma

---

## Acknowledgements

* GTSRB Dataset
* TensorFlow and OpenCV documentation
