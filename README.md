# Aerial Object Classification and Detection

This project is a deep learning-based system for identifying aerial objects as either a **bird** or a **drone**. It combines image classification and object detection in a simple Streamlit application so a user can upload an image and compare how different models respond.

## Live App

Streamlit Demo: https://koushik-gupta-ariel-object-detection-project-app-jzpzm0.streamlit.app/

## What This Project Does

The application performs three model-based checks on an uploaded image:

- **Custom CNN classification** to predict whether the image contains a bird or a drone
- **ResNet50 classification** for a stronger transfer learning-based prediction
- **YOLO detection** to detect and localize the object in the image with a bounding box

In short, this project does not only classify the image, but also shows object detection output for the same input.

## Main Idea

The goal of the project is to compare basic and advanced deep learning approaches for aerial object analysis:

- a simple CNN as the baseline model
- ResNet50 as the high-performance transfer learning model
- YOLO as the detection model

This makes the project useful for understanding the difference between:

- **classification**, where the model predicts the class label
- **detection**, where the model predicts the class and the object location

## Models Used

### 1. Custom CNN

A simple convolutional neural network built for binary image classification.

- Input: image resized to `224 x 224`
- Output classes: `bird` or `drone`
- Role: baseline classifier

### 2. ResNet50

A transfer learning model based on ResNet50 with a custom final layer for two classes.

- Input: image resized to `224 x 224`
- Output classes: `bird` or `drone`
- Role: best classification model in the project

### 3. YOLO

A YOLO model used for object detection.

- Input: uploaded image
- Output: detected object with bounding box and label
- Role: localization and detection

## Project Files

```text
Aerial Object Classification and Detection/
|-- app.py                                  # Streamlit application
|-- cnn_final.pth                           # Trained custom CNN weights
|-- resnet_final.pth                        # Trained ResNet50 weights
|-- best.pt                                 # Trained YOLO model
|-- bird_vs_drone_classifica tion.ipynb     # ipynb file
`-- README.md                               # Project documentation
```

## How the App Works

When an image is uploaded:

1. The image is loaded and displayed.
2. The CNN model predicts `bird` or `drone` with confidence.
3. The ResNet50 model predicts `bird` or `drone` with confidence.
4. The YOLO model runs detection and returns an output image with bounding boxes.

This allows a side-by-side comparison of classification and detection results for the same image.

## Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **Ultralytics YOLO**
- **Streamlit**
- **Pillow**

## Installation

Make sure Python is installed, then install the required libraries:

```bash
pip install streamlit torch torchvision ultralytics pillow
```

## Run the Project

From the project folder, run:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, upload an image, and view:

- CNN prediction
- ResNet50 prediction
- YOLO detection result

## Expected Output

For a given uploaded image, the app displays:

- the original image
- CNN predicted class with confidence score
- ResNet50 predicted class with confidence score
- YOLO detection image with bounding box visualization

## Use Cases

This kind of system can be useful for:

- airspace monitoring
- drone surveillance
- bird vs drone differentiation
- safety and security applications
- learning and comparing deep learning models

## Project Highlights

- Combines **classification** and **detection** in one project
- Uses both a **custom model** and **pretrained model**
- Includes a simple **web interface**
- Good for both **demo** and **learning purposes**

## Future Improvements

- Add a `requirements.txt` file
- Add batch image prediction
- Show model confidence comparison in charts
- Improve UI design for better presentation
- Add dataset and training scripts for full reproducibility

## Conclusion

This project is a complete bird-vs-drone analysis system built with deep learning. It demonstrates how a custom CNN, a transfer learning model, and a YOLO detector can be used together in one pipeline. The final Streamlit app makes the system easy to test with real images and clearly shows the difference between classification and detection outputs.
