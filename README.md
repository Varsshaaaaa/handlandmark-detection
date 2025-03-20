# Hand Landmark Detection Using CNN & Heatmaps

## Overview
This project implements a hand landmark detection model using a convolutional neural network (CNN) trained to generate heatmaps for keypoints. The model is built using TensorFlow's MobileNetV2 as a feature extractor and is trained to predict 21 hand landmarks. The dataset is collected in real-time using OpenCV and MediaPipe.

## Models Used

### 1. **MobileNetV2 (Backbone Feature Extractor)**
- MobileNetV2 is a lightweight deep neural network designed for efficient feature extraction.
- It is used here as a base model to process input images and extract meaningful representations.
- The fully connected layers are replaced with a custom heatmap generation head.

### 2. **Custom CNN for Heatmap Prediction**
- The output of MobileNetV2 is passed through a series of convolutional layers.
- The final layer outputs a 16x16 heatmap for each of the 21 hand landmarks.
- The heatmaps represent the probability distribution of the keypoints, which are extracted using a simple argmax operation.

## Training Approach

### 1. **Data Collection**
- The dataset is collected using OpenCV and MediaPipe's hand-tracking solution.
- Images are captured from the webcam and resized to **128x128 pixels**.
- Ground truth keypoints are extracted from MediaPipe and converted into 16x16 heatmaps using a Gaussian distribution.

### 2. **Preprocessing**
- The images are normalized to a range of [0, 1].
- Heatmaps are generated for each hand landmark and used as labels.

### 3. **Model Training**
- The model is trained using **Mean Squared Error (MSE)** as the loss function, ensuring heatmap accuracy.
- **Adam optimizer** is used with a learning rate of 0.001.
- The dataset is split into **90% training and 10% validation**.
- The model is trained for **20 epochs** with a batch size of **16**.

## Key Functions

### 1. **create_heatmap(img_size, landmarks, output_size=16, sigma=2)**
- Converts keypoint locations into a Gaussian heatmap representation.

### 2. **build_heatmap_model()**
- Defines the CNN model architecture based on MobileNetV2.

### 3. **capture_hand_data(num_samples=1000)**
- Captures and processes hand images from a webcam.
- Generates corresponding heatmaps for model training.

### 4. **extract_keypoints(heatmap)**
- Extracts the most likely keypoints from predicted heatmaps.

### 5. **real_time_hand_tracking()**
- Runs real-time hand tracking using the trained model.

## Results & Future Improvements
- The model successfully predicts hand landmarks without using pre-trained MediaPipe models.
- Improvements can include:
  - Increasing the resolution of heatmaps for better accuracy.
  - Training with a larger dataset to improve generalization.
  - Experimenting with different CNN architectures for better performance.
  
![Screenshot 2025-03-20 175924](https://github.com/user-attachments/assets/cb84e822-7a75-4d6c-bde6-0cfe482feeef)



