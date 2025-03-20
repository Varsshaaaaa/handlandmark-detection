import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, UpSampling2D, BatchNormalization, Input
from tensorflow.keras.applications import MobileNetV2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Function to create a heatmap (Fixed size to match model output)
def create_heatmap(img_size, landmarks, output_size=16, sigma=2):
    heatmap = np.zeros((output_size, output_size, len(landmarks)))
    scale = output_size / img_size  # Scale factor for resizing

    for i, (x, y) in enumerate(landmarks):
        x, y = int(x * output_size), int(y * output_size)  # Scale to new size
        if 0 <= x < output_size and 0 <= y < output_size:
            for dx in range(-sigma, sigma + 1):
                for dy in range(-sigma, sigma + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < output_size and 0 <= ny < output_size:
                        heatmap[ny, nx, i] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
    
    return heatmap

# Define CNN model (Ensured output matches heatmap size 16x16)
def build_heatmap_model(input_shape=(128, 128, 3), num_landmarks=21):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = True  

    x = Conv2D(256, (3, 3), padding="same", activation="relu")(base_model.output)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)

    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = UpSampling2D(size=(2, 2))(x)  # Output now matches 16x16 heatmap

    x = Conv2D(num_landmarks, (3, 3), activation="sigmoid", padding="same")(x)

    model = Model(inputs=base_model.input, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Build model
model = build_heatmap_model()

# Collect Hand Data (Fixed to correctly generate 16x16 heatmaps)
def capture_hand_data(num_samples=1000, input_shape=(128, 128, 3), img_size=16):
    images, heatmaps = [], []
    cap = cv2.VideoCapture(0)

    while len(images) < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm_list = [[lm.x, lm.y] for lm in hand_landmarks.landmark]

                resized_frame = cv2.resize(frame, (input_shape[0], input_shape[1])) / 255.0
                heatmap = create_heatmap(img_size, lm_list)

                images.append(resized_frame)
                heatmaps.append(heatmap)

                # Draw landmarks for debugging
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Collecting Data', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(images), np.array(heatmaps)

# Collect data
X_train, y_train = capture_hand_data()

# Train the model
model.fit(X_train, y_train, batch_size=16, epochs=20, validation_split=0.1)

# Decode heatmap to extract keypoints
def extract_keypoints(heatmap):
    keypoints = []
    for i in range(heatmap.shape[-1]):
        y, x = np.unravel_index(np.argmax(heatmap[:, :, i]), heatmap[:, :, i].shape)
        keypoints.append((x / heatmap.shape[1], y / heatmap.shape[0]))  # Normalize
    return keypoints

# Real-time hand tracking with heatmap-based keypoints
def real_time_hand_tracking():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        input_img = cv2.resize(frame, (128, 128)) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        heatmap = model.predict(input_img)[0]
        keypoints = extract_keypoints(heatmap)

        for (x, y) in keypoints:
            x_pixel, y_pixel = int(x * frame.shape[1]), int(y * frame.shape[0])
            cv2.circle(frame, (x_pixel, y_pixel), 5, (0, 255, 0), -1)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run real-time tracking
real_time_hand_tracking()
