import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

# Define the MeanIoU metric
m_iou = tf.keras.metrics.MeanIoU(num_classes=2)

# Define the model path
model_path = r"C:\Users\NA\Documents\Free Code Camp\my_trained_model.h5"

# Load and compile the model
model = tf.keras.models.load_model(model_path, compile=False)
model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[m_iou])

print("Model loaded and recompiled successfully!")

# Define the video file path
video_path = r"testing\challenge.mp4"
output_path = r"C:\Users\NA\Documents\Free Code Camp\output_video1.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

# Process the video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Preprocess frame for the model
    input_frame = cv2.resize(frame, (128, 128))
    input_frame = input_frame / 255.0
    input_frame = np.expand_dims(input_frame, axis=0)

    # Predict segmentation mask
    prediction = model.predict(input_frame)
    mask = (prediction[0] > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (frame_width, frame_height))
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Overlay mask on original frame
    overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

    # Write to output video
    out.write(overlay)

    frame_count += 1
    print(f"Processed frame {frame_count}")

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. {frame_count} frames written to {output_path}")