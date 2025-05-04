import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load your custom-trained YOLOv8 model
model = YOLO("C:/Users/Ramkumar K S/OneDrive/Desktop/Project Work/best.pt")  # Replace with actual model path

# Class labels
class_labels = {0: 'Fresh Vegetable', 1: 'Degraded Vegetable'}

# Function: Preprocess the frame (Gaussian blur + CLAHE + resize to 416x416)
def preprocess_frame(frame):
    # Convert to LAB color space for CLAHE
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged_lab = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # Apply Gaussian Blur for noise reduction
    blurred_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)

    return blurred_frame

# Initialize camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the captured frame
    processed_frame = preprocess_frame(frame)

    # Resize to 416x416 (YOLO input size)
    resized_frame = cv2.resize(processed_frame, (416, 416))

    # Run YOLO inference (original frame must be passed to display output)
    results = model(resized_frame)[0]

    # Extract predictions
    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    labels = results.boxes.cls.cpu().numpy().astype(int)

    # Apply NMS manually (optional: YOLO already does this)
    indices = cv2.dnn.NMSBoxes(
        bboxes=[box.tolist() for box in boxes],
        scores=scores.tolist(),
        score_threshold=0.25,
        nms_threshold=0.45
    )

    for i in indices:
        i = i[0] if isinstance(i, (list, np.ndarray)) else i
        x1, y1, x2, y2 = map(int, boxes[i])
        label = labels[i]
        score = scores[i]

        # Choose color
        color = (0, 255, 0) if label == 0 else (0, 0, 255)

        # Scale coordinates back to original frame size if needed
        h_ratio = frame.shape[0] / 416
        w_ratio = frame.shape[1] / 416
        x1 = int(x1 * w_ratio)
        y1 = int(y1 * h_ratio)
        x2 = int(x2 * w_ratio)
        y2 = int(y2 * h_ratio)

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{class_labels.get(label, 'Unknown')}: {score:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show result
    cv2.imshow("Vegetable Sorting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
