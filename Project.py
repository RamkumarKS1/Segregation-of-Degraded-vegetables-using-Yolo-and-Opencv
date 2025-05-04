import cv2
import torch
import numpy as np
from ultralytics import YOLO  # Assuming YOLOv8

model = YOLO("C:/Users/Ramkumar K S/OneDrive/Desktop/Project Work/best.pt")  # Use a custom-trained model if available

class_labels = {0: 'Fresh Vegetable', 1: 'Degraded Vegetable'}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        labels = result.boxes.cls.cpu().numpy().astype(int)  # Class labels
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) if label == 0 else (0, 0, 255)  # Green for fresh, Red for degraded
            
    
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
        
            label_text = f"{class_labels.get(label, 'Unknown')}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    

    cv2.imshow("Vegetable Sorting", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
