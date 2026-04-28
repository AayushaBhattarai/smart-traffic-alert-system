import cv2
from model_init import initialize_yolov11

# Initialize YOLOv11 model
model = initialize_yolov11()

cap = cv2.VideoCapture("data/videos/traffic.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLOv11 Vehicle Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
