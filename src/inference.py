from ultralytics import YOLO
import cv2

# Load model
model = YOLO(".../football-possession-yolo/models/yolo11_fine_tuned/weights/best.pt")

# Đọc ảnh
image_path = ".../football-possession-yolo/data/track-football-player--4/test/images"
results = model.predict(image_path, conf=0.5)

# Lấy ảnh đã vẽ bounding box
annotated = results[8].plot()

# Hiển thị
cv2.imshow("Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()