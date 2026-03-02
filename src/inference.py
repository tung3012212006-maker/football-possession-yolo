from ultralytics import YOLO
import cv2

# Load the fine-tuned YOLO model
model = YOLO(".../football-possession-yolo/models/yolo11_fine_tuned/weights/best.pt")

# Path to a directory containing test images
image_path = ".../football-possession-yolo/data/track-football-player--4/test/images"

# Run inference on all images inside the folder
results = model.predict(image_path, conf=0.5)

# Select one prediction result (e.g., the 9th image)
annotated = results[8].plot()

# Display the image with bounding boxes
cv2.imshow("Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
