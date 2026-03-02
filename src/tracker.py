from ultralytics import YOLO
import cv2

class FootballTracker:
    def __init__(self, model_path):
        # Load the trained YOLOv11 model
        self.model = YOLO(model_path)
        # Define the tracker configuration (bytetrack.yaml or botsort.yaml)
        self.tracker_type = "bytetrack.yaml"

    def track_players(self, frame):
        """
        Performs tracking on a single frame.
        Returns: list of results containing bounding boxes and track IDs.
        """
        # persist=True is crucial for maintaining IDs across frames
        results = self.model.track(
            source=frame, 
            persist=True, 
            tracker=self.tracker_type,
            conf=0.5,
            iou=0.6,
            device='mps' # Optimized for MacBook GPU
        )
        return results[0]

    def draw_tracks(self, frame, results):
        """
        Draws bounding boxes and Track IDs on the frame for visualization.
        """
        # Check if any objects were detected/tracked
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            names = self.model.names

            for box, id, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                label = f"{names[cls]} #{id}"
                
                # Draw Rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw Label Background
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame