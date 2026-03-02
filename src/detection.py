from ultralytics import YOLO
import supervision as sv
import numpy as np

class PlayerBallDetector:
    def __init__(self, model_path='models/football/yolo11_fine_tuned/weights/best.pt'):
        """
        Initialize the fine-tuned YOLOv11 model.
        """
        # Load the best trained weights
        self.model = YOLO(model_path)

        # Class names defined in the custom dataset
        self.class_names = ['ARb', 'FCB', 'RMA', 'bal', 'gar FSB', 'garRMA']

        # Store the class ID of the ball for possession logic
        self.ball_class_id = self.class_names.index('bal')

        # Store class IDs for all non-ball objects (players, referees, goalkeepers)
        self.player_class_ids = [
            i for i, name in enumerate(self.class_names) if name != 'bal'
        ]

    def get_detections(self, frame):
        """
        Run YOLO inference on a frame and return results 
        formatted as Supervision Detections.
        """
        results = self.model.predict(
            frame,
            conf=0.3,
            imgsz=640,
            verbose=False
        )[0]

        # Convert Ultralytics results to Supervision format
        detections = sv.Detections.from_ultralytics(results)

        return detections

    def get_separated_detections(self, detections):
        """
        Separate player detections and ball detections 
        for downstream logic (e.g., possession estimation).
        """
        # All detections except the ball
        players = detections[
            np.isin(detections.class_id, self.player_class_ids)
        ]

        # Only the ball detection(s)
        ball = detections[
            detections.class_id == self.ball_class_id
        ]

        return players, ball

    def get_player_crops(self, frame, detections):
        """
        Extract cropped images of detected players.
        Useful for further analysis such as jersey classification 
        or feature extraction.
        """
        player_crops = []

        player_detections = detections[
            np.isin(detections.class_id, self.player_class_ids)
        ]

        for xyxy in player_detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            player_crops.append(crop)

        return player_crops
