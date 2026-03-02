import numpy as np

class  PossessionEstimator:
    def __init__(self, proximity_threshold=40):
        """
        Initialize the Possession Estimator.
        :param proximity_threshold: Max distance (in pixels) to consider a player has the ball.
        """
        self.proximity_threshold = proximity_threshold
        self.current_possession_id = None

    def calculate_distance(self, p1, p2):
        """Calculates Euclidean distance between two points (x, y)."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def get_player_possession(self, results):
        """
        Logic to determine which player ID is in possession of the ball.
        :param results: YOLO tracking results for a single frame.
        :return: track_id of the player in possession, or None.
        """
        if results.boxes.id is None:
            return None

        # Extract boxes, IDs, and Classes
        boxes = results.boxes.xyxy.cpu().numpy()
        ids = results.boxes.id.cpu().numpy().astype(int)
        clss = results.boxes.cls.cpu().numpy().astype(int)
        
        ball_center = None
        players = [] # List of (player_id, center_point)

        # 1. Identify Ball and Players positions
        for box, id, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            
            if cls == 3: # Assuming Class 3 is 'ball'
                ball_center = center
            else: # Assuming Class else is 'player'
                # Use bottom-center (feet) for players instead of center for better accuracy
                feet_pos = [(x1 + x2) / 2, y2]
                players.append((id, feet_pos))

        # 2. Find the closest player to the ball
        if ball_center is None:
            return self.current_possession_id # Keep last known possession if ball is lost

        min_dist = float('inf')
        assigned_player_id = None

        for player_id, player_feet in players:
            dist = self.calculate_distance(ball_center, player_feet)
            
            if dist < min_dist and dist < self.proximity_threshold:
                min_dist = dist
                assigned_player_id = player_id

        self.current_possession_id = assigned_player_id
        return assigned_player_id