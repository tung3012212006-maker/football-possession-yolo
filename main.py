import cv2
import argparse
from src.tracker import FootballTracker
from src.possession import PossessionEstimator

def run_analysis(video_source, model_weights):
    # 1. Initialize Modules
    tracker = FootballTracker(model_weights)
    possession_strategy = PossessionEstimator(proximity_threshold=60)
    
    # 2. Setup Video Capture
    cap = cv2.VideoCapture(video_source)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))  
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"--- Processing Video: {video_source} ---")
    print(f"--- Resolution: {frame_width}x{frame_height} | FPS: {fps} ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Perform Tracking (Detection + ID Assignment)
        results = tracker.track_players(frame)

        # 4. Calculate Possession
        possessor_id = possession_strategy.get_player_possession(results)

        # 5. Visualization
        # Draw all tracking boxes
        annotated_frame = tracker.draw_tracks(frame, results)

        # Highlight the player in possession
        if possessor_id is not None:
            cv2.putText(annotated_frame, f"POSSESSION: Player #{possessor_id}", 
                        (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
            # Optional: Draw a specific circle under the possessor's feet
            # (Requires logic to find the specific box of possessor_id)

        # 6. Display Output
        cv2.imshow('Football AI Analytics System', annotated_frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- Analysis Completed ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football Analytics Main Entry")
    parser.add_argument("--source", type=str, default="/Users/macbook/football-possession-yolo/data/track-football-player--4/test_match.mp4", help="Path to video")
    parser.add_argument("--weights", type=str, default="/Users/macbook/football-possession-yolo/models/yolo11_fine_tuned/weights/best.pt", help="Path to YOLO weights")
    
    args = parser.parse_args()
    run_analysis(args.source, args.weights)