from ultralytics import YOLO
import supervision as sv
import numpy as np

class PlayerBallDetector:
    def __init__(self, model_path='models/football/yolo11_fine_tuned/weights/best.pt'):
        """
        Khởi tạo model YOLOv11 đã fine-tuned.
        """
        # Load trọng số model tốt nhất sau khi train
        self.model = YOLO(model_path)
        # Ánh xạ class name từ dataset của bạn
        self.class_names = ['ARb', 'FCB', 'RMA', 'bal', 'gar FSB', 'garRMA']
        # Lưu ID của quả bóng để dùng cho logic Ball Possession sau này
        self.ball_class_id = self.class_names.index('bal')
        # ID của các nhóm cầu thủ/trọng tài (tất cả trừ quả bóng)
        self.player_class_ids = [i for i, name in enumerate(self.class_names) if name != 'bal']
    def get_detections(self, frame):
        """
        Thực hiện inference và trả về kết quả định dạng Supervision
        """
        results = self.model.predict(frame, conf=0.3, imgsz=640, verbose=False)[0]
        
        # Chuyển đổi sang Supervision
        detections = sv.Detections.from_ultralytics(results)
        
        return detections

    def get_separated_detections(self, detections):
        """
        Tách riêng quả bóng và các cầu thủ để xử lý logic
        """
        # Lấy tất cả những gì không phải là bóng
        players = detections[np.isin(detections.class_id, self.player_class_ids)]
        
        # Lấy riêng quả bóng
        ball = detections[detections.class_id == self.ball_class_id]
        
        return players, ball

    def get_player_crops(self, frame, detections):
        """
        Cắt ảnh các cầu thủ (để phân tích sâu hơn nếu cần)
        """
        player_crops = []
        player_detections = detections[np.isin(detections.class_id, self.player_class_ids)]
        
        for xyxy in player_detections.xyxy:
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            player_crops.append(crop)
            
        return player_crops