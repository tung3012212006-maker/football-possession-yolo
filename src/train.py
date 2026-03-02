from ultralytics import YOLO

def train_football_model():
    # 1. Khởi tạo model YOLOv11 
    model = YOLO("yolo11m.pt") 
    # 2. Tiến hành Fine-tuning
    results = model.train(
        data=".../football-possession-yolo/data/track-football-player--4/data.yaml", # Đường dẫn file yaml
        epochs=100,                              # Số lượt train
        imgsz=640,                               # Kích thước ảnh
        batch=16,                                # Tùy vào bộ nhớ GPU (8, 16, 32)
        device=0,                                # 0 cho GPU, 'cpu' nếu không có GPU
        project="models",                        # Thư mục lưu kết quả
        name="yolo11_fine_tuned",                # Tên phiên bản train
        patience=20,                             # Early stopping nếu không cải thiện
        save=True                                # Lưu lại trọng số (.pt) tốt nhất
    )
    
    print("Training Complete! > The optimized model weights have been saved to: models/yolo11_fine_tuned/weights/best.pt")

if __name__ == "__main__":
    train_football_model()