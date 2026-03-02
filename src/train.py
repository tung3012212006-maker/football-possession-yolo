from ultralytics import YOLO


def train_football_model():
    """
    Fine-tune a pretrained YOLOv11 model on the custom football dataset.
    """

    # 1. Initialize the pretrained YOLOv11 model (medium version)
    model = YOLO("yolo11m.pt")

    # 2. Start fine-tuning on the custom dataset
    results = model.train(
        data=".../football-possession-yolo/data/track-football-player--4/data.yaml",  # Path to dataset config
        epochs=100,                 # Number of training epochs
        imgsz=640,                  # Input image size
        batch=16,                   # Batch size (adjust based on GPU memory)
        device=0,                   # Use GPU (0) or 'cpu' if no GPU is available
        project="models",           # Directory to save training outputs
        name="yolo11_fine_tuned",   # Name of this training run
        patience=20,                # Early stopping if no improvement
        save=True                   # Save best model weights
    )

    print(
        "Training Complete! "
        "Optimized model weights saved at: "
        "models/yolo11_fine_tuned/weights/best.pt"
    )


if __name__ == "__main__":
    train_football_model()
