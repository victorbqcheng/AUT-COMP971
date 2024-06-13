from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    # Train the model
    results = model.train(data='data.yaml', epochs=5, imgsz=640, plots=True)