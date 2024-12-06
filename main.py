from ultralytics import YOLO

# model = YOLO("yolov8n.yaml") # not pretrained
model = YOLO("yolov8n.pt") # pretrained

results = model.train(data="config.yaml", epochs=10)