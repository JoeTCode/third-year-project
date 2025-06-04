from ultralytics import YOLO

# model = YOLO("yolov8n.yaml") # not pretrained
model = YOLO("/Users/joe/Code/third-year-project/ANPR/yolov8n/weights/run9_best.pt") # pretrained
# results = model.train(data="config.yaml", epochs=25, patience=5)
results = model.val(data='config.yaml')