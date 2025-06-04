MAX_ANNOTATIONS = 5
BBOX_LENGTH = 4
BBOX_WEIGHT = 1
LABELS_WEIGHT = 1
EPOCHS = 1
HPC = False
NUM_LOGS = 10
MAP_MIN_DIFFERENCE = 0.01
PATIENCE = 5
BATCH_SIZE = 4
TRAIN_IMAGES_ROOT = "/Users/joe/Desktop/eu-dataset/train/images"
TRAIN_ANNOTATIONS_ROOT = "/Users/joe/Desktop/eu-dataset/train/labels"
VALID_IMAGES_ROOT = "/Users/joe/Desktop/eu-dataset/valid/images"
VALID_ANNOTATIONS_ROOT = "/Users/joe/Desktop/eu-dataset/valid/labels"
TEST_IMAGES_ROOT = "/Users/joe/Desktop/eu-dataset/test/images"
TEST_ANNOTATIONS_ROOT = "/Users/joe/Desktop/eu-dataset/test/labels"
VERBOSE = False
SAVE_IMAGE_DIRECTORY = "/Users/joe/Code/third-year-project/ANPR/ssd/prediction_images"
FONT_SIZE=20
TRAIN_MOSAIC_PROBABILITY = 0.5

# CHANGE THESE accordingly to allow the ANPR system to work
FONT_PATH="/Users/joe/Code/third-year-project/ANPR/fonts/DejaVuSans.ttf"
YOLO_WEIGHT="/Users/joe/Code/third-year-project/ANPR/yolov8n/weights/run9_best.pt"
YOLO_SAVE_OCR_IMAGE_DIR='/Users/joe/Code/third-year-project/ANPR/backend/ocr-image'
MOBILENET_V2_WEIGHTS='/Users/joe/Code/third-year-project/ANPR/classifier/save_weights/mobilenet_v2_weights_1.pth'