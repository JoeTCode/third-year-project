from test_ssd_inference import model
from ssd.custom_yolo_dataset_loader import AnprYoloDataset, validation_transform, testing_transform, Resize
from config import config
from torchvision import transforms
import torch
from PIL import Image, ImageDraw
from ssd.show_predictions import crop_numberplate, preprocess_image
import easyocr
import numpy as np

valid_dataset = AnprYoloDataset(
    annotations_root=config.VALID_ANNOTATIONS_ROOT,
    images_root=config.VALID_IMAGES_ROOT,
    transform=validation_transform
)

valid_dataset_without_norm = AnprYoloDataset(
    annotations_root=config.VALID_ANNOTATIONS_ROOT,
    images_root=config.VALID_IMAGES_ROOT,
    transform=testing_transform
)

valid_dataset_test = AnprYoloDataset(
    annotations_root=config.VALID_ANNOTATIONS_ROOT,
    images_root=config.VALID_IMAGES_ROOT,
    test=True
)

model.eval()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

denormalize = transforms.Normalize(
    mean=-torch.tensor(mean) / torch.tensor(std),
    std=1.0 / torch.tensor(std)
)

def test_ocr():
    reader = easyocr.Reader(['en'], gpu=False)
    original = Image.open("/Users/joe/Code/third-year-project/ANPR/ssd/test/ocr-test/original-crop-with-resized-bboxes.png")
    resized = Image.open("/Users/joe/Code/third-year-project/ANPR/ssd/test/ocr-test/original-resized-300x300-crop.png")

    np_original = preprocess_image(original)
    pil = Image.fromarray(np_original)
    pil.show()
    # np_original = np.array(original)
    # np_resized = np.array(resized)

    # Original(Higher quality):
    # [([[0, 6], [150, 6], [150, 127], [0, 127]], 'HhU', 0.11308909988423067)]
    original_detections = reader.readtext(np_original)
    print("Original (Higher quality):")
    print(original_detections)

    # resized_detections = reader.readtext(np_resized)
    # print("Resized (Lower quality):")
    # print(resized_detections)

test_ocr()

def test_crop():
    for i, resized_sample in enumerate(valid_dataset):

        resized_image_tensor, annotations = resized_sample[0], resized_sample[1]
        print(annotations)

        # SHOW ORIGINAL UNPROCESSED DATASET IMAGE
        image_id = annotations['image_id']
        dataset_image = valid_dataset_test[image_id]
        dataset_image.show()
        print(dataset_image.size)

        # DENORMALIZE
        denormalized_image_tensor = denormalize(resized_image_tensor)
        resized_image = transforms.ToPILImage()(denormalized_image_tensor)  # 300x300
        print(resized_image.size)

        # DRAW BBOXES (IN GREEN) AND SHOW 300X300 IMAGE
        resized_copy = resized_image.copy()
        resized_draw = ImageDraw.Draw(resized_copy)

        for bbox in annotations['boxes']:

            crop = crop_numberplate(resized_copy, bbox)
            resized_crop = crop.resize((150, 150), Image.Resampling.LANCZOS)
            resized_crop.show()

            min_x, min_y, max_x, max_y = bbox
            resized_draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=1)

        resized_copy.show()

        # RESIZE TO PROCESSED IMAGE TO 640x640 TO GET RESIZED BBOXES. DRAW BBOXES (IN RED) OVER ORIGINAL UNPROCESSED DATASET IMAGE
        # During eval, take image fed to model, take annotations generated from model. Then resize both before performing inference
        resized_to_original_sample = Resize((640, 640))({"image": resized_image, "annotations": annotations})
        # DISCARD RESIZED IMAGE AS IT IS LOW QUALITY
        _, original_sized_annotations = resized_to_original_sample["image"], resized_to_original_sample["annotations"]

        # TAKE RESIZED (640x640) BBOXES AND PASTE IT OVER THE ORIGINAL UNPROCESSED DATASET IMAGE
        original_image_draw = ImageDraw.Draw(dataset_image)

        for bbox in original_sized_annotations['boxes']:

            crop = crop_numberplate(dataset_image, bbox)
            resized_crop = crop.resize((150, 150), Image.Resampling.LANCZOS)
            resized_crop.show()

            min_x, min_y, max_x, max_y = bbox
            original_image_draw.rectangle([min_x, min_y, max_x, max_y], outline="blue", width=1)
        dataset_image.show()

        if i == 0:
            break
