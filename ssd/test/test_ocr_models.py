import time
import torch
import easyocr
import numpy as np
from tqdm import tqdm
from paddleocr import PaddleOCR
import cv2
from torchvision import datasets

gpu = False
if torch.cuda.is_available(): gpu = True

dataset_root = '/Users/joe/Code/third-year-project/ANPR/classifier/val'
validation_dataset = datasets.ImageFolder(root=dataset_root)

def test_ocr(validation_dataset):

    reader = easyocr.Reader(['en'], gpu=gpu)
    start = time.time()
    image_count = 0

    for i, sample in enumerate(validation_dataset):
        image = sample[0]
        img = np.array(image)
        bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        reader.readtext(bgr)
        image_count += 1
        if i == 50:
            break

    end = time.time() - start
    return end * 1000 / image_count


average_time_per_image = test_ocr(validation_dataset)
print(f'Average time per image: {round(average_time_per_image, 2)} ms')

def test_paddle(validation_dataset):
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    start = time.time()
    image_count = 0
    for i, sample in enumerate(tqdm(validation_dataset)):
        image = sample[0]
        img = np.array(image)
        bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ocr.ocr(bgr)
        image_count += 1
        if i == 50:
            break

    end = time.time() - start
    return end * 1000 / image_count

average_time_per_image = test_paddle(validation_dataset)
print(f'Average time per image: {round(average_time_per_image, 2)} ms')