# EU: 0, Non-EU: 1
import os
import time

import cv2
import easyocr
from paddleocr import PaddleOCR
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image
from torchvision import datasets, transforms, models
from torchvision.transforms import ToPILImage
import uuid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu = False
if device == 'cuda':
    gpu = True

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

validation_dataset = datasets.ImageFolder(root='val', transform=transform)

model = models.mobilenet_v2()

# Modify classification head
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load('save_weights/mobilenet_v2_weights_1.pth'))
model.eval()


def unnormalise(img_tensor, mean, std):
    # Convert mean and std to tensors for broadcasting
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    # Undo normalization: img = img * std + mean
    img_tensor = img_tensor * std + mean

    # Clamp to [0,1]
    img_tensor = torch.clamp(img_tensor, 0, 1)

    return img_tensor

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def evaluate():
    start = time.time()
    correct = 0
    with torch.no_grad():
        for image, label in validation_dataset:
            output = model(image.unsqueeze(0).to(device))
            _, predicted = torch.max(output, 1)
            class_idx = predicted.item()

            classes = ['EU', 'Non-EU']
            print(f"Prediction: {classes[class_idx]}, Actual: {classes[label]}")
            if class_idx != label:
                unnormalised = unnormalise(image, mean, std)
                img = ToPILImage()(unnormalised)
                img_draw = ImageDraw.Draw(img)

                font = ImageFont.truetype('/Users/joe/Code/third-year-project/ANPR/fonts/DejaVuSans.ttf', 20)
                text = f"P: {classes[class_idx]}, A: {classes[label]}"
                img_draw.text((10, 10), text, fill="white", font=font)

                img.show()
            if class_idx == label:
                correct += 1

    total = time.time() - start
    average_time_per_image = total / len(validation_dataset)
    print(f'Average classification time per image: {average_time_per_image*1000:.3f} ms')
    print(f'Accuracy: {correct*100/len(validation_dataset):.3f}%')


def calculate_correct(predicted_text, actual_text, np_img=None):
    correct = 0
    # format actual text
    actual_text = actual_text.replace('_', '').lower()
    # format predicted text
    predicted_text = predicted_text.replace(' ', '').replace('.', '').replace('-', '').lower()

    print('Actual:', actual_text)
    print('Predicted:', predicted_text)

    total = len(actual_text)

    for i in range(min(len(predicted_text), total)):
        if predicted_text[i] == actual_text[i]:
            correct += 1

    if np_img is not None:
        if correct == 0 or len(predicted_text) > len(actual_text) * 2:
            id = uuid.uuid1()
            cv2.imwrite(f'/Users/joe/Desktop/easyocr-failed-images/{id}.png', np_img)

    return correct, total


# Accuracy achieved: 87.566988%
# Time per image: 529.21 ms

def test_paddle(image_directory):
    image_directory_list = os.listdir(image_directory)
    image_directory_list = [image for image in image_directory_list if not image.startswith('.')]

    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    total = 0
    correct = 0
    start = time.time()

    for i, image_name in enumerate(image_directory_list):
        image_filepath = os.path.join(image_directory, image_name)

        np_img = cv2.imread(image_filepath)
        detections = ocr.ocr(np_img)
        number_plate_text = ''

        if detections[0] is not None:
            for i in range(len(detections)):
                detection = detections[i]
                for det in detection:
                    if len(det) >= 2:
                        number_plate_text += det[1][0]

        gt, _ = os.path.splitext(image_name)
        counts = calculate_correct(number_plate_text, gt)
        correct += counts[0]
        total += counts[1]

    total_time = time.time() - start
    average_time_per_image = total_time / len(image_directory_list)

    return correct/total, average_time_per_image


# Accuracy achieved: 49.624866%
# Time per image: 567.99 ms

def test_easyocr(image_directory):
    image_directory_list = os.listdir(image_directory)
    image_directory_list = [image for image in image_directory_list if not image.startswith('.')]

    reader = easyocr.Reader(['en'], gpu=gpu)
    total = 0
    correct = 0
    start = time.time()

    for i, image_name in enumerate(image_directory_list):
        image_filepath = os.path.join(image_directory, image_name)

        np_img = cv2.imread(image_filepath)
        detections = reader.readtext(np_img)
        number_plate_text = ''

        for detection in detections:
            number_plate_text += detection[1]

        gt, _ = os.path.splitext(image_name)
        counts = calculate_correct(number_plate_text, gt, np_img)
        correct += counts[0]
        total += counts[1]

    total_time = time.time() - start
    average_time_per_image = total_time / len(image_directory_list)

    return correct/total, average_time_per_image

eu_val_directory = '/Users/joe/Code/third-year-project/ANPR/classifier/val/EU'

test_dir = '/Users/joe/Desktop/easyocr-failed-images-for-paddle'
accuracy, time = test_paddle(test_dir)
print(f'Accuracy achieved: {accuracy*100:2f}%')
print(f'Time per image: {time*1000:.2f} ms')

# accuracy, time = test_easyocr(eu_val_directory)
# print(f'Accuracy achieved: {accuracy*100:2f}%')
# print(f'Time per image: {time*1000:.2f} ms')