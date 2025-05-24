import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from io import BytesIO
from flask import Flask, request, redirect, url_for
from flask import render_template
from werkzeug.utils import secure_filename
from yolov8n.load_model_and_infer import draw_bbox
from PIL import Image, ImageDraw
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import models
from ssd.show_predictions import crop_numberplate
from torchvision import transforms
from paddleocr import PaddleOCR

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # need to run only once to download and load model into memory
model = YOLO("/Users/joe/Code/third-year-project/ANPR/yolov8n/weights/run9_best.pt")  # pretrained

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend', 'templates'),
    static_folder=os.path.join(BASE_DIR, 'frontend', 'static')
)

@app.route('/')
def index_page():
    return render_template('index.html')

@app.post('/upload-image')
def upload_image():

    if 'image' not in request.files:
        return 'No file', 400
    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    _, extension = os.path.splitext(filename)

    preprocessing = request.form.getlist('preprocessing')
    sharpen = True if 'sharpen' in preprocessing else False
    grayscale = True if 'grayscale' in preprocessing else False
    threshold = True if 'threshold' in preprocessing else False
    histogram_equalisation = True if 'histogram_equalisation' in preprocessing else False
    show_steps = True if 'show_steps' in preprocessing else False

    image = Image.open(BytesIO(file.read()))

    input_filename = save_image(extension, image, save_image_directory_name='input_images')
    #image.save(f"frontend/static/input_images/{filename}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_path = '/Users/joe/Code/third-year-project/ANPR/classifier/save_weights/mobilenet_v2_weights_1.pth'
    classifier = load_plate_classifier(model_path, device)

    # Run inference
    results = model(source=image, show=False, conf=0.4, verbose=False, save=False)


    all_steps = [] # holds all preprocessing steps for each bbox detection detected from the input image.

    for r in results:
        draw = ImageDraw.Draw(image)
        bbox_predictions = r.boxes.xyxy
        prediction_scores = r.boxes.conf
        for i, predicted_bbox in enumerate(bbox_predictions):
            prediction_score = prediction_scores[i]

            cropped_image = crop_numberplate(image, predicted_bbox)
            plate_types = ['EU', 'Non-EU']
            plate_id = return_plate_type(cropped_image, classifier, transform, device)
            plate_type = plate_types[plate_id]

            steps = draw_bbox(
                image, draw, predicted_bbox, prediction_score, ocr, plate_type=plate_type,
                sharpen=sharpen,
                grayscale=grayscale,
                threshold=threshold,
                histogram_equalisation=histogram_equalisation,
                show_steps=show_steps
            )

            if steps:
                all_steps.append(list(steps.values()))

    stacks = []
    for steps in all_steps:
        stack = stack_one_bbox_images(steps)
        stacks.append(stack)

    steps_filename = None

    if len(stacks) == 1:
        steps_filename = save_image('.jpg', stacks[0], save_image_directory_name='output_images_steps')
    if len(stacks) > 1:
        steps_filename = save_image('.jpg', align_stacks(stacks), save_image_directory_name='output_images_steps')

    print(steps_filename, stacks)

    output_filename = save_image(extension, image, save_image_directory_name='output_images')


    return render_template(
        'index.html',
        input_filename=input_filename,
        output_filename=output_filename,
        steps_filename=steps_filename
    )

def load_plate_classifier(model_path, device):
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model

def return_plate_type(cropped_image, model, transform, device):
    """

    :param cropped_image:
    :return: (Int) Returns 0 or 1. Where EU: 0 and Non-EU: 1
    """

    if cropped_image.mode != 'RGB':
        cropped_image = cropped_image.convert('RGB')
    img = transform(cropped_image)


    with torch.no_grad():
        output = model(img.unsqueeze(0).to(device))
        _, pred = torch.max(output, 1)
        class_idx = pred.item()
        print(f'Classifier pred: {class_idx}')
        return class_idx # EU: 0, Non-EU: 1


def save_image(extension, image, *, save_image_directory_name):
    """

    :param image_name: (Str) Image file's extension.
    :param image: PIL image.
    :param save_image_directory_name: (Str) 'output_images' or 'input_images'.
    :return: (Str) Filename of the saved image. Either input_<int> or output_<int>.
    """

    image_dir = os.path.join(BASE_DIR, 'frontend', 'static', save_image_directory_name)
    images = [image for image in os.listdir(image_dir) if not image.startswith('.')]

    image_id = -1
    if len(images) > 0:
        for filename in images:
            filename, _ = os.path.splitext(filename)
            image_id = max(image_id, int(filename.split('_')[1]))

    image_id += 1
    image.save(os.path.join(image_dir, f'image_{image_id}{extension}'))
    return f'image_{image_id}{extension}'


def stack_one_bbox_images(images):
    valid_images = [image for image in images if image is not None]
    widths = [image.size[0] for image in valid_images]
    heights = [image.size[1] for image in valid_images]

    max_width = max(widths)
    total_height = sum(heights)

    canvas = Image.new('RGB', (max_width, total_height))

    y = 0
    for image in valid_images:
        canvas.paste(image, (0, y))
        y+=image.size[1]

    return canvas

def align_stacks(stacks): # preserves aspect ratio while keep all stacks the same height
    max_height = max(stack.size[1] for stack in stacks)
    resized_stacks = []
    for stack in stacks:
        w, h = stack.size
        new_w = int(w * (max_height / h))
        resized_stacks.append(stack.resize((new_w, max_height)))

    total_width = sum(stack.size[0] for stack in resized_stacks)
    canvas = Image.new('RGB', (total_width, max_height))

    x = 0
    for stack in resized_stacks:
        canvas.paste(stack, (x, 0))
        x += stack.size[0]

    return canvas