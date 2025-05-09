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

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

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
        return 'No file part', 400
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


    print(sharpen, grayscale, threshold, histogram_equalisation, show_steps)

    image = Image.open(BytesIO(file.read()))

    input_filename = save_image(extension, image, save_image_directory_name='input_images')
    #image.save(f"frontend/static/input_images/{filename}")

    model = YOLO("/Users/joe/Code/third-year-project/ANPR/yolov8n/weights/run9_best.pt")  # pretrained
    # Run inference
    results = model(source=image, show=False, conf=0.4, verbose=False, save=False)


    all_steps = [] # holds all preprocessing steps for each bbox detection detected from the input image.

    for r in results:
        draw = ImageDraw.Draw(image)
        bbox_predictions = r.boxes.xyxy
        prediction_scores = r.boxes.conf
        for i, predicted_bbox in enumerate(bbox_predictions):
            prediction_score = prediction_scores[i]
            steps = draw_bbox(
                image, draw, predicted_bbox, prediction_score,
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


def save_image(extension, image, *, save_image_directory_name):
    """

    :param image_name: (Str) Image file's extension.
    :param image: PIL image.
    :param save_image_directory_name: (Str) 'output_images' or 'input_images'.
    :return: (Str) Filename of the saved image. Either input_<int> or output_<int>.
    """

    image_dir = os.path.join(BASE_DIR, 'frontend', 'static', save_image_directory_name)
    images = os.listdir(image_dir)

    latest_image_num = 0
    if len(images) > 0:
        for filename in images:
            if not filename.startswith('.'): # skip hidden files (like .DS_Store on mac)
                filename, _ = os.path.splitext(filename)
                latest_image_num = max(latest_image_num, int(filename.split('_')[1]))


    image.save(os.path.join(image_dir, f'image_{latest_image_num+1}{extension}'))

    if latest_image_num == 1: return f'image_0{extension}'
    else: return f'image_{latest_image_num+1}{extension}'


def stack_one_bbox_images(images):
    valid_images = [image for image in images if image is not None]
    widths, heights = zip(*(image.size for image in valid_images))

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