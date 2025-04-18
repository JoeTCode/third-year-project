from ultralytics import YOLO
from ssd.show_predictions import crop_numberplate, resize_image_maintain_aspect_ratio
from config import config
import os
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import cv2
import easyocr

# Load model using fine-tuned weights from HPC
model = YOLO("./weights/run9_best.pt") # pretrained

# Initialise EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)


def reformat_bbox(bbox, image_height, image_width):
    """
    Converts YOLO bounding box: [x_center, y_center, width, height] to pytorch bounding box: [x_min, y_min, x_max, y_max].
    :param image_width:
    :param image_height:
    :param bbox: (List), [x_center, y_center, width, height].
    :return: (List), [x_min, y_min, x_max, y_max].
    """
    x_center, y_center, width, height = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # Convert YOLO normalized values to absolute values
    x_center = x_center * image_width
    y_center = y_center * image_height
    width = width * image_width
    height = height * image_height

    # Calculate the top-left and bottom-right corners
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2

    return [x_min, y_min, x_max, y_max]


for i, image in enumerate(os.listdir(config.VALID_IMAGES_ROOT)):
    pil_image = Image.open(os.path.join(config.VALID_IMAGES_ROOT, image))
    draw = ImageDraw.Draw(pil_image)
    image_width, image_height = pil_image.size

    txt = os.path.splitext(image)[0] + '.txt'
    txt_path = os.path.join(config.VALID_ANNOTATIONS_ROOT, txt)

    # Get target labels and bboxes
    f = open(txt_path, 'r')
    lines = f.readlines()
    bboxes = [reformat_bbox(line.split(" ")[1:], image_height, image_width) for line in lines]
    if len(bboxes) == 0:
        labels = [0]  # If no bboxes are found, make the image a background class image (0)
        bboxes = [[0, 0, 1, 1]]  # Create a small bbox for background image
    else:
        labels = [int(line.split(" ")[0]) + 1 for line in lines]
    f.close()

    # Run inference
    results = model(source=pil_image, show=False, conf=0.4, verbose=False, save=False)
    for r in results:
        print(r.boxes.xyxy)
        print(bboxes)
        # Iterate through target bboxes
        for j, bbox in enumerate(bboxes):
            # Get confidence score from inferred boxes
            score = r.boxes.conf[j]
            x_min, y_min, x_max, y_max = bbox

            # Crop the image to get the inferred numberplate
            cropped_image = crop_numberplate(pil_image, bbox)

            # Preprocess image
            resized_image = resize_image_maintain_aspect_ratio(cropped_image, new_width=200, new_height=200)
            # Apply sharpening filter
            sharpened_image = resized_image.filter(ImageFilter.SHARPEN)
            #sharpened_image.show()
            np_sharpened_image = np.array(sharpened_image)
            opencv_image = cv2.cvtColor(np_sharpened_image, cv2.COLOR_RGB2BGR)
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            thresholded_image = cv2.adaptiveThreshold(
                gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            np_img = np.array(gray_image)
            detections = reader.readtext(np_img)
            print(detections)

            license_plate_text = ''
            for detection in detections:
                license_plate_text += detection[1]
            if len(license_plate_text) == 0:
                license_plate_text = 'NaN'

            # Gets score for bounding box in the desired format.
            text = str(round(score.item(), 2))
            text += '    '
            text += license_plate_text
            # Gets dimensions of text so a rectangle can be mapped under it (anchor set to FONT_SIZE pixels above predicted bbox)
            if not config.HPC:
                font_path = "/Users/joe/Code/third-year-project/ANPR/fonts/DejaVuSans.ttf"
            else:
                font_path = "/gpfs/home/hyg22ktu/fonts/DejaVuSans.ttf"
            font = ImageFont.truetype(font_path, size=config.FONT_SIZE)
            xmin, ymin, xmax, ymax = draw.textbbox(xy=(x_min, y_min - config.FONT_SIZE), text=text, font=font)

            # draw red background box with dimensions equal to text dimensions.
            draw.rectangle([xmin, ymin, xmax, ymax], fill="red")
            # Draw confidence score and plate text in white overlaid onto red background, and FONT_SIZE pixels above the predicted bbox
            draw.text(xy=(x_min, y_min - config.FONT_SIZE), text=text, fill="white", font=font)

            # Draws the predicted bounding box outline in red
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=1)

    pil_image.show()


    if i == 2:
        break