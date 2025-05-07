from ultralytics import YOLO
from ssd.show_predictions import crop_numberplate, resize_image_maintain_aspect_ratio
from config import config
import os
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np
import cv2
import easyocr
from paddleocr import PaddleOCR

# Load model using fine-tuned weights from HPC
model = YOLO("/Users/joe/Code/third-year-project/ANPR/yolov8n/weights/run9_best.pt") # pretrained

# Initialise EasyOCR reader
#reader = easyocr.Reader(['en'], gpu=config.HPC)
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory

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


def draw_bbox(image, draw, predicted_bbox, prediction_score, **preprocess_kwargs):
    """
    Draws bounding box on the image. Takes one predicted bbox and prediction score at a time. If image has multiple
    license plates, each model predicted bbox and score should be provided one by one.
    :param image: PIL image (for the localise and preprocess function to feed to EasyOCR reader)
    :param draw: PIL Image Draw (a global Draw Image to draw bbox on image)
    :param predicted_bbox: Model's (Singular) predicted bounding box
    :param prediction_score: Model's (Singular) prediction score'
    :return:
    """
    # Get confidence score from inferred boxes
    # score = r.boxes.conf[j]
    x_min, y_min, x_max, y_max = predicted_bbox

    steps, np_img = localise_and_preprocess_license_plate(image, predicted_bbox, **preprocess_kwargs)

    #detections = reader.readtext(np_img)
    #print(detections)

    # license_plate_text = ''
    # for detection in detections:
    #     license_plate_text += detection[1]
    # if len(license_plate_text) == 0:
    #     license_plate_text = 'NaN'

    # formatted_np_img = np.array(np_img)[:, :, ::-1]  # Convert RGB to BGR
    detections = ocr.ocr(np_img, cls=True)

    license_plate_text = ''
    if detections is not None or len(detections) != 0 or detections[0] is not None:
        for i in range(len(detections)):
            detection = detections[i]
            for det in detection:
                if len(det) >= 2:
                    license_plate_text += det[1][0]

    if len(license_plate_text) == 0:
        license_plate_text = 'NaN'

    # Gets score for bounding box in the desired format.
    text = str(round(prediction_score.item(), 2))
    text += '    '
    text += license_plate_text

    # if not config.HPC:
    font_path = "/Users/joe/Code/third-year-project/ANPR/fonts/DejaVuSans.ttf"
    # else:
    #     font_path = "/gpfs/home/hyg22ktu/fonts/DejaVuSans.ttf"

    # Gets dimensions of text so a rectangle can be mapped under it (anchor set to FONT_SIZE pixels above predicted bbox)
    font = ImageFont.truetype(font_path, size=config.FONT_SIZE)
    xmin, ymin, xmax, ymax = draw.textbbox(xy=(x_min, y_min - config.FONT_SIZE), text=text, font=font)

    # check if textbbox clips outside of image
    text_x = x_min
    text_y = y_min - config.FONT_SIZE
    if ymin < 0:
        text_y = y_min + config.FONT_SIZE
    if xmax > image.size[0]:
        text_x = image.size[0] - (xmax - xmin)


    # draw red background box with dimensions equal to text dimensions.
    # draw.rectangle([xmin, ymin, xmax, ymax], fill="red")
    draw.rectangle(
        draw.textbbox(xy=(text_x, text_y), text=text, font=font),
        fill="red"
    )
    # Draw confidence score and plate text in white overlaid onto red background, and FONT_SIZE pixels above the predicted bbox
    # draw.text(xy=(x_min, y_min - config.FONT_SIZE), text=text, fill="white", font=font)
    draw.text(xy=(text_x, text_y), text=text, fill="white", font=font)

    # Draws the predicted bounding box outline in red
    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=1)

    return steps # can be None if show_steps is False

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def localise_and_preprocess_license_plate(image, predicted_bbox, *, sharpen=True, grayscale=True, threshold=False, histogram_equalisation=False, rotate=0, show_steps=False):
    """
    Crops the image according to the predicted bbox. Resizes the crop, then sharpens and grayscales the crop. Then converts
    crop to a numpy array. (Optional thresholding step).
    :param image: PIL image
    :return: Preprocessed numpy image array
    """
    steps = { "resized": None, "sharpened": None, "grayscale": None, "threshold": None, "histogram_equalisation": None, "rotated": None } # contains at least one PIL image

    # Crop the image to get the inferred numberplate
    cropped_image = crop_numberplate(image, predicted_bbox)
    # Preprocess image
    resized_image = resize_image_maintain_aspect_ratio(cropped_image, new_width=200, new_height=200)
    steps["resized"] = resized_image

    # OPTIONAL STEPS
    if sharpen:
        # Apply sharpening filter
        resized_image = resized_image.filter(ImageFilter.SHARPEN)
        steps["sharpened"] = resized_image
    # sharpened_image.show()

    # Convert to np array to apply cv2 image processing
    np_image = np.array(resized_image)

    # Convert the image to grayscale
    if grayscale:
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        steps["grayscale"] = Image.fromarray(np_image)

    if histogram_equalisation and grayscale:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        np_image = clahe.apply(np_image)
        steps["histogram_equalisation"] = Image.fromarray(np_image)

    if threshold and grayscale: # Thresholding is only meaningful in grayscale images
        np_image = cv2.adaptiveThreshold(
            np_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 2 # blockSize=11
        )
        steps["threshold"] = Image.fromarray(np_image)

    if rotate != 0:
        np_image = rotate_image(np_image, rotate)
        steps["rotated"] = Image.fromarray(np_image)

    if show_steps:
        return steps, np_image

    return None, np_image

if __name__ == '__main__':
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
            predicted_bboxes = r.boxes.xyxy
            print(predicted_bboxes)
            print(bboxes)
            # Iterate through predicted bboxes
            for j, bbox in enumerate(predicted_bboxes):
                # Get confidence score from inferred boxes
                score = r.boxes.conf[j]
                x_min, y_min, x_max, y_max = bbox

                np_img = localise_and_preprocess_license_plate(image, bbox)

                #detections = reader.readtext(np_img)
                detections = None
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

        if not config.HPC:
            pil_image.show()


        if i == 10:
            break
