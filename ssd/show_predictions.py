from PIL import ImageDraw, Image, ImageFilter, ImageFont
from torchvision import transforms
import os
import time
import easyocr
import numpy as np
from torchvision.ops import nms
import cv2
import torch
from config import config

# Initialise EasyOCR reader
gpu = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device.type == 'cuda':
    gpu = True
reader = easyocr.Reader(['en'], gpu=gpu)

def crop_numberplate(pil_image, bbox):
    x_min, y_min, x_max, y_max = bbox

    if torch.is_tensor(x_min):
        cropped_img = pil_image.crop((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
    else:
        cropped_img = pil_image.crop((x_min, y_min, x_max, y_max))

    return cropped_img


def resize_image_maintain_aspect_ratio(image, new_width=None, new_height=None, HPC=False):
    # Get original image dimensions
    original_width, original_height = image.size

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    if new_width is not None:
        new_height = int(new_width / aspect_ratio)

    if new_height is not None:
        new_width = int(new_height * aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    if not HPC:
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else: resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image


def map_bbox_to_image(batch_images, batch_target_bboxes, batch_predicted_bboxes, batch_scores, save_directory, save=True):
    for i, image in enumerate(batch_images):
        image = transforms.ToPILImage()(image)  # Convert from tensor to PIL image for visualization
        draw = ImageDraw.Draw(image)

        predicted_bboxes = batch_predicted_bboxes[i]['boxes']
        predicted_scores = batch_scores[i]['scores']
        target_bboxes = batch_target_bboxes[i]['boxes']

        # Filter out all bboxes that have confidence scores below the score threshold (NMS doesn't effectively get rid
        # of all un-accurate bounding boxes, and there can be hundreds of detections per image)
        bbox_score_threshold = 0.4
        score_confidence_mask = predicted_scores > bbox_score_threshold
        masked_predicted_bboxes = predicted_bboxes[score_confidence_mask]
        masked_predicted_scores = predicted_scores[score_confidence_mask]

        # Apply Non-Maximum Suppression to bboxes. This eliminates lower confidence score boxes that overlap multiple
        # other bboxes, reducing the amount of redundant predictions.
        keep_indices = nms(masked_predicted_bboxes, masked_predicted_scores,
                           iou_threshold=0.5)
        nms_predicted_bboxes = masked_predicted_bboxes[keep_indices]
        nms_predicted_scores = masked_predicted_scores[keep_indices]

        if len(nms_predicted_bboxes) != 0:
            # Draw predicted bboxes in red
            for j, nms_predicted_bboxes in enumerate(nms_predicted_bboxes):
                x_min, y_min, x_max, y_max = predicted_bboxes[j]

                # Read the license plate text
                cropped_image = crop_numberplate(image, predicted_bboxes[j])

                # Resize the cropped image (make larger)
                resized_image = resize_image_maintain_aspect_ratio(cropped_image, new_width=200, new_height=200)

                # Apply sharpening filter
                sharpened_image = resized_image.filter(ImageFilter.SHARPEN)

                # Convert image to from PIL to openCV
                np_cropped_image = np.array(sharpened_image)
                opencv_image = cv2.cvtColor(np_cropped_image, cv2.COLOR_RGB2BGR)

                # Convert the image to grayscale
                gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
                thresholded_image = cv2.adaptiveThreshold(
                    gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
                np_img = np.array(thresholded_image)

                # Perform OCR on license plate and extract text
                detections = reader.readtext(np_img)
                #print(detections)
                license_plate_text = ''

                for detection in detections:
                    license_plate_text += detection[1]
                if len(license_plate_text) == 0:
                    license_plate_text = 'NaN'

                # Gets score for bounding box in the desired format.
                text = str(round(nms_predicted_scores[j].item(), 2))
                text += '    '
                text += license_plate_text

                # Gets dimensions of text so a rectangle can be mapped under it (anchor set to FONT_SIZE pixels above predicted bbox)
                if not config.HPC: font_path = "/Users/joe/Code/third-year-project/ANPR/fonts/DejaVuSans.ttf"
                else: font_path = "/gpfs/home/hyg22ktu/fonts/DejaVuSans.ttf"
                font = ImageFont.truetype(font_path, size=config.FONT_SIZE)

                # Gets dimensions of text so a rectangle can be mapped under it (anchor set to 12 pixels above predicted
                # bbox
                xmin, ymin, xmax, ymax = draw.textbbox(xy=(x_min, y_min - config.FONT_SIZE), text=text, font=font)

                # draw red background box with dimensions equal to text dimensions.
                draw.rectangle([xmin, ymin, xmax, ymax], fill="red")
                # Draw confidence score in white overlaid onto red background, and 12 pixels above the predicted bbox
                draw.text(xy=(x_min, y_min - config.FONT_SIZE), text=text, fill="white", font=font)

                # Draws the predicted bounding box outline in red
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=1)

        #else: print(f'{i} - SKIP')

        # Draw target (actual) bbox in green
        for i in range(target_bboxes.shape[0]):
            x_min, y_min, x_max, y_max = target_bboxes[i]
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=1)

        # Show image
        if not config.HPC:
            image.show()
        if save:
            timestamp = time.time()
            if len(nms_predicted_bboxes) == 0:
                timestamp = 'SKIP_' + str(timestamp)
            filepath = os.path.join(save_directory, f'{timestamp}.png')
            image.save(filepath)
