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

gpu = False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if device.type == 'cuda':
    gpu = True
# reader = easyocr.Reader(['en'], gpu=gpu)

def crop_numberplate(pil_image, bbox):
    x_min, y_min, x_max, y_max = bbox

    if torch.is_tensor(x_min):
        cropped_img = pil_image.crop((x_min.item(), y_min.item(), x_max.item(), y_max.item()))
    else:
        cropped_img = pil_image.crop((x_min, y_min, x_max, y_max))

    return cropped_img


def resize_image_maintain_aspect_ratio(image, new_width=None, new_height=None, HPC=False):

    original_width, original_height = image.size

    aspect_ratio = original_width / original_height

    if new_width is not None:
        new_height = int(new_width / aspect_ratio)

    if new_height is not None:
        new_width = int(new_height * aspect_ratio)

    if not HPC:
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    else: resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    return resized_image

def filter_model_predictions(predicted_bboxes, predicted_scores, predicted_labels, bbox_score_threshold=0.4, iou_threshold=0.5):
    """

    :param predicted_bboxes: Takes a list of SSD bbox predictions (predictions[i]['boxes']).
    :param predicted_scores: Takes a list of SSD score predictions (predicted_scores = predictions[i]['scores']).
    :param predicted_labels: Takes a list of SSD label predictions (target_bboxes = targets[i]['boxes']).
    :param bbox_score_threshold: (float) Optional min confidence score value.
    :param iou_threshold: (float) Optional IOU value.
    :return: Returns lists of filtered bboxes, scores, and labels predictions.
    """

    # 2 steps: Confidence score threshold filter, then NMS filter.
    # Filter out all bboxes that have confidence scores below the score threshold
    score_confidence_mask = predicted_scores > bbox_score_threshold  # true false array
    bboxes = predicted_bboxes[score_confidence_mask]
    scores = predicted_scores[score_confidence_mask]
    labels = predicted_labels[score_confidence_mask]

    # Apply Non-Maximum Suppression to bboxes. This eliminates lower confidence score boxes that overlap multiple
    # other bboxes, reducing the amount of redundant predictions.
    keep_indices = nms(bboxes, scores, iou_threshold=iou_threshold)
    return bboxes[keep_indices], scores[keep_indices], labels[keep_indices]

def preprocess_image(image, bbox=None, crop_size=None):
    """
    Takes a PIL image and bbox, and preprocesses the image.
    :param crop_size: (Optional) (tuple) In the form (width, height).
    :param image: PIL image.
    :param bbox: (Optional) A bbox.
    :return: A processed image converted to a numpy array.
    """

    resized_image = image

    if crop_size is not None and bbox is not None:

        cropped_image = crop_numberplate(image, bbox)

        assert isinstance(crop_size, tuple), "Size argument needs to a tuple in the form (width, height)"
        # Resize the cropped image (make larger)
        resized_image = resize_image_maintain_aspect_ratio(cropped_image, new_width=crop_size[0], new_height=crop_size[1])


    # Apply sharpening filter
    sharpened_image = resized_image.filter(ImageFilter.SHARPEN)

    # Convert image to from PIL to openCV
    np_cropped_image = np.array(sharpened_image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(np_cropped_image, cv2.COLOR_BGR2GRAY)
    thresholded_image = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return np.array(thresholded_image)

def map_bbox_to_image(batch_images, targets, predictions, save_directory, original_image_loader, resize, save=True):
    """

    :param batch_images: A list of torch tensor images.
    :param targets: A list of annotation dictionaries. (WITH IMAGE_ID)
    :param predictions:
        A list of prediction dictionaries (one per image) containing predicted bboxes, scores, and labels in the form
        [{
            'boxes': Tensor[N, 4],       # predicted bounding boxes
            'labels': Tensor[N],         # predicted class labels
            'scores': Tensor[N]          # confidence scores for each prediction
        }, ...]
    :param save_directory:
    :param save:
    :return:
    """
    reader = easyocr.Reader(['en'], gpu=gpu)

    for i, image in enumerate(batch_images):
        image = transforms.ToPILImage()(image)  # Convert from tensor to PIL image for visualization


        predicted_bboxes = predictions[i]['boxes']
        predicted_scores = predictions[i]['scores']
        predicted_labels = predictions[i]['labels']
        target_bboxes = targets[i]['boxes']
        target_id = targets[i]['image_id']

        original_image = original_image_loader[target_id]
        draw = ImageDraw.Draw(original_image)

        resized_target_dict = resize(original_image.size)({
            "image": image,
            "annotations": {"image_id": target_id, "boxes": target_bboxes, "labels": targets[i]['labels']}
        })
        resized_target_boxes = resized_target_dict['annotations']['boxes']



        filtered_bboxes, filtered_scores, filtered_labels = filter_model_predictions(predicted_bboxes, predicted_scores, predicted_labels)

        if len(filtered_bboxes) != 0:
            resized_to_original_dict = resize(original_image.size)({
                "image": image,
                "annotations": {"image_id": target_id, "boxes": filtered_bboxes, "labels": filtered_labels}
            })
            bboxes = resized_to_original_dict["annotations"]["boxes"]
            # Draw predicted bboxes in red
            for j, nms_bbox in enumerate(bboxes):
                x_min, y_min, x_max, y_max = nms_bbox


                np_img = preprocess_image(original_image, nms_bbox, (200,200))

                # Perform OCR on license plate and extract text
                detections = reader.readtext(np_img)
                #print(detections)
                license_plate_text = ''

                for detection in detections:
                    license_plate_text += detection[1]
                if len(license_plate_text) == 0:
                    license_plate_text = 'NaN'

                # Gets score for bounding box in the desired format.
                text = str(round(filtered_scores[j].item(), 2))
                text += '    '
                text += license_plate_text

                # Gets dimensions of text so a rectangle can be mapped under it (anchor set to FONT_SIZE pixels above predicted bbox)
                if not config.HPC: font_path = config.FONT_PATH
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
        for i in range(resized_target_boxes.shape[0]):
            x_min, y_min, x_max, y_max = resized_target_boxes[i]
            draw.rectangle([x_min, y_min, x_max, y_max], outline="green", width=1)

        # Show image
        if not config.HPC:
            original_image.show()
        if save:
            timestamp = time.time()
            if len(filtered_bboxes) == 0:
                timestamp = 'SKIP_' + str(timestamp)
            filepath = os.path.join(save_directory, f'{timestamp}.png')
            image.save(filepath)
