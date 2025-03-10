# outlined by https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html

from PIL import Image
import os
import json
from config import MAX_ANNOTATIONS

def preprocess_dataset(images_path, labels_path):
    # creates list of image names without file extension
    file_names = []

    for image in os.listdir(images_path):
        file_name = os.path.splitext(image)[0]
        image_path = os.path.join(labels_path, file_name)
        if os.path.getsize(f"{image_path}.txt") == 0:
            continue
        else:
            file_names.append(file_name)

    return file_names

def convert_to_coco(images_path, labels_path, output_json_path):
    images = []
    annotations = []
    categories = [{"id": 0, "name": "license_plate"}]

    """
    "images": [
        {"id": 1202, "width": 426, "height": 640, "file_name": "xxxxxxxxx.jpg", "date_captured": "2013-11-15 02:41:42"},
        {"id": 2949, "width": 640, "height": 480, "file_name": "nnnnnnnnnn.jpg", "date_captured": "2013-11-18 02:53:27"}
    ],
    """
    image_id = 0
    annotation_id = 0
    date = "2025-03-03 02:41:42"

    downloaded_images = preprocess_dataset(images_path, labels_path)
    print("test", downloaded_images)
    for image_name in downloaded_images:
        # populate "images" list
        image_path = os.path.join(images_path, image_name)
        image = Image.open(image_path + ".jpg")
        width, height = image.size

        images.append({"id": image_id, "width": width, "height": height, "file_name": image_name+".jpg", "date_captured": date})

        # populate "annotations" list
        """
        "annotations": [
            {"id": 125686, "category_id": 0, "image_id": 242287, "bbox": [19.23, 383.18, 314.5, 244.46]},
            {"id": 1409619, "category_id": 0, "image_id": 245915, "bbox": [399, 251, 155, 101]},
            {"id": 1410165, "category_id": 1, "image_id": 245915, "bbox": [86, 65, 220, 334]}
        ],
        """

        path = os.path.join(labels_path, image_name)
        annotation_for_image_count = 0
        with open(f"{path}.txt", "r") as f:
            for line in f:
                if annotation_for_image_count == MAX_ANNOTATIONS:
                    break
                class_id, x_center, y_center, bbox_width, bbox_height = line.split(" ")

                # Convert String to float
                x_center = float(x_center)
                y_center = float(y_center)
                bbox_width = float(bbox_width)
                bbox_height = float(bbox_height)

                # Convert YOLO format to COCO format
                xmin = (x_center - bbox_width / 2) * width
                ymin = (y_center - bbox_height / 2) * height
                bbox_width = bbox_width * width
                bbox_height = bbox_height * height

                annotations.append({"id": annotation_id, "category_id": categories[0]["id"], "image_id": image_id, "bbox": [xmin, ymin, bbox_width, bbox_height]})
                annotation_id += 1
                annotation_for_image_count += 1

        image_id += 1

    container = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json_path, 'w') as json_file:
        json.dump(container, json_file, indent=4)


# path to the training subset of the eu-dataset, containing the images and labels folders
images_path = "/Users/joe/Desktop/eu-dataset/train/images"
labels_path = "/Users/joe/Desktop/eu-dataset/train/labels"

# Path to output JSON file
output_file = 'eu-train-dataset-coco.json'

convert_to_coco(images_path, labels_path, output_file)
