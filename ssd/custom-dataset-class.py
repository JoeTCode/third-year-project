# credit to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import json
import torch
import warnings
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from config import MAX_ANNOTATIONS

train_root = "/Users/joe/Desktop/eu-dataset/train/images"
train_annotations_file = "eu-train-dataset-coco.json"

class Resize():
    def __init__(self, size):
        assert isinstance(size, tuple) and len(size) == 2, "Size must be a tuple (height, width)"
        self.size = size
    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']

        h, w = image.size
        scale_x = self.size[1] / w
        scale_y = self.size[0] / h

        if h != w:
            warnings.warn(f"Warning, image's height {h} and width {w} do not match. This function does not "
                          f"preserve aspect ratio. This could lead to distortion.", UserWarning)

        resize_image = transforms.Resize(self.size)
        resized_image = resize_image(image)

        for annotation in annotations:
            if annotation != 0:
                bbox = annotation['bbox']
                x_min, y_min, width, height = bbox
                new_x_min = x_min * scale_x
                new_y_min = y_min * scale_y
                new_width = width * scale_x
                new_height = height * scale_y
                annotation['bbox'] = [new_x_min, new_y_min, new_width, new_height]

        return {"image": resized_image, "annotations": annotations}

class ToTensor():
    """ Converts PIL image into tensor. """
    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)
        return {'image': image_tensor, 'annotations': annotations}


# perform some transformations like resizing,
# centering and tensor conversion
# using transforms function
transform = transforms.Compose([
        # transforms.Resize((300, 300)),
        Resize((300, 300)),
	    ToTensor()
     ])

class AnprCocoDataset(Dataset):
    def __init__(self, train_annotations_file_path, train_images_root, transform=None):
        """

        :param train_annotations_file_path: Image annotations.
        :param train_images_root: Images folder.
        :param transform: Optional transform to be applied on a sample.
        """
        self.train_annotations_file_path = train_annotations_file_path
        self.train_images_root = train_images_root
        self.transform = transform

        with open(self.train_annotations_file_path, "r") as f:
            data = json.load(f)
        self.train_annotations = data["annotations"]
        self.all_annotations = data["images"] # list of dicts

    def __len__(self):
        return len(self.train_annotations)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.all_annotations[idx]["file_name"]
        image_path = os.path.join(self.train_images_root, image_name)
        image = Image.open(image_path)

        all_annotations = self.train_annotations
        annotations = [annotation for annotation in all_annotations if annotation["image_id"] == idx]
        padding = [0] * (MAX_ANNOTATIONS - len(annotations))
        annotations.extend(padding)

        sample = {'image': image, 'annotations': annotations}

        # If necessary, perform transforms (e.g., resize, normalize)
        if self.transform:
            sample = self.transform(sample) # transformed sample

        return sample

anpr_coco_dataset = AnprCocoDataset(
    train_annotations_file_path=train_annotations_file,
    train_images_root=train_root,
    transform=transform
)


def show_bbox(sample, transform=None):
    """
    Draws bounding boxes on an image using PIL.
    :param sample: List of dicts, containing annotations w.r.t. to the image.
    :return: Image with bounding boxes.
    """

    image, annotations = sample['image'], sample['annotations'] # transform image and transform annotations
    if transform:
        image = transforms.ToPILImage()(image)  # Convert back to PIL for visualization

    draw = ImageDraw.Draw(image)
    # Draw each bbox
    for annotation in annotations:
        if annotation != 0: # Check it is not padding
            bbox = annotation["bbox"]
            x_min, y_min, width, height = bbox
            # Reformatting the COCO bbox values to visualise a rectangular bbox on the image
            x_max = x_min + width
            y_max = y_min + height
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # Show image
    image.show()

for i, sample in enumerate(anpr_coco_dataset):
    image, annotations = sample['image'], sample['annotations']

    if isinstance(image, Image.Image):
        print(i, sample['image'].size, len(annotations))
        with open(train_annotations_file, "r") as f:
            data = json.load(f)
        show_bbox(sample)

    else:
        print(i, sample['image'].size(), len(annotations))
        with open(train_annotations_file, "r") as f:
            data = json.load(f)
        show_bbox(sample, transform=transform)

    if i == 3:
        break

