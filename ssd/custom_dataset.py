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
from config import MAX_ANNOTATIONS, BBOX_LENGTH

train_root = "/Users/joe/Desktop/eu-dataset/train/images"
train_annotations_file = "eu-train-dataset-coco.json"

class Resize:
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

def annotations_to_tensor(annotations, bbox_length=BBOX_LENGTH):
    annotations_list = []
    for annotation in annotations:
        if annotation == 0:
            annotations_list.append([0]*(bbox_length + 3)) # extra 3 padding represents id, category_id, image_id
        else:
            bbox = list(annotation.values())[3]
            identifiers = list(annotation.values())[:3] # id, category_id, image_id
            identifiers.extend(bbox)
            annotations_list.append(identifiers)

    annotations_numpy_list = np.array(annotations_list)
    to_tensor = transforms.ToTensor()
    return to_tensor(annotations_numpy_list)

def is_padding(subarray):
    return all(value == 0 for value in subarray)

class ToTensor:
    """ Converts PIL image into tensor. """
    def __call__(self, sample):
        image, annotations = sample['image'], sample['annotations']
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)
        annotations_tensor = annotations_to_tensor(annotations)

        return {'image': image_tensor, 'annotations': annotations_tensor}


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
        self.image_annotations = data["images"] # list of dicts

    def __len__(self):
        return len(self.image_annotations)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(idx)

        image_name = self.image_annotations[idx]["file_name"]
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

def convert_annotations_tensor(annotations_tensor, max_annotations=MAX_ANNOTATIONS):
    annotations_list = annotations_tensor.tolist()[0]
    anno_list = []
    for annotation in annotations_list:
        if not is_padding(annotation):
            anno_dict = {"id": int(annotation[0]), "category_id": int(annotation[1]), "image_id": int(annotation[2]),
                         "bbox": annotation[3:]}
            anno_list.append(anno_dict)
    length = len(anno_list)
    anno_list.extend([0]*(max_annotations - length))
    return anno_list

def show_bbox(sample, transform=None):
    """
    Draws bounding boxes on an image using PIL.
    :param sample: List of dicts, containing annotations w.r.t. to the image.
    :return: Image with bounding boxes.
    """

    image, annotations = sample['image'], convert_annotations_tensor(sample['annotations']) # transform image and transform annotations

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

def test_sample_dataset(dataset):
    for i, sample in enumerate(dataset):
        image, annotations = sample['image'], convert_annotations_tensor(sample['annotations'])

        if isinstance(image, Image.Image): # If image is type PIL (the images were not transformed)
            print(f"{i}: {sample['image'].size()} -- {len(annotations)}, {annotations}")
            show_bbox(sample)

        else: # Images were transformed
            print(f"{i}: {sample['image'].size()} -- {len(annotations)}, {annotations}")
            show_bbox(sample, transform=transform)

        if i == 3:
            break

if __name__ == '__main__':
    anpr_coco_dataset = AnprCocoDataset(
        train_annotations_file_path=train_annotations_file,
        train_images_root=train_root,
        transform=transform
    )
    test_sample_dataset(anpr_coco_dataset)

    # test = [{'id': 0, 'category_id': 0, 'image_id': 0, 'bbox': [127.734375, 172.03124999999997, 19.21875, 10.3125]},
    #         {'id': 0, 'category_id': 0, 'image_id': 0, 'bbox': [127.734375, 172.03124999999997, 19.21875, 10.3125]}, 0, 0, 0]
    # tensor = annotations_to_tensor(test)
    # annotation = convert_annotations_tensor(tensor)
    # print(annotation)