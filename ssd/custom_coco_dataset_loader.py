# credit to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os
import json
import torch
import warnings
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ssd.config.config import MAX_ANNOTATIONS, BBOX_LENGTH, COCO_TRAIN_ROOT, COCO_TRAIN_ANNOTATIONS_FILE, BATCH_SIZE


class Resize:
    """
    Resizes an image to the desired dimensions
    """

    def __init__(self, size):
        """
        :param size: The dimensions to resize the image to.
        """
        assert isinstance(size, tuple) and len(size) == 2, "Size must be a tuple (height, width)"
        self.size = size

    def __call__(self, sample):
        """

        :param sample: A dictionary containing the image, and the annotations. Annotations is a dictionary containing
            image_id, boxes, labels.
        :return: A dictionary with resized image and annotation boxes.
        """

        image, annotations = sample["image"], sample["annotations"]
        bboxes = annotations['boxes']

        h, w = image.size  # Original image size
        scale_x = self.size[1] / w
        scale_y = self.size[0] / h

        if h != w:
            warnings.warn(f"Warning, image's height {h} and width {w} do not match. This function does not "
                          f"preserve aspect ratio. This could lead to distortion.", UserWarning)

        resize_image = transforms.Resize(self.size)
        resized_image = resize_image(image)

        # Update annotations to reflect the resized image
        bboxes[:, [0, 2]] *= scale_x  # Scale x_min and x_max
        bboxes[:, [1, 3]] *= scale_y  # Scale y_min and y_max

        sample = {"image": resized_image, "annotations": annotations, }
        return sample


def annotations_to_tensor(annotations, bbox_length=BBOX_LENGTH):
    annotations_list = []
    for annotation in annotations:
        if annotation == 0:
            continue
            #annotations_list.append([0]*(bbox_length + 3)) # extra 3 padding represents id, category_id, image_id
        else:
            bbox = list(annotation.values())[3]
            identifiers = list(annotation.values())[:3]  # id, category_id, image_id
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
        """
        :param sample: A dictionary containing image, and annotations.
        :return: (Tuple), The image in tensor form and the annotations.
        """

        image, annotations = sample["image"], sample["annotations"]
        to_tensor = transforms.ToTensor()
        image_tensor = to_tensor(image)
        #annotations_tensor = annotations_to_tensor(annotations)

        return image_tensor, annotations


# perform some transformations like resizing,
# centering and tensor conversion
# using transforms function
transform = transforms.Compose([
    # transforms.Resize((300, 300)),
    Resize((300, 300)),
    ToTensor()
])


def reformat_bbox(bbox):
    """
    Converts COCO bounding box: [x_min, y_min, width, height] to pytorch bounding box: [x_min, y_min, x_max, y_max].
    :param bbox: (List), [x_min, y_min, width, height].
    :return: (List), [x_min, y_min, x_max, y_max].
    """
    x_min, y_min, width, height = bbox
    x_max, y_max = x_min + width, y_min + height
    return [x_min, y_min, x_max, y_max]


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
        self.image_annotations = data["images"]  # list of dicts

    def __len__(self):
        return len(self.image_annotations)

    def __getitem__(self, idx):
        """

        :param idx: (Int) Idx is equal to the image_id.
        :return: (Tuple), image and annotations. Annotations is a dictionary containing image_id, boxes, and labels.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(idx)

        image_name = self.image_annotations[idx]["file_name"]
        image_path = os.path.join(self.train_images_root, image_name)
        image = Image.open(image_path)

        all_annotations = self.train_annotations
        # Converts bbox (from COCO) for image, and stores it in the form [x_min, y_min, x_max, y_max]
        bboxes = [reformat_bbox(annotation['bbox']) for annotation in all_annotations if annotation["image_id"] == idx]
        labels = [annotation['category_id'] for annotation in all_annotations if annotation["image_id"] == idx]

        annotations = {
            # Needs to be dtype int64 otherwise model throws error
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "boxes": torch.tensor(bboxes, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.int64)  # category id
        }

        # padding = [0] * (MAX_ANNOTATIONS - len(annotations))
        # annotations.extend(padding)

        #sample = {'image': image, 'annotations': annotations}

        # If necessary, perform transforms (e.g., resize, normalize) using custom transform
        sample = {"image": image, "annotations": annotations}  # Combine, as compose only accepts 1 argument
        if self.transform:
            image, annotations = self.transform(sample)  # Transformed sample

        return image, annotations


def convert_annotations_tensor(annotations_tensor, max_annotations=MAX_ANNOTATIONS):
    annotations_list = annotations_tensor.tolist()[0]
    anno_list = []
    for annotation in annotations_list:
        if not is_padding(annotation):
            anno_dict = {"id": int(annotation[0]), "category_id": int(annotation[1]), "image_id": int(annotation[2]),
                         "bbox": annotation[3:]}
            anno_list.append(anno_dict)
    length = len(anno_list)
    # anno_list.extend([0]*(max_annotations - length))
    return anno_list


