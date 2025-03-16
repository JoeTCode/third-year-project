import os
import json
import torch
import warnings
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ssd.config.config import MAX_ANNOTATIONS, BBOX_LENGTH, TRAIN_IMAGES_ROOT, TRAIN_ANNOTATIONS_ROOT, BATCH_SIZE

image_files = [image for image in os.listdir(TRAIN_IMAGES_ROOT)]
annotations_files = [annotation for annotation in os.listdir(TRAIN_ANNOTATIONS_ROOT)]


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
        if bboxes.dim() != 1: # Check for images that have no detections (background) and therefore no bboxes
            # Update annotations to reflect the resized image
            bboxes[:, [0, 2]] *= scale_x  # Scale x_min and x_max
            bboxes[:, [1, 3]] *= scale_y  # Scale y_min and y_max

        sample = {"image": resized_image, "annotations": annotations, }
        return sample


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


def reformat_bbox(bbox, image_height, image_width):
    """
    Converts COCO bounding box: [x_min, y_min, width, height] to pytorch bounding box: [x_min, y_min, x_max, y_max].
    :param image_width:
    :param image_height:
    :param bbox: (List), [x_min, y_min, width, height].
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

def extract_ordered_labels(images_root, annotations_root):
    images = []
    annotations = []

    for i, image in enumerate(os.listdir(images_root)):
        images.append(image)
        image_name_without_extension = os.path.splitext(image)[0]

        annotation_file = f"{image_name_without_extension}.txt"

        if annotation_file in os.listdir(annotations_root):
            annotations.append(annotation_file)
        else: print(f"No annotations.txt file found for the image at index: {i}")

    return images, annotations


class AnprYoloDataset(Dataset):
    def __init__(self, annotations_root, images_root, transform=None):
        """

        :param annotations_root: Image annotations.
        :param images_root: Images folder.
        :param transform: Optional transform to be applied on a sample.
        """

        self.annotations_root = annotations_root
        self.images_root = images_root
        self.transform = transform
        images_list, annotations_list = extract_ordered_labels(self.images_root, self.annotations_root)
        self.image_files = images_list
        self.annotations_files = annotations_list


    def __len__(self):
        return len(self.annotations_files)

    def __getitem__(self, idx):
        """

        :param idx: (Int) Idx is equal to the image_id.
        :return: (Tuple), image and annotations. Annotations is a dictionary containing image_id, boxes, and labels.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #print(idx)

        image_path = os.path.join(self.images_root, self.image_files[idx])
        image = Image.open(image_path)
        image_height, image_width = image.size

        file_path = os.path.join(self.annotations_root, self.annotations_files[idx])
        f = open(file_path, 'r')
        lines = f.readlines()
        bboxes = [reformat_bbox(line.split(" ")[1:], image_height, image_width) for line in lines]
        if not lines: labels = [0] # If no bboxes are found, make the image a background class image (0)
        else: labels = [int(line.split(" ")[0]) + 1 for line in lines]
        f.close()

        annotations = {
            # Needs to be dtype int64 otherwise model throws error
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "boxes": torch.tensor(bboxes, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.int64)  # category id
        }

        # If necessary, perform transforms (e.g., resize, normalize) using custom transform
        sample = {"image": image, "annotations": annotations}  # Combine, as compose only accepts 1 argument
        if self.transform:
            image, annotations = self.transform(sample)  # Transformed sample

        return image, annotations


transform = transforms.Compose([
    # transforms.Resize((300, 300)),
    Resize((300, 300)),
    ToTensor()
])

anpr_yolo_dataset = AnprYoloDataset(
        annotations_root=TRAIN_ANNOTATIONS_ROOT,
        images_root=TRAIN_IMAGES_ROOT,
        transform=transform
    )

def collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])  # Image tensor
        targets.append(sample[1])  # List of annotations (dictionary)

    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, targets  # Targets remain as a list of dicts (not a tensor)


# train_dataloader = DataLoader(anpr_yolo_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
# for i_batch, sample_batched in enumerate(train_dataloader):
#     print(i_batch, sample_batched)
