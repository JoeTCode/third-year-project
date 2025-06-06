import os
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from config import config
from random import choices, randint
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

image_files = [image for image in os.listdir(config.TRAIN_IMAGES_ROOT)]
annotations_files = [annotation for annotation in os.listdir(config.TRAIN_ANNOTATIONS_ROOT)]

def find_overlap_bbox(bbox, crop, overlap_ratio_to_original=0.1):
    """
    Checks if a bbox overlaps with the cropped image, if true,
    resizes (if necessary) and returns the bbox.
    :param bbox: Original bounding box.
    :param crop: Cropped image.
    :param overlap_ratio_to_original:
        (Float) The acceptable overlap ratio of the cropped bounding box
        compared to the original.
    :return: The cropped bounding box.
    """
    overlap_bbox = None
    x_min, y_min, x_max, y_max = bbox
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = crop

    x_overlap_min = max(x_min, crop_x_min)
    y_overlap_min = max(y_min, crop_y_min)
    x_overlap_max = min(x_max, crop_x_max)
    y_overlap_max = min(y_max, crop_y_max)

    if x_overlap_max > x_overlap_min and y_overlap_max > y_overlap_min:
        # Overlap exists
        overlap_bbox = [x_overlap_min - crop_x_min, y_overlap_min - crop_y_min,
                        x_overlap_max - crop_x_min, y_overlap_max - crop_y_min]
        # Adjust bounding box edges to be fully inside the cropped image
        if int(overlap_bbox[0]) == 0:
            overlap_bbox[0] = 1
        if int(overlap_bbox[1]) == 0:
            overlap_bbox[1] = 1
        if int(overlap_bbox[2]) == 300:
            overlap_bbox[2] = 299
        if int(overlap_bbox[3]) == 300:
            overlap_bbox[3] = 299

    if overlap_bbox:
        overlap_width = overlap_bbox[2] - overlap_bbox[0]
        overlap_height = overlap_bbox[3] - overlap_bbox[1]
        overlap_bbox_area = overlap_width * overlap_height
        overlap_bbox_area = float(overlap_bbox_area)

        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]
        bbox_area = bbox_width * bbox_height
        bbox_area = float(bbox_area)

        if overlap_bbox_area/bbox_area < overlap_ratio_to_original:
            return None

    return overlap_bbox

def create_mosaic(images, annotations, idx=0):
    # Doesn't add all image ids (random_idx) to image_id, only the one image_id = idx
    """

    :param idx: Index of the first image in the mosaic
    :param images:
        PIL image list in the form [image, image, image, image]. These will be stitched together to make a 2x2 image
        and 600x600 dimension mosaic.
    :param annotations:
        List of dictionaries in the form [{'image_id': tensor(1), 'boxes': tensor([[bbox], [bbox]]), 'labels': tensor([1, 1])}]
        corresponding to each image.
    :return:
        Returns an image tensor and its annotations. The image is cropped mosaic of size 300x300. The annotation is a
        single annotation's dictionary. If the cropped image has no plate detections, the annotation label should be
        set as background.
    """

    for i, image in enumerate(images):
        w, h = image.size
        if w > 300 or h > 300:
            # resized_dict = Resize((300,300))({"image": image, "annotations": annotations[i]})
            resized = resize(image=np.array(image), bboxes=annotations[i]['boxes'], labels=annotations[i]['labels'])
            resized_image, resized_boxes = resized['image'], resized['bboxes']
            resized_pil_image = Image.fromarray(resized_image)
            # resized_image, resized_annotations = resized_dict["image"], resized_dict["annotations"]
            images[i] = resized_pil_image
            annotations[i]['boxes'] = resized_boxes

    im1, im2, im3, im4 = images

    total_width = im1.width + im2.width + im3.width + im4.width
    total_height = im1.height + im2.height + im3.height + im4.height
    # Canvas to stitch together images
    # Will have size of 600x600 - will crop to make it 300x300
    mosaic = Image.new('RGB', (int(total_width/2), int(total_height/2))) # 2x2 mosaic
    mosaic.paste(im1, (0, 0))
    mosaic.paste(im2, (im1.width, 0))
    mosaic.paste(im3, (0, im1.height))
    mosaic.paste(im4, (im1.width, im1.height))

    mosaic_bboxes = [[], [], [], []]

    for image_index, annotation in enumerate(annotations):
        bboxes = annotation['boxes']
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            if isinstance(x_min, torch.Tensor):
                x_min = x_min.item()
                y_min = y_min.item()
                x_max = x_max.item()
                y_max = y_max.item()

            if image_index == 0:
                mosaic_bboxes[0].append([x_min, y_min, x_max, y_max])

            if image_index == 1:
                x_min += im1.width
                x_max += im1.width
                mosaic_bboxes[1].append([x_min, y_min, x_max, y_max])

            if image_index == 2:
                y_min += im1.height
                y_max += im1.height
                mosaic_bboxes[2].append([x_min, y_min, x_max, y_max])
            if image_index == 3:
                x_min += im1.width
                x_max += im1.width
                y_min += im1.height
                y_max += im1.height
                mosaic_bboxes[3].append([x_min, y_min, x_max, y_max])

            # draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=1)

    # Resulting random crop is 300x300 in size
    crop_y_min, crop_x_min, crop_height, crop_width = transforms.RandomCrop.get_params(mosaic, output_size=(300, 300))
    crop_y_max = crop_y_min + crop_height
    crop_x_max = crop_x_min + crop_width
    cropped_mosaic = mosaic.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
    # cropped_mosaic_draw = ImageDraw.Draw(cropped_mosaic)

    # Adjust bboxes after stitching images together into mosaic
    final_bboxes = []
    final_labels = []
    for i, bboxes in enumerate(mosaic_bboxes):
        for j, bbox in enumerate(bboxes):
            overlap_bbox = find_overlap_bbox(bbox, (crop_x_min, crop_y_min, crop_x_max, crop_y_max))
            if overlap_bbox:
                final_bboxes.append(overlap_bbox)
                final_labels.append(1) # Bbox is valid, add label

    # If there are no detections in the cropped image, then set the label to background, and insert dummy bbox
    if len(final_bboxes) == 0:
        print("MOSAIC ASSIGNED BACKGROUND")
        final_bboxes = [[0, 0, 1, 1]]
        final_labels = [0]

    final_annotations = {
        "image_id": torch.tensor(idx, dtype=torch.int64),
        "boxes": torch.tensor(final_bboxes, dtype=torch.float),
        "labels": torch.tensor(final_labels, dtype=torch.int64)
    }

    #cropped_mosaic.show()

    return cropped_mosaic, final_annotations


def reformat_bbox(bbox, image_height, image_width):
    """
    Converts YOLO bounding box: [x_center, y_center, width, height] to PASCAL VOC bounding box: [x_min, y_min, x_max, y_max].
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

def get_PIL_image(image_root_path, image_file_path, idx):
    image_path = os.path.join(image_root_path, image_file_path[idx])
    return Image.open(image_path)

def get_matching_annotations(annotations_root_path, annotations_file_path, idx, image_height, image_width):
    file_path = os.path.join(annotations_root_path, annotations_file_path[idx])
    f = open(file_path, 'r')
    lines = f.readlines()

    # Bounding boxes
    bboxes = [reformat_bbox(line.split(" ")[1:], image_height, image_width) for line in lines]

    # Labels
    if len(bboxes) == 0:
        labels = [0]  # If no bboxes are found, make the image a background class image (0)
        bboxes = [[0, 0, 1, 1]]  # Create a small bbox for background image
    else:
        labels = [int(line.split(" ")[0]) + 1 for line in lines]

    f.close()
    return bboxes, labels

def apply_transform(image, annotations, transform):
    """

    :param image: One (PIL) image.
    :param annotations: Corresponding annotation dict.
    :param transform: Transform function (contains a series of transformations to be applied to the image and annotations).
    :return: (Tuple) Transformed image and annotations.
    """
    np_image = np.array(image)
    bboxes = annotations['boxes']
    labels = annotations['labels']

    if not isinstance(bboxes, list):
        bboxes = bboxes.tolist()
    if not isinstance(labels, list):
        labels = labels.tolist()

    augmented = transform(image=np_image, bboxes=bboxes, labels=labels)

    # Extract the augmented results
    image = augmented["image"]
    bboxes = augmented["bboxes"]
    labels = augmented["labels"]

    # Update annotations with transformed values
    annotations["boxes"] = torch.tensor(bboxes, dtype=torch.float32)
    annotations["labels"] = torch.tensor(labels, dtype=torch.int64)

    return image, annotations


class AnprYoloDataset(Dataset):
    def __init__(self, annotations_root, images_root, transform=None, mosaic=False, test=False):
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
        self.mosaic = mosaic
        self.test = test

    def __len__(self):
        return len(self.annotations_files)

    def __getitem__(self, idx):
        """

        :param idx: (Int) Idx is equal to the image_id.
        :return: (Tuple), image and annotations. Annotations is a dictionary containing image_id, boxes, and labels.
        """

        if self.test:
            return get_PIL_image(self.images_root, self.image_files, idx)

        generate_mosaic = False

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mosaic:
            mosaic_true_prob = config.TRAIN_MOSAIC_PROBABILITY
            mosaic_false_prob = 1 - mosaic_true_prob
            generate_mosaic = choices([True, False], weights=[mosaic_true_prob, mosaic_false_prob])[0]

        if generate_mosaic:
            annotations_list = []

            images = []
            for i in range(4):
                annotations = {"image_id": idx}
                # Images
                random_idx = randint(0, len(self.image_files) - 1)
                image = get_PIL_image(self.images_root, self.image_files, random_idx)
                image_width, image_height = image.size
                images.append(image)

                # Bboxes and labels
                bboxes, labels = get_matching_annotations(self.annotations_root, self.annotations_files, random_idx, image_height, image_width)
                annotations["boxes"] = bboxes
                annotations["labels"] = labels
                annotations_list.append(annotations)

            mosaic, annotations = create_mosaic(images, annotations_list, idx)

            sample = {"image": mosaic, "annotations": annotations}  # Combine, as compose only accepts 1 argument
            if self.transform:
                mosaic, annotations = apply_transform(sample['image'], sample['annotations'], self.transform)

            if annotations['boxes'].shape[0] == 0:  # If target boxes are empty (after transform) set as background
                annotations['boxes'] = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
                annotations['labels'] = torch.tensor([0], dtype=torch.int64)

            return mosaic, annotations

        # Get image (with index=idx) file path, and convert to PIL image
        image = get_PIL_image(self.images_root, self.image_files, idx)
        image_width, image_height = image.size

        # Get the matching annotation file path (index = idx) and reformat and store the bboxes and labels
        bboxes, labels = get_matching_annotations(self.annotations_root, self.annotations_files, idx, image_height, image_width)

        annotations = {
            "image_id": torch.tensor(idx, dtype=torch.int64),
            "boxes": torch.tensor(bboxes, dtype=torch.float),
            "labels": torch.tensor(labels, dtype=torch.int64)  # category id
        }

        if self.transform: # Albumentations augmentations
            image, annotations = apply_transform(image, annotations, self.transform)

        if annotations['boxes'].shape[0] == 0: # If target boxes are empty (after transform) set as background
            annotations['boxes'] = torch.tensor([[0, 0, 1, 1]], dtype=torch.float)
            annotations['labels'] = torch.tensor([0], dtype=torch.int64)

        return image, annotations


train_transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Resize(300, 300),
        A.Normalize(),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'], # name (key) corresponding to the labels list
        min_visibility=0.1
    )
)

validation_transform = A.Compose(
    [
        A.Resize(300, 300),
        A.Normalize(),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'], # name (key) corresponding to the labels list
        min_visibility=0.1
    )
)

testing_transform = A.Compose( # Not for test dataset, only for debugging purposes
    [
        A.Resize(300, 300),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'], # name (key) corresponding to the labels list
        min_visibility=0.1
    )
)

resize = A.Compose( # Not for test dataset, only for debugging purposes
    [
        A.Resize(300, 300)
    ],
    bbox_params=A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'], # name (key) corresponding to the labels list
        min_visibility=0.1
    )
)

anpr_yolo_dataset = AnprYoloDataset(
        annotations_root=config.TRAIN_ANNOTATIONS_ROOT,
        images_root=config.TRAIN_IMAGES_ROOT,
        transform=train_transform
    )
