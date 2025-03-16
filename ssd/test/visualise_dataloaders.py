import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ssd.config.config import MAX_ANNOTATIONS, BBOX_LENGTH, TRAIN_IMAGES_ROOT, TRAIN_ANNOTATIONS_ROOT, BATCH_SIZE
from ssd.custom_yolo_dataset_loader import AnprYoloDataset, Resize, ToTensor


def collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])  # Image tensor
        targets.append(sample[1])  # List of annotations (dictionary)

    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, targets  # Targets remain as a list of dicts (not a tensor)


transform = transforms.Compose([
    Resize((300, 300)),
    ToTensor()
])

def show_bbox(image, annotations, transform=None):
    """
    Draws bounding boxes on an image using PIL.
    :param image: The image, either in PIL or tensor form.
    :param annotations: (dict), Containing image_id, boxes, labels.
    :param transform: Indicates if a transform was applied to a sample.
    :return: Displays the image with bounding boxes overlay.
    """
    # print('SHOW_BBOX:' ,image, annotations) # print as tensor

    if transform:
        image = transforms.ToPILImage()(image)  # Convert back to PIL for visualization

    draw = ImageDraw.Draw(image)
    # Draw each bbox
    bboxes = annotations['boxes']
    for i in range(bboxes.shape[0]):
        x_min, y_min, x_max, y_max = bboxes[i]
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # Show image
    image.show()


def test_sample_dataset(dataset):
    for i, sample in enumerate(dataset):
        image, annotations = sample[0], sample[1]

        if isinstance(image, Image.Image):  # If image is type PIL (the images were not transformed)
            print(f"{i}: {image.size} -- {len(annotations)}, {annotations}")
            show_bbox(image, annotations)

        else:  # Images were transformed
            print(f"{i}: {image.size()} -- {len(annotations)}, {annotations}")
            show_bbox(image, annotations, transform=transform)

        if i == 3:
            break


def collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])  # Image tensor
        targets.append(sample[1])  # List of annotations (dictionary)

    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, targets  # Targets remain as a list of dicts (not a tensor)


def test_sample_dataset_new(dataset):
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    to_pil = transforms.ToPILImage()  # Convert tensor to PIL Image

    for i, (images, annotations) in enumerate(train_dataloader):  # Unpack batch
        for j in range(len(images)):  # Iterate through batch
            img = images[j]
            annots = annotations[j]

            # Convert tensor to PIL image if needed
            if isinstance(img, torch.Tensor):
                img = to_pil(img)

            print(f"Sample {i}-{j}: {img.size} -- {len(annots)}, {annots}")

            # Display image with bounding boxes
            show_bbox(img, annots)

        #if i == 3:  # Stop after 4 batches
        break


if __name__ == '__main__':
    anpr_yolo_dataset = AnprYoloDataset(
        annotations_root=TRAIN_ANNOTATIONS_ROOT,
        images_root=TRAIN_IMAGES_ROOT,
        transform=transform
    )
    test_sample_dataset_new(anpr_yolo_dataset)

    # test = [{'id': 0, 'category_id': 0, 'image_id': 0, 'bbox': [127.734375, 172.03124999999997, 19.21875, 10.3125]},
    #         {'id': 0, 'category_id': 0, 'image_id': 0, 'bbox': [127.734375, 172.03124999999997, 19.21875, 10.3125]}, 0, 0, 0]
    # tensor = annotations_to_tensor(test)
    # annotation = convert_annotations_tensor(tensor)
    # print(annotation)
