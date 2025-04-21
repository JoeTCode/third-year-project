import time
import torch
import torchvision
from torchvision import transforms
from ssd.custom_yolo_dataset_loader import AnprYoloDataset, Resize, ToTensor, train_transform, validation_transform
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from config import config
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from ssd.show_predictions import map_bbox_to_image

# perform inference on the GPU, or on the CPU if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Perform transformations using transforms function
transform = transforms.Compose([
    Resize((300, 300)),
    ToTensor()
])

valid_dataset = AnprYoloDataset(
    annotations_root=config.VALID_ANNOTATIONS_ROOT,
    images_root=config.VALID_IMAGES_ROOT,
    transform=validation_transform
)

train_dataset = AnprYoloDataset(
    annotations_root=config.VALID_ANNOTATIONS_ROOT,
    images_root=config.VALID_IMAGES_ROOT,
    transform=train_transform,
    mosaic=True
)


def collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])  # Image tensor
        targets.append(sample[1])  # List of annotations (dictionary)

    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, targets  # Targets remain as a list of dicts (not a tensor)

# Use dataloader function load the dataset in the specified transformation.
if config.HPC:
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn
    )
else:
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

# Load the model with pretrained weights
model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

# 2 classes: license plate = 1, background = 0
num_classes = 2

# Retrieve the list of input channels.
in_channels = _utils.retrieve_out_channels(model.backbone, (300, 300))

# List containing number of anchors based on aspect ratios.
num_anchors = model.anchor_generator.num_anchors_per_location()

# The classification head.
model.head.classification_head = SSDClassificationHead(
    in_channels=in_channels,
    num_anchors=num_anchors,
    num_classes=num_classes,
)

model = model.to(device)

model.eval()
# image_id = torch.tensor(0, dtype=torch.int64)
# bboxes = torch.tensor([[ 22.5312, 143.1797, 111.1250, 200.1328]], dtype=torch.float32)
# labels = torch.tensor([1], dtype=torch.int64)
# bboxes.to(device)
# labels.to(device)
# annotations = {
#     'boxes': bboxes,
#     'labels': labels,
# }

# pil = Image.open('mosaic-test.png')
# img = transforms.ToTensor()(pil)
# predictions = model([img])

for i, sample in enumerate(train_dataloader):
    images = sample[0];
    annotations = sample[1]
    loss_dict = model(images, annotations)
    if i == 1:
        print(loss_dict)
        break
