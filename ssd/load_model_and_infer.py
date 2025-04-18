import time
import torch
import torchvision
from torchvision import transforms
from custom_yolo_dataset_loader import AnprYoloDataset, Resize, ToTensor
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from config import config
from torchmetrics.detection import MeanAveragePrecision
from show_predictions import map_bbox_to_image

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

# Use dataloader function load the dataset in the specified transformation.
if config.HPC:
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn
    )
else:
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn
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

# Load Checkpoint
checkpoint = torch.load('ssd_checkpoints/checkpoint_epoch_25.pth', map_location=torch.device(device))
# Pass loaded states to model
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

model.eval()

metric = MeanAveragePrecision(iou_type="bbox").to(device)  # Initialise and move metric to device

evaluation_start_time = time.time()
# Loop over the validation set
for i_batch, sample_batched in enumerate(valid_dataloader):
    images = sample_batched[0]
    annotations = sample_batched[1]

    # Move images to device (CPU or GPU)
    images = images.to(device)

    # Remove image_id from annotations, and move bboxes and labels to device
    targets = [
        {key: value.to(device) for key, value in annotation.items() if key != 'image_id'} for annotation
        in annotations
    ]

    # Make the predictions
    predictions = model(images)
    metric.update(predictions, targets)

    # Generate image only when the eval is at its last batch
    #if i_batch == len(valid_dataloader) - 1:
    # pass predictions twice as we access boxes and scores from it\

    if i_batch % config.NUM_LOGS == 0:
        map_bbox_to_image(images, targets, predictions, predictions,
                          config.SAVE_IMAGE_DIRECTORY)

# Compute final mAP over all batches
metrics = metric.compute()
map, map_50, map_75 = metrics['map'].item(), metrics['map_50'].item(), metrics['map_75'].item()
evaluation_time = time.time() - evaluation_start_time
print(f'mAP: {(map):<4}'
      f'  mAP-50: {map_50:<4}'
      f'  mAP-75: {map_75:<4}'
      f'  time: {round(evaluation_time, 2)} s')

