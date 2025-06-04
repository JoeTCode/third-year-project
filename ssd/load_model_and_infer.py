import time
import torch
import torchvision
from custom_yolo_dataset_loader import AnprYoloDataset, ToTensor, validation_transform
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from config import config
from torchmetrics.detection import MeanAveragePrecision
from show_predictions import map_bbox_to_image
from tqdm import tqdm

# perform inference on the GPU, or on the CPU if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

weights_path = 'ssd_checkpoints/checkpoint_epoch_25.pth'

test_dataset = AnprYoloDataset(
    annotations_root=config.TEST_ANNOTATIONS_ROOT,
    images_root=config.TEST_IMAGES_ROOT,
    transform=validation_transform
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
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn
    )
else:
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn
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
checkpoint = torch.load(weights_path, map_location=torch.device(device))
# Pass loaded states to model
model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

model.eval()

metric = MeanAveragePrecision(iou_type="bbox").to(device)  # Initialise and move metric to device

evaluation_start_time = time.time()
# Loop over the validation set
with torch.no_grad():
    for i_batch, sample_batched in enumerate(tqdm(test_dataloader)):
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

# Compute final mAP over all batches
metrics = metric.compute()
map, map_50, map_75 = metrics['map'].item(), metrics['map_50'].item(), metrics['map_75'].item()
evaluation_time = time.time() - evaluation_start_time
num_images = len(test_dataset)
average_time_per_image = evaluation_time/num_images
print(f'mAP: {(map):<4}'
      f'  mAP-50: {map_50:<4}'
      f'  mAP-75: {map_75:<4}'
      f'  time: {round(evaluation_time, 2)} s'
      f'  Average inference time per image: {round(average_time_per_image*1000, 2)} ms'
      )


