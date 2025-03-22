# credit to https://www.geeksforgeeks.org/loading-data-in-pytorch/
# credit to https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# credit to https://medium.com/@piyushkashyap045/transfer-learning-in-pytorch-fine-tuning-pretrained-models-for-custom-datasets-6737b03d6fa2#:~:text=limited%20hardware%20resources.-,Loading%20Pre%2DTrained%20Models%20in%20PyTorch,models%20module.
# credit to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import math
import time
import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from custom_yolo_dataset_loader import AnprYoloDataset, Resize, ToTensor
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from ssd.config.config import EPOCHS, HPC, PRINT_FREQ, IMPROVEMENT_FACTOR, TRAIN_IMAGES_ROOT, TRAIN_ANNOTATIONS_ROOT, \
    VALID_IMAGES_ROOT, VALID_ANNOTATIONS_ROOT, BATCH_SIZE
from torchmetrics.detection import MeanAveragePrecision

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("CUDA detected: ", torch.cuda.is_available())  # Should print True if CUDA is detected
print("CUDA version: ", torch.version.cuda)  # Check the CUDA version PyTorch is built with
print("GPU count: ", torch.cuda.device_count())  # Number of GPUs available
print("GPU name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print('Device used:', device)

# Perform transformations using transforms function
transform = transforms.Compose([
    Resize((300, 300)),
    ToTensor()
])

train_dataset = AnprYoloDataset(
    annotations_root=TRAIN_ANNOTATIONS_ROOT,
    images_root=TRAIN_IMAGES_ROOT,
    transform=transform
)

valid_dataset = AnprYoloDataset(
    annotations_root=VALID_ANNOTATIONS_ROOT,
    images_root=VALID_IMAGES_ROOT,
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
if HPC:
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True, collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn
    )
else:
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn
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

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

best_loss = math.inf
bbox_weight = 1
labels_weight = 1
model = model.to(device)

for epoch in range(EPOCHS):

    total_epoch_loss = 0
    time_till_print = 0
    model.train()  # Set model to train

    for i_batch, sample_batched in enumerate(train_dataloader):

        start_time = time.time()
        images = sample_batched[0]
        annotations = sample_batched[1]

        # Move images to device (CPU or GPU)
        images = images.to(device)

        # Remove image_id from annotations, and move bboxes and labels to device
        targets = [
            {key: value.to(device) for key, value in annotation.items() if key != 'image_id'} for annotation
            in annotations
        ]

        # Perform a forward pass and calculate the training loss
        loss_dict = model(images, targets)

        bbox_loss = loss_dict['bbox_regression']
        class_loss = loss_dict['classification']

        # Can just do predictions['classification'] + predictions['bbox_regression']
        total_loss = (bbox_weight * bbox_loss) + (labels_weight * class_loss)

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        total_epoch_loss += total_loss.item()
        time_till_print += (time.time() - start_time)

        if i_batch % PRINT_FREQ == 0:
            multiplier = PRINT_FREQ if i_batch != 0 else 1
            bbox_regression = round(float(list(loss_dict.values())[0]), 4)
            classification = round(float(list(loss_dict.values())[1]), 4)

            print(f'epoch: {epoch:<3}'
                  f'step: {i_batch + 1}/{len(train_dataloader) + 1:<6}'
                  f'bbox_regression: {bbox_regression}, classification: {classification:<6}'
                  f'    time: {round(time_till_print, 2)} s')
            time_till_print = 0

    # Step scheduler after epoch
    lr_scheduler.step()

    # switch off autograd
    with torch.no_grad():
        metric = MeanAveragePrecision(iou_type="bbox").to(device)  # Initialise and move metric to device
        model.eval()  # set model to evaluation mode
        model = model.to(device)

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

        # Compute final mAP over all batches
        metrics = metric.compute()
        map, map_50, map_75 = metrics['map'].item(), metrics['map_50'].item(), metrics['map_75'].item()

        print(f'mAP: {(map):<4}'
              f'  mAP-50: {map_50:<4}'
              f'  mAP-75: {map_75}')

    average_epoch_loss = total_epoch_loss / len(train_dataloader)
    print('Average training loss for epoch:', average_epoch_loss)

    if average_epoch_loss < best_loss * IMPROVEMENT_FACTOR:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_epoch_loss,
        }

        torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')
