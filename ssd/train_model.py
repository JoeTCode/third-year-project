# credit to https://www.geeksforgeeks.org/loading-data-in-pytorch/
# credit to https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# credit to https://medium.com/@piyushkashyap045/transfer-learning-in-pytorch-fine-tuning-pretrained-models-for-custom-datasets-6737b03d6fa2#:~:text=limited%20hardware%20resources.-,Loading%20Pre%2DTrained%20Models%20in%20PyTorch,models%20module.
# credit to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import time
from itertools import zip_longest

import torch
import torch.optim as optim
import torchvision
from six import print_
from torchvision import transforms
from custom_yolo_dataset_loader import AnprYoloDataset, Resize, ToTensor
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from config import config
from torchmetrics.detection import MeanAveragePrecision
from show_predictions import map_bbox_to_image, filter_model_predictions
from torchvision.ops import nms
import os

def compute_confusion_counts(predicted_labels, target_labels):
    """

    :param predicted_labels:
        Takes a 1d list of predicted labels. Predicted labels length can be equal to, larger than, or less than target
        labels due to the nature of the filtering.
    :param target_labels: Takes a 1d list of target labels
    :return:
    """
    TP, FP, TN, FN, = 0, 0, 0, 0

    # Convert to lists if tensors
    if isinstance(predicted_labels, torch.Tensor):
        predicted_labels = predicted_labels.tolist()
    if isinstance(target_labels, torch.Tensor):
        target_labels = target_labels.tolist()

    assert len(target_labels) > 0, "Invalid target labels"

    if len(predicted_labels) == 0:
        return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    # Don't need to check if len(pred) < len(target)
    for pred, target in zip_longest(predicted_labels, target_labels, fillvalue=-1):
        if pred == -1: # If predictions is shorter
            FN += 1
            continue
        if target == -1: # If target is shorter
            FP += 1
            continue

        if pred == target == 1:
            TP += 1
        elif pred == target == 0:
            TN += 1
        elif pred != target:
            if pred == 1 and target == 0:
                FP += 1
            else:  # pred == 0 and target == 1
                FN += 1

    return {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

def print_confusion_matrix(TP, TN, FP, FN, pretty_print=False):
    total = TP + TN + FP + FN
    if pretty_print:
        print("\nConfusion Matrix:")
        print(f"{'':<18}{'Predicted: LP (1)':<20}{'Predicted: BG (0)'}")
        print(f"{'Target: LP (1)':<25}{TP:<21}{FN}")
        print(f"{'Target: BG (0)':<25}{FP:<21}{TN}")
    else: print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN} - Total = {total}")


if config.HPC: SAVE_CHECKPOINTS_DIRECTORY = '/gpfs/home/hyg22ktu/train-ssd/ssd-weights'
else: SAVE_CHECKPOINTS_DIRECTORY = './ssd_weights'

def create_checkpoints_sub_directory(save_directory_path):
    folder_number = 1
    checkpoint_directory_name = 'ssd_weights_'

    if not os.path.exists(save_directory_path):
        os.makedirs(save_directory_path)

    checkpoints = os.listdir(save_directory_path)
    if len(checkpoints) > 0:
        numbers = [int(checkpoint.split('_')[2]) for checkpoint in checkpoints]
        numbers.sort()
        folder_number = numbers[-1] + 1

    checkpoint_directory_name += str(folder_number)
    os.makedirs(os.path.join(save_directory_path, checkpoint_directory_name))
    return checkpoint_directory_name


CHECKPOINTS_SUB_DIRECTORY = create_checkpoints_sub_directory(SAVE_CHECKPOINTS_DIRECTORY)

# train on the GPU, or on the CPU if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)  # Check the CUDA version PyTorch is built with
    print("GPU count:", torch.cuda.device_count())  # Number of GPUs available
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print('Device used:', device)

# Perform transformations using transforms function
transform = transforms.Compose([
    Resize((300, 300)),
    ToTensor()
])

train_dataset = AnprYoloDataset(
    annotations_root=config.TRAIN_ANNOTATIONS_ROOT,
    images_root=config.TRAIN_IMAGES_ROOT,
    transform=transform,
    mosaic=True
)

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
    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=1, pin_memory=True, collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=1, pin_memory=True, collate_fn=collate_fn
    )
else:
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn
    )
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

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

best_map = 0
patience_counter = 0
model = model.to(device)

total_steps = len(train_dataloader)  # Finds the number of batches
log_interval = total_steps // config.NUM_LOGS  # Find the intervals at which to print the log

if log_interval == 0:  # Prevents modulo error if total_steps < NUM_LOGS meaning log_interval will equal zero
    log_interval = 1

for epoch in range(config.EPOCHS):
    total_images = 0
    total_targets = 0
    total_preds = 0
    TP, FP, TN, FN = 0, 0, 0, 0

    start_time_for_epoch = time.time()
    total_epoch_loss = 0
    time_for_log_interval = 0
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

        total_loss = (config.BBOX_WEIGHT * bbox_loss) + (config.LABELS_WEIGHT * class_loss)

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        total_epoch_loss += total_loss.item()
        time_for_log_interval += (time.time() - start_time)

        if i_batch % log_interval == 0 and config.VERBOSE:
            bbox_regression = round(float(list(loss_dict.values())[0]), 4)
            classification = round(float(list(loss_dict.values())[1]), 4)

            print(f'epoch: {epoch:<3}'
                  f'step: {i_batch + 1}/{len(train_dataloader) + 1:<6}'
                  f'bbox_regression: {bbox_regression}, classification: {classification:<6}'
                  f'    time: {round(time_for_log_interval, 2)} s')
            time_for_log_interval = 0

    time_for_epoch = time.time() - start_time_for_epoch
    average_epoch_loss = total_epoch_loss / len(train_dataloader)
    print(f'Average training loss for epoch {epoch + 1}: {average_epoch_loss:<4}'
          f'  time: {round(time_for_epoch, 2)} s')

    # Step scheduler after epoch
    lr_scheduler.step()


    evaluation_start_time = time.time()
    # switch off autograd
    with torch.no_grad():
        metric = MeanAveragePrecision(iou_type="bbox").to(device)  # Initialise and move metric to device
        # model = model.to(device)
        model.eval()  # set model to evaluation mode

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

            for i, image in enumerate(images):
                predicted_bboxes = predictions[i]['boxes']
                predicted_scores = predictions[i]['scores']
                predicted_labels = predictions[i]['labels']
                target_bboxes = targets[i]['boxes']
                target_labels = targets[i]['labels']

                total_images += len(images)
                total_targets += len(target_labels)
                total_preds += len(predicted_labels)

                filtered_bboxes, filtered_scores, filtered_labels = filter_model_predictions(predicted_bboxes, predicted_scores, predicted_labels)
                count_dict = compute_confusion_counts(filtered_labels, target_labels)

                TP += count_dict['TP']
                TN += count_dict['TN']
                FP += count_dict['FP']
                FN += count_dict['FN']

            if not config.HPC:
                # Generate image only when the eval is at its last batch
                if i_batch == len(valid_dataloader) - 1:
                    # Generate and store visualised predictions
                    map_bbox_to_image(images, targets, predictions,
                                      config.SAVE_IMAGE_DIRECTORY, save=False)

        # Compute final mAP over all batches
        metrics = metric.compute()
        map, map_50, map_75 = metrics['map'].item(), metrics['map_50'].item(), metrics['map_75'].item()
        evaluation_time = time.time() - evaluation_start_time
        print(f'mAP: {(map):<4}'
              f'  mAP-50: {map_50:<4}'
              f'  mAP-75: {map_75:<4}'
              f'  time: {round(evaluation_time, 2)} s')
        print_confusion_matrix(TP, TN, FP, FN)

        if map > best_map + config.MAP_MIN_DIFFERENCE:
            best_map = map
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': average_epoch_loss,
            }
            save_path = os.path.join(SAVE_CHECKPOINTS_DIRECTORY, CHECKPOINTS_SUB_DIRECTORY, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(checkpoint, save_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"No mAP improvement, patience limit reached. Stopping at epoch {epoch}")
                break