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
from ssd.config.config import EPOCHS, HPC, PRINT_FREQ, IMPROVEMENT_FACTOR, TRAIN_IMAGES_ROOT, TRAIN_ANNOTATIONS_ROOT, BATCH_SIZE

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device used: ', device)

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

# Use dataloader function load the dataset in the specified transformation.
if HPC:
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                              collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True,)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)

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
for epoch in range(EPOCHS):

    bbox_weight = 1
    labels_weight = 1
    total_epoch_loss = 0

    for i_batch, sample_batched in enumerate(train_dataloader):
        start_time = time.time()
        images = sample_batched[0]
        annotations = sample_batched[1]

        # Move images and model to device (CPU or GPU)
        images = images.to(device)
        model = model.to(device)

        # Remove image_id from annotations, and move bboxes and labels to device
        targets = [{key: value.to(device) for key, value in annotation.items() if key != 'image_id'} for annotation in
                   annotations]

        # Print the image dimensions, the shape of the bounding boxes for the first image,
        # and the shape of the labels for the first image
        # e.g. torch.Size([2, 3, 300, 300]), torch.Size([1, 4]), torch.Size([1])
        # (NUM_BATCH, C, H, W), (NUM_BOXES (PER IMAGE), BBOX FORMAT), (NUM_BOXES)
        # print(images.size(), targets[0]['boxes'].size(), targets[0]['labels'].size())

        # Set model to train
        model.train()
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
        lr_scheduler.step()

        if i_batch % PRINT_FREQ == 0:
            print('epoch:', epoch,
                  '\tstep:', i_batch + 1, '/', len(train_dataloader) + 1,
                  '\ttrain loss:', '{:.4f}'.format(loss_dict.item()),
                  '\ttime:', '{:.4f}'.format((time.time() - start_time) * PRINT_FREQ), 's')

    average_epoch_loss = total_epoch_loss / len(train_dataloader)

    if average_epoch_loss < best_loss * IMPROVEMENT_FACTOR:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_epoch_loss,
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch + 1}.pth')

# for i_batch, sample_batched in enumerate(train_dataloader):
#     images = sample_batched[0]
#     annotations = sample_batched[1]
#     print(images.size())
