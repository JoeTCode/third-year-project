# credit to https://www.geeksforgeeks.org/loading-data-in-pytorch/
# credit to https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# credit to https://medium.com/@piyushkashyap045/transfer-learning-in-pytorch-fine-tuning-pretrained-models-for-custom-datasets-6737b03d6fa2#:~:text=limited%20hardware%20resources.-,Loading%20Pre%2DTrained%20Models%20in%20PyTorch,models%20module.
# credit to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
import math

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import CocoDetection
from custom_dataset import AnprCocoDataset, Resize, ToTensor
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from config import EPOCHS

train_root = "/Users/joe/Desktop/eu-dataset/train/images"
train_annotations_file = "eu-train-dataset-coco.json"
# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# Load the COCO dataset
# dataset = CocoDetection(root=train_root,
#                         annFile=train_annotations_file,
#                         transform=transform)

# perform some transformations like resizing,
# centering and tensor conversion
# using transforms function
transform = transforms.Compose([
        Resize((300, 300)),
	    ToTensor()
     ])

dataset = AnprCocoDataset(
    train_annotations_file_path=train_annotations_file,
    train_images_root=train_root,
    transform=transform
)

# def collate_fn(batch):
#
#     images = []
#     targets = []
#
#     for img, ann in batch:
#         print(img.size(), len(ann))
#         images.append(img)
#         targets.append(ann)  # Keep annotations as a list
#
#     images = torch.stack(images, dim=0)  # Stack images into a batch tensor
#     return images, targets  # Keep targets as a list (does not need padding)

def collate_fn(batch):
    images = []
    targets = []

    for sample in batch:
        images.append(sample[0])  # Image tensor
        targets.append(sample[1])  # List of annotations (dictionary)

    images = torch.stack(images, dim=0)  # Stack images into a batch
    return images, targets  # Targets remain as a list of dicts (not a tensor)


# now use dataloader function load the
# dataset in the specified transformation.
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

# Load the model with pretrained weights
model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

num_classes = 2  #  1 class (license plate) + background
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

# model.head.classification_head.num_classes = num_classes

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
class_loss_function = CrossEntropyLoss()
bbox_loss_function = MSELoss()

best_loss = math.inf
for epoch in range(EPOCHS):

    bbox_weight = 1
    labels_weight = 1
    total_epoch_loss = 0
   
    for i_batch, sample_batched in enumerate(train_dataloader):
        images = sample_batched[0]
        annotations = sample_batched[1]

        # Move images and model to device (CPU or GPU)
        images = images.to(device)
        model = model.to(device)

        # Remove image_id from annotations, and move bboxes and labels to device
        targets = [{key: value.to(device) for key, value in annotation.items() if key != 'image_id'} for annotation in annotations]

        # Check the targets list of dictionaries
        print('targets',targets)

        # Print the image dimensions, the shape of the bounding boxes for the first image,
        # and the shape of the labels for the first image
        # e.g. torch.Size([2, 3, 300, 300]), torch.Size([1, 4]), torch.Size([1])
        # (NUM_BATCH, C, H, W), (NUM_BOXES (PER IMAGE), BBOX FORMAT), (NUM_BOXES)
        print(images.size(), targets[0]['boxes'].size(), targets[0]['labels'].size())

        # Set model to train
        model.train()
        # perform a forward pass and calculate the training loss
        loss_dict = model(images, targets)
        print(loss_dict)

        bbox_loss = loss_dict['bbox_regression']
        class_loss = loss_dict['classification']

        # Can just do predictions['classification'] + predictions['bbox_regression']
        total_loss = (bbox_weight * bbox_loss) + (labels_weight * class_loss)

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    average_epoch_loss = total_epoch_loss / len(train_dataloader)
    if average_epoch_loss < best_loss:
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