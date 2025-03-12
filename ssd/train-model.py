# credit to https://www.geeksforgeeks.org/loading-data-in-pytorch/
# credit to https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# credit to https://medium.com/@piyushkashyap045/transfer-learning-in-pytorch-fine-tuning-pretrained-models-for-custom-datasets-6737b03d6fa2#:~:text=limited%20hardware%20resources.-,Loading%20Pre%2DTrained%20Models%20in%20PyTorch,models%20module.
# credit to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

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

print(model)
# model.head.classification_head.num_classes = num_classes

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
class_loss_function = CrossEntropyLoss()
bbox_loss_function = MSELoss()


for epoch in range(EPOCHS):


    bbox_weight = 1
    labels_weight = 1
    totalLoss = 0
    # Initialise the training loss
    train_loss = 0
    # Initialise correct predictions
    train_correct = 0
   
    for i_batch, sample_batched in enumerate(train_dataloader):
        images = sample_batched[0]
        annotations = sample_batched[1]

        # print(i_batch, images.size(), len(annotations))
        print(type(images))  # This will show the type of 'images'
        print(images.size())  # Check the shape to verify it's a tensor
        images = images.to(device)
        # removed imageid from annotations
        targets = [{key: value.to(device) for key, value in annotation.items() if key != 'image_id'} for annotation in annotations]
        print('targets',targets)
        print(images.size(), targets[0]['boxes'].size(), targets[0]['labels'].size())
        # Set model to train
        model.train()
        # perform a forward pass and calculate the training loss
        predictions = model(images, targets)
        print(predictions)
        print(predictions['classification'].size(), predictions['bbox_regression'].size())

        bbox_targets = [target['boxes'] for target in targets]
        bbox_targets = torch.stack(bbox_targets)  # Stack into a tensor
        labels = [target['labels'] for target in targets]
        labels = torch.stack(labels)
        labels = labels.squeeze()
        print('labels', labels)

        bbox_loss = bbox_loss_function(predictions['bbox_regression'], bbox_targets)
        class_loss = class_loss_function(predictions['classification'], labels)
        total_loss = (bbox_weight * bbox_loss) + (labels_weight * class_loss)

        # zero out the gradients, perform the backpropagation step,
        # and update the weights
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # add the loss to the total training loss so far and
        # calculate the number of correct predictions
        # totalTrainLoss += totalLoss
        # trainCorrect += (predictions['classification'].argmax(1) == labels).type(
        #     torch.float).sum().item()
        


# for i_batch, sample_batched in enumerate(train_dataloader):
#     images = sample_batched[0]
#     annotations = sample_batched[1]
#     print(images.size())