# credit to https://www.geeksforgeeks.org/loading-data-in-pytorch/
# credit to https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
# credit to https://medium.com/@piyushkashyap045/transfer-learning-in-pytorch-fine-tuning-pretrained-models-for-custom-datasets-6737b03d6fa2#:~:text=limited%20hardware%20resources.-,Loading%20Pre%2DTrained%20Models%20in%20PyTorch,models%20module.

import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import CocoDetection
from custom_dataset import AnprCocoDataset, Resize, ToTensor
from torch.utils.data import DataLoader
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
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

# now use dataloader function load the
# dataset in the specified transformation.
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Load the model with pretrained weights
model = torchvision.models.detection.ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)

num_classes = 2  #  1 class (license plate) + background
model.head.classification_head.num_classes = num_classes

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
class_loss_function = CrossEntropyLoss()
bbox_loss_function = MSELoss()


for epoch in range(1):
    # Set model to train
    model.train()

    # Initialise the training loss
    train_loss = 0
    # Initialise correct predictions
    train_correct = 0
    for images, targets in train_dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        print(outputs)
        # class_loss = class_loss_function(outputs[0], targets)
        # bbox_loss = bbox_loss_function(outputs[1], targets)


# for images, targets in train_dataloader:
#     print(f"images: {images}\n targets: {targets}")
#     break

"""
# iter function iterates through all the
# images and labels and stores in two variables
images, labels = next(iter(train_dataloader))

print('')
# print the total no of samples
print(f'Number of samples: {len(images)}')
image = images[2][0] # load 3rd sample

# visualize the image
plt.imshow(image, cmap='gray')
plt.show()

# print the size of image
print("Image Size: ", image.size())
"""