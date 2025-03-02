# credit to https://www.geeksforgeeks.org/loading-data-in-pytorch/

import torch
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

root = "/Users/joe/Desktop/eu-dataset/train/images"
annotations_file = "eu-train-dataset-coco.json"

# perform some transformations like resizing,
# centering and tensor conversion
# using transforms function
transform = transforms.Compose([
        transforms.Resize((224, 224)),
	    transforms.ToTensor()
     ])

# Load the COCO dataset
dataset = CocoDetection(root=root,
                        annFile=annotations_file,
                        transform=transform)

def collate_fn(batch):
    images = []
    targets = []

    for img, ann in batch:
        images.append(img)
        targets.append(ann)  # Keep annotations as a list

    images = torch.stack(images, dim=0)  # Stack images into a batch tensor
    return images, targets  # Keep targets as a list (does not need padding)

# now use dataloader function load the
# dataset in the specified transformation.
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)



# iter function iterates through all the
# images and labels and stores in two variables
images, labels = next(iter(dataloader))

print('')
# print the total no of samples
print(f'Number of samples: {len(images)}')
image = images[2][0] # load 3rd sample

# visualize the image
plt.imshow(image, cmap='gray')
plt.show()

# print the size of image
print("Image Size: ", image.size())
