# EU: 0, Non-EU: 1

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.transforms import ToPILImage


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root='test', transform=transform)

print(test_dataset.class_to_idx)

model = models.mobilenet_v2()

# Modify classification head
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load('save_weights/mobilenet_v2_weights_1.pth'))
model.eval()


def unnormalise(img_tensor, mean, std):
    # Convert mean and std to tensors for broadcasting
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)

    # Undo normalization: img = img * std + mean
    img_tensor = img_tensor * std + mean

    # Clamp to [0,1]
    img_tensor = torch.clamp(img_tensor, 0, 1)

    return img_tensor


# Example usage:
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

with torch.no_grad():
    for image, label in test_dataset:
        output = model(image.unsqueeze(0).to(device))
        _, predicted = torch.max(output, 1)
        class_idx = predicted.item()

        classes = ['EU', 'Non-EU']
        print(f"Prediction: {classes[class_idx]}, Actual: {classes[label]}")
        if class_idx != label:
            unnormalised = unnormalise(image, mean, std)
            img = ToPILImage()(unnormalised)
            img.show()


