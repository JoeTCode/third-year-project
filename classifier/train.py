# EU: 0, Non-EU: 1
# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
from torchvision.models import MobileNet_V2_Weights
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 2
batch_size = 8
num_epochs = 10
learning_rate = 0.001


transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.8),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Resize(224, 224),
        A.Normalize(),
        ToTensorV2()
    ]
)

train_dataset = datasets.ImageFolder(root='train', transform=transform)
val_dataset = datasets.ImageFolder(root='val', transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Adjust classification head
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_val_accuracy = 0
patience = 3
difference_multiplier = 1.01 # 1%
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_epoch_loss = 0
    total_preds = 0
    total_correct = 0

    for images, labels in tqdm(train_dataloader):
        images, labels = images.to(device), labels.to(device)

        # forward
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate loss for whole batch
        total_epoch_loss += loss.item() * images.size(0)

        total_correct += (predicted == labels).sum().item()
        total_preds += labels.size(0)

    train_loss = total_epoch_loss / total_preds
    train_accuracy = total_correct / total_preds

    val_correct = 0
    val_total = 0
    model.eval()

    val_accuracy = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

        val_accuracy = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.2f}, Train Acc: {train_accuracy:.2f},  Val Acc: {val_accuracy:.2f}"
        )

    if val_accuracy > best_val_accuracy * difference_multiplier:
        best_val_accuracy = val_accuracy
        patience_counter = 0

        save_path = os.path.join('save_weights', f'mobilenet_v2_weights_{epoch + 1}.pth')
        torch.save(model.state_dict(), save_path)

    else:
        patience_counter +=1
        if patience_counter >= patience:
            print(f"No accuracy improvement, patience limit reached. Stopping at epoch {epoch}")
            break
