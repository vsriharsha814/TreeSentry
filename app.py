# deforestation_detection.py
# Python script using PyTorch to train a ResNet-50 CNN on the Planet Amazon Rainforest dataset

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# 1. Custom Dataset class
class PlanetDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_labels=None):
        """
        Args:
            csv_file (string): Path to the CSV with image filenames and labels.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transforms to be applied on a sample.
            target_labels (list of strings): labels indicating deforestation
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.targets = target_labels or ['slash_and_burn', 'selective_logging', 'clear']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Read image
        img_name = self.annotations.iloc[idx]['image_name']
        img_path = os.path.join(self.img_dir, img_name + '.png')
        image = Image.open(img_path).convert('RGB')  # 3-channel (we drop NIR for simplicity)

        # Read labels (multi-label columns)
        labels = self.annotations.iloc[idx][self.targets].values.astype(float)
        # Binary label: any deforestation label -> 1, else 0
        label = 1.0 if labels.sum() > 0 else 0.0

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# 2. Data transforms and loaders
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Paths - change these to your actual paths
train_csv = 'data/train_labels.csv'
val_csv   = 'data/val_labels.csv'
train_dir = 'data/train_images'
val_dir   = 'data/val_images'

train_dataset = PlanetDataset(train_csv, train_dir, transform=train_transform)
val_dataset   = PlanetDataset(val_csv,   val_dir,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

# 3. Model definition - modify ResNet-50 for binary classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet50(pretrained=True)
# Replace final fully connected layer
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid()
)
model = model.to(device)

# 4. Loss, optimizer, scheduler
total_steps = len(train_loader)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 5. Training and validation loops
num_epochs = 20
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        preds = (outputs > 0.5).float()
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # Validate
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    avg_val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    scheduler.step(avg_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}] ' \
          f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} ' \
          f'| Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_deforestation_model.pth')

print('Training complete. Best Val Loss: {:.4f}'.format(best_val_loss))

# 6. Inference example
def predict(image_path, model, transform):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model(img_t).item()
    return prob

# Example usage:
# model.load_state_dict(torch.load('best_deforestation_model.pth'))
# prob = predict('data/val_images/abc123.png', model, val_transform)
# print(f'Deforestation probability: {prob:.2f}')
