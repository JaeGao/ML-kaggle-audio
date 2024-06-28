import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import librosa
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set PyTorch to use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the data augmentation transforms
transforms = A.Compose([
    A.Resize(height=128, width=128),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])

# Custom dataset class
class AudioDataset(Dataset):
    def __init__(self, features, labels, transforms=None):
        self.features = features
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        if self.transforms:
            feature = self.transforms(image=feature)['image']

        return feature, label

# Model architecture
class VoiceIdentificationModel(nn.Module):
    def __init__(self):
        super(VoiceIdentificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Prepare the data
train_data, val_data, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

train_dataset = AudioDataset(train_data, train_labels, transforms=transforms)
val_dataset = AudioDataset(val_data, val_labels, transforms=transforms)
test_dataset = AudioDataset(test_features, np.zeros(len(test_features)), transforms=transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Create the model
model = VoiceIdentificationModel().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_correct = 0
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / len(val_dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Update the learning rate
    scheduler.step()

# Make predictions on the test set
model.eval()
predicted_labels = []
with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        predicted_labels.extend(predicted.cpu().tolist())

submission = pd.DataFrame({'id': range(len(predicted_labels)), 'category': predicted_labels})
submission.to_csv('submission.csv', index=False)