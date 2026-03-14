import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device (CPU only)
device = torch.device("cpu")

# 1️⃣ Transforms (Image Preprocessing)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(            # Normalize pixel values
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 2️⃣ Load Dataset
train_dataset = datasets.ImageFolder("data/train", transform=transform)
val_dataset = datasets.ImageFolder("data/val", transform=transform)

# 3️⃣ DataLoader (Batches)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Classes:", train_dataset.classes)
print("Number of training images:", len(train_dataset))

import torch.nn as nn
from torchvision import models

# 1️⃣ Load Pretrained EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)

# 2️⃣ Replace Final Layer
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

# 3️⃣ Move Model to Device
model = model.to(device)

print(model)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epochs = 3

for epoch in range(epochs):
    model.train()  # set model to training mode
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")
        # 🔹 Validation phase
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

torch.save(model.state_dict(), "pneumonia_model.pth")
print("Model saved successfully!")

