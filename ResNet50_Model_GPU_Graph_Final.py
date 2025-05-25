# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 19:32:15 2025

@author: RaDoN
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

# Fix for OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Use 'Agg' backend for Matplotlib to prevent display issues
matplotlib.use('Agg')

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Basic settings
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Image size
BATCH_SIZE = 32
EPOCHS = 25

# Directory paths
base_dir = r"C:\\Users\\razda\\OneDrive\\Desktop\\First Degree\\Introduction to artificial intelligence\\AI Project\\Pictures\\Model_Data"
output_dir = r"C:\\Users\\razda\\OneDrive\\Desktop\\First Degree\\Introduction to artificial intelligence\\AI Project\\Pictures\\ResNet50_Model_Files"
os.makedirs(output_dir, exist_ok=True)

# Transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load datasets
dataset = datasets.ImageFolder(base_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define the model
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# Initialize the model
model = ResNetModel().to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store training and validation metrics
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct / len(val_dataset))

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}")

# Save metrics as CSV
metrics = pd.DataFrame({
    'Epoch': list(range(1, EPOCHS + 1)),
    'Training Loss': train_losses,
    'Validation Loss': val_losses,
    'Validation Accuracy': val_accuracies
})
metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
metrics.to_csv(metrics_csv_path, index=False)
print(f"Metrics saved successfully at {metrics_csv_path}")

# Combined Plot for Loss and Accuracy
plt.figure(figsize=(12, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', color='blue', linestyle='--')
plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss', color='orange', linestyle='-')
plt.plot(range(1, EPOCHS + 1), val_accuracies, label='Validation Accuracy', color='green', linestyle='-')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('ResNet50_Parkinson_Model')
plt.legend()
plt.grid(True)

# Save the plot without showing it
plot_path = os.path.join(output_dir, "training_validation_metrics.png")
plt.savefig(plot_path)
plt.close()
print(f"Plot saved successfully at {plot_path}")

# Save the model
model_path = os.path.join(output_dir, "ResNet50_Parkinson_Model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at {model_path}")

# Function to log results to a file
def log_results(message):
    results_file_path = os.path.join(output_dir, "ResNet50_Results.txt")
    with open(results_file_path, "a") as f:
        f.write(message + "\n")

# Function for prediction on a new image
def predict_image(image_path):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img).item()

    # Decision with intermediate stages
    if output == 1.0:
        result = f"Prediction for {image_path}: Parkinson's detected with absolute confidence"
    elif output == 0.0:
        result = f"Prediction for {image_path}: Healthy with absolute confidence"
    elif output > 0.8:
        result = f"Prediction for {image_path}: Likely Parkinson's ({output * 100:.2f}%)"
    elif output < 0.2:
        result = f"Prediction for {image_path}: Likely Healthy ({(1 - output) * 100:.2f}%)"
    else:
        result = f"Prediction for {image_path}: Uncertain ({output * 100:.2f}%)"

    print(result)
    log_results(result)

# Example predictions
test_image_path = r"C:\\Users\\razda\\OneDrive\\Desktop\\First Degree\\Introduction to artificial intelligence\\AI Project\\Pictures\\Model_Data\\test_pic.png"
predict_image(test_image_path)

test_image_path = r"C:\\Users\\razda\\OneDrive\\Desktop\\First Degree\\Introduction to artificial intelligence\\AI Project\\Pictures\\Model_Data\\test_pic2.png"
predict_image(test_image_path)
