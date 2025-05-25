# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:29:48 2025

@author: RaDoN
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

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
output_dir = r"C:\\Users\\razda\\OneDrive\\Desktop\\First Degree\\Introduction to artificial intelligence\\AI Project\\Pictures\\ImageDataGenerator_Model_Files"
os.makedirs(output_dir, exist_ok=True)

# Transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomHorizontalFlip(),  # Flip images
    transforms.RandomRotation(10),     # Rotate images
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

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 14 * 14, 128)  # Adjust size based on input
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# Initialize the model
model = SimpleCNN().to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Lists to store training and validation metrics
train_losses = []
val_losses = []
val_accuracies = []
f1_scores = []  # List to store F1 scores
confusion_matrices = []  # List to store confusion matrices

# Early Stopping Variables
best_val_accuracy = 0
epochs_without_improvement = 0
patience = 5  # Number of epochs to wait for improvement

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
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == labels).sum().item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(correct / len(val_dataset))

    # Calculate F1 Score
    f1 = f1_score(all_labels, all_preds)
    f1_scores.append(f1)

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    confusion_matrices.append(cm)

    # Early Stopping Check
    if val_accuracies[-1] > best_val_accuracy:
        best_val_accuracy = val_accuracies[-1]
        epochs_without_improvement = 0  # Reset counter if we have an improvement
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

    print(f"Epoch {epoch + 1}/{EPOCHS},Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}, F1 Score: {f1_scores[-1]:.2f}")
    scheduler.step()

# Ensure all lists have the same length
min_length = min(len(train_losses), len(val_losses), len(val_accuracies), len(f1_scores))

train_losses = train_losses[:min_length]
val_losses = val_losses[:min_length]
val_accuracies = val_accuracies[:min_length]
f1_scores = f1_scores[:min_length]

# Save metrics as CSV
metrics = pd.DataFrame({
    'Epoch': list(range(1, min_length + 1)),
    'Training Loss': train_losses,
    'Validation Loss': val_losses,
    'Validation Accuracy': val_accuracies,
    'F1 Score': f1_scores
})
metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
metrics.to_csv(metrics_csv_path, index=False)
print(f"Metrics saved successfully at {metrics_csv_path}")

# Combined Plot for Loss, Accuracy, and F1 Score
plt.figure(figsize=(12, 6))
plt.plot(range(1, min_length + 1), train_losses, label='Training Loss', color='blue', linestyle='--')
plt.plot(range(1, min_length + 1), val_losses, label='Validation Loss', color='orange', linestyle='-')
plt.plot(range(1, min_length + 1), val_accuracies, label='Validation Accuracy', color='green', linestyle='-')
plt.plot(range(1, min_length + 1), f1_scores, label='F1 Score', color='red', linestyle='-.')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('SimpleCNN_Parkinson_Model')
plt.legend()
plt.grid(True)

# Save the plot without showing it
plot_path = os.path.join(output_dir, "training_validation_metrics.png")
plt.savefig(plot_path)
plt.close()
print(f"Plot saved successfully at {plot_path}")

# Save the model
model_path = os.path.join(output_dir, "SimpleCNN_Parkinson_Model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at {model_path}")

# Calculate and save the final confusion matrix as an image
if len(confusion_matrices) > 0:
    final_cm = confusion_matrices[-1]  # Use the last confusion matrix from the list
    final_cm_path = os.path.join(output_dir, "final_confusion_matrix.png")

    plt.figure(figsize=(8, 8))
    sns.heatmap(final_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Parkinson's"],
                yticklabels=["Healthy", "Parkinson's"])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Final Confusion Matrix')
    plt.savefig(final_cm_path)  # Save the confusion matrix as an image
    plt.close()
    print(f"Final confusion matrix saved as an image at {final_cm_path}")
else:
    print("No confusion matrix available to save.")

# Function to log results to a file
def log_results(message):
    results_file_path = os.path.join(output_dir, "SimpleCNN_Results.txt")
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
