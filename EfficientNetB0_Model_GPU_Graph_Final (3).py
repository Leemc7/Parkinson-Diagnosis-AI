# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 17:26:31 2025

@author: RaDoN
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
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
output_dir = r"C:\\Users\\razda\\OneDrive\\Desktop\\First Degree\\Introduction to artificial intelligence\\AI Project\\Pictures\\EfficientNetB0_Model_Files"
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
class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model._fc = nn.Sequential(
            nn.Linear(self.base_model._fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

# Initialize the model
model = EfficientNetModel().to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store training and validation metrics
train_losses = []
val_losses = []
val_accuracies = []
f1_scores = []  # List to store F1 scores
confusion_matrices = []  # List to store confusion matrices

# Early Stopping Variables
best_val_accuracy = 0
epochs_without_improvement = 0
patience = 7  # Number of epochs to wait for improvement

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

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}, F1 Score: {f1_scores[-1]:.2f}")

# Make sure all lists are of the same length
max_epochs = min(EPOCHS, len(train_losses), len(val_losses), len(val_accuracies), len(f1_scores))

# Save metrics as CSV
metrics = pd.DataFrame({
    'Epoch': list(range(1, max_epochs + 1)),
    'Training Loss': train_losses[:max_epochs],
    'Validation Loss': val_losses[:max_epochs],
    'Validation Accuracy': val_accuracies[:max_epochs],
    'F1 Score': f1_scores[:max_epochs]
})
metrics_csv_path = os.path.join(output_dir, "training_metrics.csv")
metrics.to_csv(metrics_csv_path, index=False)
print(f"Metrics saved successfully at {metrics_csv_path}")

# Combined Plot for Loss, Accuracy, and F1 Score
plt.figure(figsize=(12, 6))
plt.plot(range(1, max_epochs + 1), train_losses[:max_epochs], label='Training Loss', color='blue', linestyle='--')
plt.plot(range(1, max_epochs + 1), val_losses[:max_epochs], label='Validation Loss', color='orange', linestyle='-')
plt.plot(range(1, max_epochs + 1), val_accuracies[:max_epochs], label='Validation Accuracy', color='green', linestyle='-')
plt.plot(range(1, max_epochs + 1), f1_scores[:max_epochs], label='F1 Score', color='red', linestyle='-.')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('EfficientNetB0_Parkinson_Model')
plt.legend()
plt.grid(True)

# Save the plot without showing it
plot_path = os.path.join(output_dir, "training_validation_metrics.png")
plt.savefig(plot_path)
plt.close()
print(f"Plot saved successfully at {plot_path}")

# Save the model
model_path = os.path.join(output_dir, "EfficientNetB0_Parkinson_Model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at {model_path}")

# Assign the final confusion matrix
if len(confusion_matrices) > 0:
    final_cm = confusion_matrices[-1]  # Use the last confusion matrix from the list

    # Save the final confusion matrix as an image
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
    results_file_path = os.path.join(output_dir, "EfficientNetB0_Results.txt")
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

