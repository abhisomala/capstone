import torch
import torch.nn as nn
from model import AlgaeBloomClassifier  
from dataset import get_data_loaders

# Load test data
dataset_path = "C:/Users/somal/OneDrive/Desktop/Capstone_Model/algae_bloom_dataset"
_, test_loader, _, _ = get_data_loaders(dataset_path, batch_size=16)

# Load the model
model = AlgaeBloomClassifier()  
model.load_state_dict(torch.load("best_model.pth"))  #Load saved weights
model.eval()  # Set model to evaluation mode

# Define loss function
criterion = nn.BCEWithLogitsLoss()

# Evaluation loop
total_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.view(-1, 1)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        predicted = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Loss: {total_loss:.4f}, Test Accuracy: {accuracy:.2f}%")
