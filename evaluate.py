import torch
from dataset import get_data_loaders
from model import get_model

# Dataset path
dataset_path = "C:/Users/somal/OneDrive/Desktop/Capstone_Model/algae_bloom_dataset"

# Load test data
_, test_loader, _, num_classes = get_data_loaders(dataset_path)

# Load model
model = get_model(num_classes)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Evaluate on test set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
