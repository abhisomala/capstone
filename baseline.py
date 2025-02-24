# Abhijit Somala
# This file serves as a baseline model that predicts algae blooms from the downlaoded dataset using algorithms rather than a machine based based code structure.
# basline model accuracy 68.7

from dataset import get_data_loaders
import numpy as np

# Load test data
dataset_path = "C:/Users/somal/OneDrive/Desktop/Capstone_Model/algae_bloom_dataset"
_, test_loader, _, _ = get_data_loaders(dataset_path, batch_size=16)

# Define baseline threshold (adjust based on dataset)
THRESHOLD = 0.5  # Example threshold, tweak based on domain knowledge

# Evaluation loop
correct = 0
total = 0

for images, labels in test_loader:
    labels = labels.view(-1, 1).numpy()  # Convert labels to NumPy for comparison
    
    # Simple rule: If the pixel intensity > threshold, classify as bloom (1), otherwise no bloom (0)
    predicted = (np.mean(images.numpy(), axis=(1, 2, 3)) > THRESHOLD).astype(float)

    correct += (predicted == labels.flatten()).sum()
    total += labels.shape[0]  

accuracy = 100 * correct / total
print(f"Baseline Model Accuracy: {accuracy:.2f}%")


