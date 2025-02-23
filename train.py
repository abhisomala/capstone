import torch 
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import get_model

# Dataset path
dataset_path = "C:/Users/somal/OneDrive/Desktop/Capstone_Model/algae_bloom_dataset"

# Load data
train_loader, test_loader, val_loader, num_classes = get_data_loaders(dataset_path)

# Get model
model = get_model(num_classes)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        labels = labels.view(-1, 1).float()  # Ensure labels are float and correctly shaped
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("Model training complete. Saved as 'model.pth'.")
