import torch
import torch.optim as optim
import torch.nn as nn
from model import AlgaeBloomClassifier
from dataset import get_data_loaders

# Load data
dataset_path = "C:/Users/somal/OneDrive/Desktop/Capstone_Model/algae_bloom_dataset"
train_loader, test_loader, _, _ = get_data_loaders(dataset_path, batch_size=16)

# Model, loss function, optimizer
model = AlgaeBloomClassifier()
criterion = nn.BCEWithLogitsLoss()  #Loss function
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Learning rate lowered to decrease loss

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        labels = labels.view(-1, 1)  # Checks if labels match output shape
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (torch.sigmoid(outputs) > 0.5).float()  #Convert logits to binary labels
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

    #Save model after each epoch
    
torch.save(model.state_dict(), "best_model.pth")
print("Model saved.")