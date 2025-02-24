# Abhijit Somala
#This file serves to train the ML algorithm using the images saved from the data_collection.py
#Final results: Epoch 20/20, Loss: 2.8033, Accuracy: 99.89% (extended results below)

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
"""
Epoch 1/20, Loss: 65.0220, Accuracy: 95.16%
Epoch 2/20, Loss: 35.9403, Accuracy: 97.42%
Epoch 3/20, Loss: 28.7328, Accuracy: 97.79%
Epoch 4/20, Loss: 25.8728, Accuracy: 97.85%
Epoch 5/20, Loss: 21.8187, Accuracy: 98.54%
Epoch 6/20, Loss: 20.8269, Accuracy: 98.53%
Epoch 7/20, Loss: 20.7290, Accuracy: 98.61%
Epoch 8/20, Loss: 18.2786, Accuracy: 98.78%
Epoch 9/20, Loss: 15.3641, Accuracy: 99.05%
Epoch 10/20, Loss: 13.3840, Accuracy: 99.26%
Epoch 11/20, Loss: 11.8104, Accuracy: 99.35%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 14/20, Loss: 8.0569, Accuracy: 99.61%
Epoch 15/20, Loss: 7.9388, Accuracy: 99.56%
Epoch 16/20, Loss: 5.5727, Accuracy: 99.80%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 14/20, Loss: 8.0569, Accuracy: 99.61%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 14/20, Loss: 8.0569, Accuracy: 99.61%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 14/20, Loss: 8.0569, Accuracy: 99.61%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 12/20, Loss: 10.0832, Accuracy: 99.50%
Epoch 13/20, Loss: 8.6414, Accuracy: 99.59%
Epoch 14/20, Loss: 8.0569, Accuracy: 99.61%
Epoch 15/20, Loss: 7.9388, Accuracy: 99.56%
Epoch 16/20, Loss: 5.5727, Accuracy: 99.80%
Epoch 17/20, Loss: 4.1028, Accuracy: 99.86%
Epoch 18/20, Loss: 4.0271, Accuracy: 99.85%
Epoch 19/20, Loss: 5.4363, Accuracy: 99.61%
Epoch 20/20, Loss: 2.8033, Accuracy: 99.89%
"""