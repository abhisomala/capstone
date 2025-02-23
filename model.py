import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):  # Change from 1 to 2
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 112 * 112, num_classes)  # Change output to 2

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x  # No softmax, since CrossEntropyLoss handles it

def get_model(num_classes):
    model = SimpleCNN(num_classes)
    return model
