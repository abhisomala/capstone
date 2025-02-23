import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class AlgaeBloomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]  # First column is the image path
        label = int(self.data.iloc[idx, 1])  # Second column is the label (0 or 1)
        
        image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)  # Match BCE loss dtype

def get_data_loaders(dataset_path, batch_size=32):
    train_csv = os.path.join(dataset_path, "labels.csv")  # Use your labels.csv
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for stability
    ])
    
    dataset = AlgaeBloomDataset(train_csv, transform=transform)

    # Split into training (80%) and test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, None, 2  # Binary classification, so 2 classes

