import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import random

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = random.randint(0, 1)  # Assigns a random label (binary classification)

        return image, torch.tensor(label, dtype=torch.long)  
def get_data_loaders(dataset_path, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")
    test_path = os.path.join(dataset_path, "test")

    train_dataset = CustomImageDataset(train_path, transform)
    val_dataset = CustomImageDataset(val_path, transform)
    test_dataset = CustomImageDataset(test_path, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = 1  # Since there are no subfolders, treating all as one class

    return train_loader, test_loader, val_loader, num_classes
