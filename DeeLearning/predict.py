import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from PIL import Image
import re
import torch.nn.functional as F

# Natural sorting function
def natural_sort_key(s):
    _nsre = re.compile('([0-9]+)')
    return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Custom Dataset class
class PredictDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')], key=natural_sort_key)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')  # Ensure image is in RGB format
        if self.transform:
            image = self.transform(image)
        image_id = int(self.image_files[idx].split('_')[1].split('.')[0])
        return image, image_id

class CNN(nn.Module):
    def __init__(self, num_classes=10, conv1_channels=64, conv2_channels=128, conv3_channels=256, conv4_channels=512, fc1_size=1024, fc2_size=512, dropout_p=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=conv1_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_channels)
        self.conv2 = nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_channels)
        self.conv3 = nn.Conv2d(in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(conv3_channels)
        self.conv4 = nn.Conv2d(in_channels=conv3_channels, out_channels=conv4_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(conv4_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv5 = nn.Conv2d(in_channels=conv4_channels, out_channels=conv4_channels, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(conv4_channels)
        self.conv6 = nn.Conv2d(in_channels=conv4_channels, out_channels=conv4_channels, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(conv4_channels)
        
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(conv4_channels * 2 * 2, fc1_size)
        self.bn7 = nn.BatchNorm1d(fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.bn8 = nn.BatchNorm1d(fc2_size)
        self.fc3 = nn.Linear(fc2_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(F.relu(self.bn6(self.conv6(x))))
        
        x = x.view(-1, self.conv4.out_channels * 2 * 2)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn8(self.fc2(x)))
        x = self.fc3(x)
        return x

def load_checkpoint(filename, model):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    print(f"Checkpoint loaded from {filename}")

def predict(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, image_ids in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            results.extend(zip(image_ids.cpu().numpy(), predicted.cpu().numpy()))
    return results

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CNN(
        num_classes=10,
        conv1_channels=64,
        conv2_channels=128,
        conv3_channels=256,
        conv4_channels=512,
        fc1_size=1024,
        fc2_size=512,
        dropout_p=0.5
    ).to(device)

    checkpoint_path = 'models/best_model.pth.tar'
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)

    # Load prediction data
    predict_dataset = PredictDataset(root_dir='nzmsa-2024/cifar10_images/test', transform=transform)
    predict_loader = DataLoader(predict_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Check if prediction data is loaded correctly
    print("Checking loaded prediction data...")
    for images, image_ids in predict_loader:
        print(f"Image IDs: {image_ids[:5]}")
        print(f"Image tensor shape: {images.shape}")
        break  # Only check one batch

    # Perform predictions
    results = predict(model, predict_loader, device)

    # Save results to CSV file
    results_df = pd.DataFrame(results, columns=['id', 'label'])  # Change column name from 'image_id' to 'id'
    results_df.sort_values(by='id', inplace=True)  # Sort by id
    results_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")
