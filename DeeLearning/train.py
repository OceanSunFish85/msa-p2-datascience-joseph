import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Data augmentation and preprocessing
print("Data augmentation and preprocessing")
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomCrop(32, padding=4),  # Random crop with padding
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize
])

# Custom Dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"image_{self.data_frame.iloc[idx, 0]}.png")
        image = Image.open(img_name).convert('RGB')  # Ensure image is in RGB format
        label = int(self.data_frame.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Load training data
print("Loading training data")
dataset = CIFAR10Dataset(csv_file='nzmsa-2024/train.csv', root_dir='nzmsa-2024/cifar10_images/train', transform=transform)

# Split dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(101))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

print("Data loaded and split into train, validation, and test sets")

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
        
        # Additional convolutional layers and batch normalization layers
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
    

model = CNN(
    num_classes=10,
    conv1_channels=64,
    conv2_channels=128,
    conv3_channels=256,
    conv4_channels=512,
    fc1_size=1024,
    fc2_size=512,
    dropout_p=0.5
).to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

def save_checkpoint(state, filename='models/checkpoint.pth.tar'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
    torch.save(state, filename)
    print(f"Checkpoint saved at {filename}")

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filename}, starting from epoch {epoch}")
    return epoch

# Define training and validation process
def train_and_validate(net, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, start_epoch=0, early_stopping_patience=5):
    scaler = GradScaler()
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    epochs_no_improve = 0

    for epoch in range(start_epoch, num_epochs):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:
                print(f'[Epoch {epoch + 1}, Iter {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        train_accuracy = 100 * correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # Validate the model
        net.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                with autocast():
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        scheduler.step(val_loss / len(val_loader))
        print(f'[Epoch {epoch + 1}] validation loss: {val_loss / len(val_loader):.3f}, accuracy: {val_accuracy:.2f}%')

        # Save checkpoint
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, filename='models/best_model.pth.tar')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping")
            break

    print('Finished Training')



if __name__ == '__main__':
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint if exists
    start_epoch = 0
    checkpoint_path = 'models/best_model.pth.tar'
    if os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(checkpoint_path, model, optimizer)

    # Train and validate the model
    print("Starting training and validation")
    train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=30, device=device, start_epoch=start_epoch, early_stopping_patience=5)
