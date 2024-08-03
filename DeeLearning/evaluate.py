import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import os
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch.nn.functional as F

# Data augmentation and preprocessing
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
dataset = CIFAR10Dataset(csv_file='nzmsa-2024/train.csv', root_dir='nzmsa-2024/cifar10_images/train', transform=transform)

# Split dataset into training, validation, and test sets
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(101))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

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

def load_checkpoint(filename, model, optimizer=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filename}, starting from epoch {epoch}")
    return epoch

def evaluate_model(model, data_loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def plot_roc_curve(labels, preds, num_classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    labels = np.array(labels)
    preds = np.array(preds)
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(labels == i, preds == i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 7))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

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

    # Evaluate the best model on the training set
    print("Evaluating the best model on the training set")
    train_labels, train_preds = evaluate_model(model, train_loader, device, num_classes=10)
    print("Training Accuracy: ", accuracy_score(train_labels, train_preds))
    print("Classification Report for Training Set:\n", classification_report(train_labels, train_preds))

    # Evaluate the best model on the test set
    print("Evaluating the best model on the test set")
    test_labels, test_preds = evaluate_model(model, test_loader, device, num_classes=10)
    print("Test Accuracy: ", accuracy_score(test_labels, test_preds))
    print("Classification Report for Test Set:\n", classification_report(test_labels, test_preds))

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, classes=[f'Class {i}' for i in range(10)], title='Confusion Matrix for Test Set')

    # ROC curve
    plot_roc_curve(test_labels, test_preds, num_classes=10)
