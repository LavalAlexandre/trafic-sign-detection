import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.labels import LABELS, label_from_filename

label_to_idx = {label: idx for idx, label in enumerate(LABELS)}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TrafficSignDataset(Dataset):
    def __init__(self, data_dir, transform=transform):
        self.data_dir = data_dir
        self.transform = transform
        self.images = [
            (os.path.join(data_dir, img_name), label_to_idx[label_from_filename(img_name)])
            for img_name in sorted(os.listdir(data_dir))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label_idx = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label_idx


class CustomResNet18(nn.Module):
    """ResNet18-shaped network (same stages and widths), but without the residual connections."""

    def __init__(self, num_classes=len(LABELS)):
        super(CustomResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class Sign_Classifier():
    def __init__(self, device=None, seed=42):
        torch.manual_seed(seed)
        self.device = device or default_device()
        self.net = CustomResNet18(num_classes=len(LABELS)).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)

    def fit(self, train_dataset, val_dataset, n_epoch=25, batch_size=32, save_path='traffic_sign_resnet18.pth'):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        for epoch in range(n_epoch):
            self.net.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(images), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * labels.size(0)

            self.net.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    predicted = self.net(images).argmax(dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch + 1}/{n_epoch}], Loss: {running_loss / len(train_dataset):.4f}, '
                  f'Validation accuracy: {100 * correct / total:.2f}%')

        if save_path is not None:
            torch.save(self.net.state_dict(), save_path)
            print(f'Model saved at: {os.path.abspath(save_path)}')

        return self

    def predict_proba(self, roi):
        """roi: BGR image crop, as returned by cv2.imread."""
        self.net.eval()
        roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        roi_tensor = transform(roi).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.net(roi_tensor), dim=1).cpu().numpy()[0]

    def predict(self, roi):
        """Common classifier interface used by src.detection: returns (label, confidence)."""
        probabilities = self.predict_proba(roi)
        best = int(np.argmax(probabilities))
        return LABELS[best], float(probabilities[best])

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()
        return self
