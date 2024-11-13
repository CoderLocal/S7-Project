# app/model.py

import torch
import torch.nn as nn

# Example image model (CNN)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 64 * 64, 512)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = x.view(-1, 16 * 64 * 64)
        x = self.fc1(x)
        return x

# Example tabular model (MLP)
class TabularMLP(nn.Module):
    def __init__(self):
        super(TabularMLP, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # Example feature size of 4
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Multimodal Fusion Model (combining image + tabular features)
class MultimodalFusionNet(nn.Module):
    def __init__(self, image_model, tabular_model):
        super(MultimodalFusionNet, self).__init__()
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.fusion_layer = nn.Linear(512 + 1, 256)  # Fusion layer
        self.fc = nn.Linear(256, 2)  # Binary classification (e.g., cancer vs. no-cancer)

    def forward(self, image, tabular):
        # Extract features
        image_features = self.image_model(image)
        tabular_features = self.tabular_model(tabular)

        # Concatenate image and tabular features
        combined_features = torch.cat((image_features, tabular_features), dim=1)

        # Fusion layer
        fused = self.fusion_layer(combined_features)
        output = self.fc(fused)
        return output

