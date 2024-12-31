import torch
import torch.nn as nn
from torchvision import models
hidden_size = 256
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC and pooling layers
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, hidden_size)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size * num_frames, c, h, w)
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(batch_size, num_frames, -1)
        return x