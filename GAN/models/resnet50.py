"""
Mnist tutorial main model
"""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class R_50_MNIST(nn.Module):
    def __init__(self, num_classes:int=3):
        super().__init__()
        self.model = resnet50(pretrained= True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features,in_features)
        self.cls_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(in_features, in_features),
                nn.ReLU(),
                nn.Linear(in_features, num_classes))


    def forward(self, x):
        x = self.model(x)
        x = self.cls_head(x)
        return x
