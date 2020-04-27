import torch
import torch.nn as nn
import torchvision as tv

class Model(nn.Module):
    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name
    def forward(self, images):
        x = images
        x = self.model(x)
        return x

class ResNet18(Model):
    def __init__(self):
        super().__init__("ResNet18")
        self.model = tv.models.resnet18(pretrained = True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 19)
    
        