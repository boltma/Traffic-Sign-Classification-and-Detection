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
    
class ResNet50(Model):
    def __init__(self):
        super().__init__("ResNet50")
        self.model = tv.models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 19)

class DenseNet(Model):
    def __init__(self):
        super().__init__("DenseNet")
        self.model = tv.models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, 19)

class Inception(Model):
    def __init__(self):
        super().__init__("inception")
        self.model = tv.models.inception_v3(pretrained = True)
        
        self.model.aux_logits = False
        num_ftrs = self.model.fc.in_features
        #num_Auxftrs = self.model.AuxLogits.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 19)
        #self.model.AuxLogits.fc = nn.Linear(num_Auxftrs, 19)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        return x