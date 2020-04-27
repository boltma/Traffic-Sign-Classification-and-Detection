import dataset
import time
import torch.nn as nn
import torch
import model
from torch.autograd import Variable
import torchvision
import os

def test(model, loader):
    model.eval()
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    for i, data in enumerate(loader):
        image, label = data['image'], data['label']
        image = Variable(image.cuda())
        label = Variable(label.cuda())

        output = model(image)

        pred = torch.argmax(output, 1)
        correct += (pred == label).sum().float()
        total += len(label)
    print("accuracy: {}".format(correct / total))
