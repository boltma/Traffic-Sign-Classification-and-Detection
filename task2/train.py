import dataset
import model
import torch.nn as nn
import os
import time
from torch.autograd import Variable
import torchvision
import torch

def train(model, num_epochs, optimizer, loader, save = True):
    loss = nn.CrossEntropyLoss().cuda()

    for epoch in range(num_epochs):
        train_loss = 0.0
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        for data in loader:
            model.train(True)
            image, label = data['image'], data['label']
            image = Variable(image.cuda())
            label = Variable(label.cuda())

            optimizer.zero_grad()

            output = loss(model(image), label)
            train_loss += output.item()
            output.backward()
            optimizer.step()
        print("loss: {}".format(train_loss))
        if save:
            torch.save(model.state_dict(), 'params.pkl')
        
    return model


