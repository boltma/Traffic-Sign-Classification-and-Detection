import dataset
import model
import torch.nn as nn
import os
import time
from torch.autograd import Variable
import torchvision
import torch
import matplotlib.pyplot as plt
import pandas as pd
from test import test

def train(model, num_epochs, optimizer, loader, val_loader, modelname, save = True, cuda = False):
    if cuda:
        loss = nn.CrossEntropyLoss().cuda()
    else:
        loss = nn.CrossEntropyLoss()
    results = []
    accuracy = []
    num = [i for i in range(num_epochs)]
    f = open("classification.log", "w")
    for epoch in range(num_epochs):
        train_loss = 0.0
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        for data in loader:
            model.train(True)
            image, label = data['image'], data['label']
            if cuda:
                image = Variable(image.cuda())
                label = Variable(label.cuda())

            optimizer.zero_grad()

            output = loss(model(image), label)
            train_loss += output.item()
            output.backward()
            optimizer.step()
        print("loss: {}".format(train_loss))
        print("{} {}".format(epoch, train_loss), file = f)
        results.append(train_loss)
        accuracy.append(test(model, val_loader, cuda))
        if save:
            torch.save(model.state_dict(), 'params'+modelname+'.pkl')
    dataframe = pd.DataFrame({'epoch': num, 'result': results})
    dataframe.to_csv("result.csv")
    dataframe = pd.DataFrame({'epoch': num, 'accuracy': accuracy})
    dataframe.to_csv("accuracy.csv")
    
    return model


