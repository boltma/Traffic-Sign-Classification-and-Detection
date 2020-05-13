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


def train(model, num_epochs, optimizer, loader, save = True, cuda = False):
    if cuda:
        loss = nn.CrossEntropyLoss().cuda()
    else:
        loss = nn.CrossEntropyLoss()
    result = []
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
        result.append(train_loss)
        if save:
            torch.save(model.state_dict(), 'params18.pkl')
    dataframe = pd.DataFrame({'epoch': num, 'result': result})
    dataframe.to_csv("result.csv")

    return model


