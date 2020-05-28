import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import os
import json
from config import PRE_TRAINING_DATA_PATH, VALIDATION_DATA_PATH
from torchvision import transforms


class Dataset(data.Dataset):
    def __init__(self, mode):
        self.image_path = []
        self.image_label_str = []
        self.image_label = []
        if mode == 'train' or mode == 'Train':
            PATH = PRE_TRAINING_DATA_PATH
        else:
            PATH = VALIDATION_DATA_PATH
        mdict = {}
        for root, dirs, files in os.walk(PATH):
            if files == []:
                continue
            for i in files:
                self.image_path.append(root + '/' + i)
                self.image_label_str.append(i.split('_')[0])

        self.imgs = map(load_img, self.image_path, range(len(self.image_path)))
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.imgs = list(self.imgs)
        temp = np.unique(self.image_label_str)
        for i, value in enumerate(temp):
            mdict[value] = i + 1000
        for i in self.image_label_str:
            self.image_label.append(mdict[i])

    def __getitem__(self, idx):
        x = self.imgs[idx]
        return x, self.image_label[idx]

    def __len__(self):
        return len(self.image_label)


def load_img(path, idx):
    x = Image.open(path)
    x = x.convert('L')
    x = x.resize((28, 28))
    shape = 1, x.size[0], x.size[1]
    x = np.array(x, np.float32, copy=False)
    x = 1.0 - torch.from_numpy(x)
    x = x.transpose(0, 1).contiguous().view(shape)
    return x
