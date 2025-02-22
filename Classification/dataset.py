import os
import torch
from PIL import Image
from torchvision import transforms
labels = {
    'i2':0,
    'i4':1,
    'i5':2,
    'io':3,
    'ip':4,
    'p5':5,
    'p11':6,
    'p23':7,
    'p26':8,
    'pl5':9,
    'pl30':10,
    'pl40':11,
    'pl50':12,
    'pl60':13,
    'pl80':14,
    'pn':15,
    'pne':16,
    'po':17,
    'w57':18
}
class my_dataset(torch.utils.data.Dataset):
    def __init__(self, mode, transforms = None):
        self.num_class = 19
        self.imgs = []
        self.img_class = []
        self.mode = mode
        self.transforms = transforms
        if self.mode == "train":
            self.classes = os.listdir(os.path.join("data/Classification/Data/Train"))
            for img_classes in self.classes:
                img_dir = os.listdir(os.path.join("data/Classification/Data/Train", img_classes))
                self.imgs += list(map(lambda x: img_classes + "/" + x, img_dir))
                c = [img_classes] * len(img_dir)
                self.img_class += c
        elif self.mode == "test":
            self.imgs += os.listdir(os.path.join("data/Classification/Data/Test"))
        elif self.mode == "det-test":
            self.imgs += os.listdir(os.path.join("data/Detection/crop"))
        
    def __getitem__(self, idx):
        if self.mode == "train":
            img_path = os.path.join("data/Classification/Data/Train/", self.imgs[idx])
        elif self.mode == "test":
            img_path = os.path.join("data/Classification/Data/Test/", self.imgs[idx])
        elif self.mode == "det-test":
            img_path = os.path.join("data/Detection/crop/", self.imgs[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        if self.mode == "train":
            sample = {'image': img, 'label': labels[self.img_class[idx]]}
        elif self.mode == "test" or self.mode == "det-test":
            sample = {'image': img}
        return sample
    def __len__(self):
        return len(self.imgs)



