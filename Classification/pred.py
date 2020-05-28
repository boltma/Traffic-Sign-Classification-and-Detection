import model
from dataset import my_dataset
import torch
import torch.optim as optim
from train import train
from torchvision import transforms
from test import test
from torch.autograd import Variable
import json

labels = [
    'i2',
    'i4',
    'i5',
    'io',
    'ip',
    'p5',
    'p11',
    'p23',
    'p26',
    'pl5',
    'pl30',
    'pl40',
    'pl50',
    'pl60',
    'pl80',
    'pn',
    'pne',
    'po',
    'w57'
]
name = open('pred.json', 'w')
train_tfs = transforms.Compose([
    transforms.Resize(299),
    transforms.RandomSizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
ds = my_dataset("test", train_tfs)
dataset_size = ds.__len__()
print(dataset_size)
loader = torch.utils.data.DataLoader(ds, 16, False)

test_model = model.Inception()
test_model = test_model.cuda()
test_model.load_state_dict(torch.load('paramsInception.pkl'))

preds = {}
num = 0
for i, data in enumerate(loader):

    image = data['image']
    print(image.size())

    image = Variable(image.cuda())
    # label = Variable(label.cuda())

    output = test_model(image)

    pred = torch.argmax(output, 1)
    # if i < 10:
    #    print(pred)
    for j in pred:
        if i < 10:
            print(j)
        preds[ds.imgs[num]] = labels[j.int()]
        num = num + 1
print(json.dumps(preds))
print(num)
name.write(json.dumps(preds))
name.close()
