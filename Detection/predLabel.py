import os
import json
import subprocess
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from Classification import model
from Classification.dataset import my_dataset

subprocess.call(['python', 'detect.py', '--cfg', 'cfg/yolov3.cfg',
                 '--names', 'data/traffic.names', '--weights', 'weights/best.pt', '--source',
                 '../../data/Detection/test', '--save-txt'], cwd='Detection/yolov3')

datapath = 'data/Detection/test'
predpath = 'Detection/yolov3/output'
croppath = 'data/Detection/crop'
if not os.path.exists(croppath):
    os.makedirs(croppath)
labels = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn',
          'pne', 'po', 'w57']

annotations = {}
print('Initializing model.')
test_model = model.Inception()
test_model = test_model.cuda()
tfs = transforms.Compose([
    transforms.Resize(299),
    transforms.RandomSizedCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
print('Initialization finished')
test_model.load_state_dict(torch.load('paramsInception.pkl'))
for img_name in os.listdir(datapath):
    img_id = os.path.splitext(img_name)[0]
    img = Image.open(os.path.join(datapath, img_name)).convert('RGB')
    print(img_id)
    path = os.path.join(datapath, img_name)
    label_path = os.path.join(predpath, img_id + '.jpg.txt')
    label_object = []
    if os.path.exists(label_path):
        label_file = open(label_path, 'r')
        lines = label_file.readlines()
        cnt = 0
        for line in lines:
            label = line.split()
            bbox = {'xmax': int(label[2]), 'xmin': int(label[0]), 'ymax': int(label[3]), 'ymin': int(label[1])}
            crop = img.crop((bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']))
            crop.save(os.path.join(croppath, img_id + '_' + str(cnt) + '.jpg'))
            label_object.append({'bbox': bbox, 'category': '', 'score': float(label[5])})
            cnt = cnt + 1
    annotations[img_id] = {'objects': label_object}

ds = my_dataset("det-test", tfs)
loader = torch.utils.data.DataLoader(ds, 16, False)
preds = {}
num = 0
for i, data in enumerate(loader):
    image = data['image']
    image = Variable(image.cuda())
    output = test_model(image)
    pred = torch.argmax(output, 1)
    for j in pred:
        crop_name = ds.imgs[num]
        img_id = crop_name.split('_')[0]
        cnt = int(crop_name.split('_')[1].split('.')[0])
        annotations[img_id]['objects'][cnt]['category'] = labels[j.int()]
        num = num + 1

test_json = json.dumps({'imgs': annotations})
file = open('pred_annotations.json', 'w')
file.write(test_json)
file.close()
