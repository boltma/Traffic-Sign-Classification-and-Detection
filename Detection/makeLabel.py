import os
import random
import json
from PIL import Image
import shutil

val_percent = 0.1
datapath = 'data/Detection'
imgsave = 'Detection/yolov3/data/images'
labelsave = 'Detection/yolov3/data/labels'
annotations = json.load(open(os.path.join(datapath, 'train_annotations.json')))['imgs']
labels = ['i2', 'i4', 'i5', 'io', 'ip', 'p11', 'p23', 'p26', 'p5', 'pl30', 'pl40', 'pl5', 'pl50', 'pl60', 'pl80', 'pn',
          'pne', 'po', 'w57']
for img_data in annotations.values():
    path = os.path.join(datapath, img_data['path'])
    img = Image.open(path)
    shutil.copy(path, imgsave)
    label_file = open(os.path.join(labelsave, str(img_data['id']) + '.txt'), 'w')
    for objects in img_data['objects']:
        bbox = objects['bbox']
        xmin = bbox['xmin']
        xmax = bbox['xmax']
        ymin = bbox['ymin']
        ymax = bbox['ymax']
        classnum = labels.index(objects['category'])
        xcenter = (xmin + xmax) / (2.0 * img.size[0])
        ycenter = (ymin + ymax) / (2.0 * img.size[1])
        width = (xmax - xmin) / img.size[0]
        height = (ymax - ymin) / img.size[1]
        label_file.write(
            str(classnum) + ' ' + str(xcenter) + ' ' + str(ycenter) + ' ' + str(width) + ' ' + str(height) + '\n')

num = len(annotations)
tv = int(num * val_percent)
name = []
for img_data in annotations.values():
    name.append(img_data['id'])
val = random.sample(name, tv)

ftrain = open('Detection/yolov3/data/train.txt', 'w')
fval = open('Detection/yolov3/data/val.txt', 'w')

for img_data in annotations.values():
    name = 'data/images/' + str(img_data['id']) + '.jpg\n'
    if img_data['id'] in val:
        fval.write(name)
    else:
        ftrain.write(name)

ftrain.close()
fval.close()
