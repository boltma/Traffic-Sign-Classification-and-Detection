import json
import numpy as np
from PIL import Image


def read_image(task):
    data = []
    label = []
    with open('data/Classification/Data/train.json', 'r') as fp:
        js = json.load(fp)
    v0 = 0
    for f, v in js.items():
        if v == 'po' or v == 'io':
            continue
        if task == "train":
            if v != v0:
                image = Image.open('data/Classification/Data/Train/' + v + '/' + f)
                v0 = v
            else:
                continue
        else:
            image = Image.open('data/Classification/Data/Train/' + v + '/' + f)
        imafter = image.resize((64, 64))
        imafter = imafter.convert("L")
        data.append(np.array(imafter))
        label.append(v)
    return data, label


def normalization(data):
    data = np.array(data)
    maxn = np.max(data, axis=0)
    minn = np.min(data, axis=0)
    return (data - minn) / (maxn - minn)


if __name__ == "__main__":
    train_data, train_label = read_image("train")
    test_data, test_label = read_image("test")
    train_data = np.array(train_data)
    test_data = np.array(test_data)
