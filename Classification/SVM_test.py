from skimage.feature import hog
import joblib
from PIL import Image
import numpy as np
import os
import json


def read_data(data_dir):
    datas = []
    names = []
    for img_name in os.listdir(data_dir):
        img = Image.open(os.path.join(data_dir, img_name))
        out = img.resize((64, 64), Image.ANTIALIAS)
        fd = hog(out, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(16, 16), block_norm='L2',
                 feature_vector=True, multichannel=True)
        datas.append(fd)
        names.append(img_name)
        print(img_name)

    datas = np.array(datas)
    names = np.array(names)
    return datas, names


datas, names = read_data('data/Classification/Data/Test')
print('Finish reading test images.')

classifier = joblib.load('../svm.m')
predicted = classifier.predict(datas)
test_labels = dict(zip(names, predicted))
test_json = json.dumps(test_labels)

file = open('pred1.json', 'w')
file.write(test_json)
file.close()
