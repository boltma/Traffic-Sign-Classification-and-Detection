import numpy as np
from util import read_image, normalization
from KNN import KNN
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
train_raw_data, train_label = read_image("train")
test_raw_data, test_label = read_image("test")
train_pixel = []
test_pixel = []
for raw in train_raw_data:
    train_pixel.append(raw.flatten())
for raw in test_raw_data:
    test_pixel.append(raw.flatten())
train_hog = []
test_hog = []
for raw in train_raw_data:
    train_hog.append(hog(raw, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(8, 8)))
for raw in test_raw_data:
    test_hog.append(hog(raw, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(8, 8)))

train_pixel = normalization(train_pixel)
test_pixel = normalization(test_pixel)
model = KNN(neighbors=7, data=train_hog, label=train_label, method="eu")
pridict = []
for i in range(len(test_label)):
    pridict.append(model.judge(test_hog[i]))

'''
model = KNeighborsClassifier(n_neighbors=2)
model.fit(train_pixel, train_label)
acc = model.score(test_pixel, test_label)
print(acc)
'''
print(pridict)
print(test_label)
correct = 0
# print(pridict == test_label)
# print(np.sum(np.array(pridict) == np.array(test_label)) / len(pridict))
for i in range(len(pridict)):
    if pridict[i] == test_label[i]:
        correct += 1
print(correct)
