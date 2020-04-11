from skimage.feature import hog
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import joblib
from PIL import Image
import numpy as np
import os


def read_data(data_dir):
    datas = []
    labels = []
    for label in os.listdir(data_dir):
        print(label)
        class_path = os.path.join(data_dir, label)
        for img_name in os.listdir(class_path):
            img = Image.open(os.path.join(class_path, img_name))
            out = img.resize((64, 64), Image.ANTIALIAS)
            fd = hog(out, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(16, 16), block_norm='L2',
                     feature_vector=True, multichannel=True)
            datas.append(fd)
            labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)
    np.savez('hog', datas=datas, labels=labels)
    return datas, labels


data = np.load('hog.npz')
datas = data['datas']
labels = data['labels']
# datas, labels = read_data('data/Classification/Data/Train')

svr = svm.SVC()
parameters = {'kernel': ('linear', 'rbf'), 'C': [0.1, 1, 10, 100, 1000], 'gamma': [10, 1, 0.1, 0.01, 0.001]}
classifier = GridSearchCV(svr, parameters, scoring='f1_weighted', n_jobs=12, cv=5, verbose=1)

X_train, X_test, y_train, y_test = train_test_split(datas, labels, test_size=0.2)

classifier.fit(X_train, y_train)
print('The parameters of the best model are: ')
print(classifier.best_params_)
joblib.dump(classifier, 'svm.m')
predicted = classifier.predict(X_test)
print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, predicted)))
