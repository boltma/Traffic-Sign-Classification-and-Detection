import numpy as np


class KNN:
    def __init__(self, neighbors, data, label, method):
        self.method = method
        self.neighbors = neighbors
        self.data = data
        self.label = label
        self.num = len(label)

    def judge(self, test):
        dist_arr = {}
        for i in range(len(self.label)):
            dist_arr[self.label[i]] = self.dist(self.data[i], test)
            # self.dist_arr.append(self.dist(x, test))
        dist_arr = sorted(dist_arr.items(), key=lambda kv: (kv[1], kv[0]))
        return dist_arr[self.neighbors - 1][0]

    def dist(self, x, y):
        if self.method == "eu":
            temp = np.sum((x - y) * (x - y))
            if temp < 0.1:
                return temp * 1000
            else:
                return temp
        if self.method == "m":
            return np.sum(np.abs(x - y))
