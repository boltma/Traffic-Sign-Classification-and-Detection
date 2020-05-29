from protonet_dataset import load_img
from protonet import ProtoNet
import numpy as np
import torch
import os
import json
from config import DATA_FEW_SHOT, EXPERIMENT_PATH


def init_protonet():
    '''
    Initialize the ProtoNet
    '''
    model = ProtoNet()
    return model


def load_from_file():
    root = DATA_FEW_SHOT + os.sep + 'Test'
    img = []
    img_names = []
    for roots, dirs, files in os.walk(root):
        for img_name in files:
            img_names.append(img_name)
            img.append(load_img(roots + os.sep + img_name, None).unsqueeze(dim=0))
    return img, img_names


def load_from_json(mode):

    with open(DATA_FEW_SHOT + os.sep + mode + '.json') as fp:
        json_data = json.load(fp)
    paths = []
    label = []
    img = []
    for key, value in json_data.items():
        if mode == 'train':
            paths.append(DATA_FEW_SHOT + os.sep + 'Train' + os.sep + value +
                         os.sep + key)
            img.append(
                load_img(
                    DATA_FEW_SHOT + os.sep + 'Train' + os.sep + value +
                    os.sep + key, None).unsqueeze(dim=0))
        else:
            paths.append(DATA_FEW_SHOT + os.sep + 'Test' + os.sep + key)
            img.append(
                load_img(DATA_FEW_SHOT + os.sep + 'Test' + os.sep + key,
                         None).unsqueeze(dim=0))
        label.append(value)
    # img = map(load_img, paths, range(len(paths)))
    return img, label


def predict(proto, support_proto, support_label):
    dist = []
    for target in support_proto:
        dist.append(
            euclidean_dist(target.detach().numpy(),
                           proto.detach().numpy()))
    dist = np.array(dist)
    return support_label[np.where(dist == dist.min())[0][0]]


def euclidean_dist(x, y):
    return np.power(x - y, 2).sum()


def main():
    model = init_protonet()
    model.load_state_dict(
        torch.load(EXPERIMENT_PATH + os.sep + 'best_model.pth'))
    model.eval()
    support_img, support_label = load_from_json('train')
    support_proto = []
    for img in support_img:
        proto = model(img)
        support_proto.append(proto.squeeze(dim=0))
    query_img, query_img_names = load_from_file()
    query_proto = []
    for img in query_img:
        proto = model(img)
        query_proto.append(proto.squeeze(dim=0))
    query_label = []
    for proto in query_proto:
        query_label.append(predict(proto, support_proto, support_label))
    query_dict = {}
    for i in range(len(query_label)):
        query_dict[query_img_names[i]] = query_label[i]
    with open(EXPERIMENT_PATH + os.sep + "pred3.json", "w") as f:
        json.dump(query_dict, f)


if __name__ == '__main__':
    main()
