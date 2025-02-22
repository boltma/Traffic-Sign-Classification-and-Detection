import model
from dataset import my_dataset
import torch
import torch.optim as optim
from train import train
from torchvision import transforms
from test import test
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--cuda', default='True')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--model', default='ResNet18', help = 'You can choose: ResNet18, ResNet50, Inception, DenseNet')
    parser.add_argument('--lr', type = float, default=0.001)
    parser.add_argument('--BS', type = int, default=16, help = 'BatchSize')
    parser.add_argument('--load', default='False', help='if you load an existing model or not')
    return parser.parse_args()
def main():
    args = parse_args()
    save = True
    use_gpu = args.cuda == 'True'
    load = args.load == 'True'
    train_tfs = transforms.Compose([
        transforms.Resize(299),
        transforms.RandomSizedCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = my_dataset("train", train_tfs)
    dataset_size = ds.__len__()
    print(dataset_size)
    train_ds, val_ds = torch.utils.data.random_split(ds, [13000, 1463])
    train_loader = torch.utils.data.DataLoader(train_ds, args.BS, False, num_workers = 8)
    val_loader = torch.utils.data.DataLoader(val_ds, args.BS, False, num_workers = 8)
    print('train: ', len(train_ds))
    print('validation:', len(val_ds))
    print(type(ds), type(train_ds))
    if args.model == 'ResNet18':
        test_model = model.ResNet18()
    if args.model == 'ResNet50':
        test_model = model.ResNet50()
    if args.model == 'Inception':
        test_model = model.Inception()
    if args.model == 'DenseNet':
        test_model = model.DenseNet()
    if use_gpu:
        test_model = test_model.cuda()
    if load:
        test_model.load_state_dict(torch.load('params' + args.model + '.pkl'))
    optimizer = optim.Adam(test_model.parameters(), lr = args.lr)
    print(use_gpu)
    result = train(test_model, args.epoch, optimizer, train_loader, val_loader, args.model, save, use_gpu)
    test(result, val_loader, use_gpu)
    
if __name__ == "__main__":
    main()