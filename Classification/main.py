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
    parser.add_argument('--cuda', default=False)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--model', default='ResNet18')
    return parser.parse_args()
def main():
    args = parse_args()
    save = True
    use_gpu = args.cuda == str(True)

    train_tfs = transforms.Compose([
        transforms.Resize(48),
        transforms.RandomSizedCrop(48),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ds = my_dataset("train", train_tfs)
    dataset_size = ds.__len__()
    print(dataset_size)
    train_ds, val_ds = torch.utils.data.random_split(ds, [13000, 1463])
    train_loader = torch.utils.data.DataLoader(train_ds, 32, False, num_workers = 8)
    val_loader = torch.utils.data.DataLoader(val_ds, 32, False, num_workers = 8)
    print('train: ', len(train_ds))
    print('validation:', len(val_ds))
    print(type(ds), type(train_ds))
    if args.model == 'ResNet18':
        test_model = model.ResNet18()
    if args.model == 'ResNet50':
        test_model = model.ResNet50()
    if use_gpu:
        test_model = test_model.cuda()
    #test_model.load_state_dict(torch.load('paramsdnn.pkl'))
    optimizer = optim.Adam(test_model.parameters(), lr = 0.001)
    print(use_gpu)
    result = train(test_model, args.epoch, optimizer, train_loader, save, use_gpu)
    test(result, val_loader)
    
if __name__ == "__main__":
    main()