import model
from dataset import my_dataset
import torch
import torch.optim as optim
from train import train
from torchvision import transforms
from test import test

def main():
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
    test_model = model.ResNet18()
    test_model = test_model.cuda()
    #test_model.load_state_dict(torch.load('paramsdnn.pkl'))
    optimizer = optim.Adam(test_model.parameters(), lr = 0.001)
    result = train(test_model, 100, optimizer, train_loader)
    test(result, val_loader)
    
if __name__ == "__main__":
    main()