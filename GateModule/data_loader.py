import torch
import os
from torch.utils.data import Dataset
from skimage import io
from glob import glob
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32, antialias=True), transforms.Normalize((0.5,) * 1, (0.5,) * 1)])

BATCH_SIZE = 64

class UserDataLoader(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.label = get_label(data_path_list)
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = io.imread(self.path_list[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.classes.index(self.label[idx])


def get_label(data_path_list):
    return [path.split('/')[-2] for path in data_path_list]


def trainloader(dataset, labels):
    base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/router_data/{dataset}/train'
    TRAIN_DATA_SET_PATH = glob(f'{base_path}/*/*[.png, .jpg]')

    train_loader = torch.utils.data.DataLoader(
        UserDataLoader(TRAIN_DATA_SET_PATH, labels, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return train_loader


def testloader(dataset, labels):
    base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/router_data/{dataset}/test'
    TEST_DATA_SET_PATH = glob(f'{base_path}/*/*[.png, .jpg]')

    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(TEST_DATA_SET_PATH, labels, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return test_loader
    

def candidate_dataloader(dataset, labels, n_dataset):
    total_data_subset = []
    
    for label in labels:
        data_label_path = glob(f'{os.path.expanduser("~")}/Workspace/DataSet/router_data/{dataset}/candidate/{label}/*[.png, .jpg]')
        data_label = UserDataLoader(data_label_path, labels, transform=transform)
        data_subset = Subset(data_label, list(range(n_dataset)))
        total_data_subset.append(data_subset)
    combined_dataset = ConcatDataset(total_data_subset)
    
    candidate_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    return candidate_loader


def confidence_dataloader(base_path, labels, n_dataset):
    total_data_subset = []
    
    for label in labels:
        data_label_path = glob(f'{base_path}/{label}/*[.png, .jpg]')
        data_label = UserDataLoader(data_label_path, labels, transform=transform)
        data_subset = Subset(data_label, list(range(len(data_label_path))))
        total_data_subset.append(data_subset)
    combined_dataset = ConcatDataset(total_data_subset)
    
    candidate_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    return candidate_loader


def cls_dataloader():
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # labels = ['apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    # 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    # 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin']
    
    # labels = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    
    TEST_DATA_SET_PATH = glob(f'{os.path.expanduser("~")}/Workspace/DataSet/genesis_data/CIFAR10_64/test/*/*[.png, .jpg]')

    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(TEST_DATA_SET_PATH, labels, transform=transform),
        batch_size=1,
        shuffle=True
    )

    return test_loader


def src_testloader(label):
    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # LABELS = ['apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    # 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    # 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin']
    
    # LABELS = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/genesis_data/CIFAR10_64/test/{label}'
    TEST_DATA_SET_PATH = glob(f'{base_path}/*[.png, .jpg]')

    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(TEST_DATA_SET_PATH, LABELS, transform=transform),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return test_loader
