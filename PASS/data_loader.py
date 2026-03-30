import torch
import os
from torch.utils.data import Dataset
from skimage import io
from glob import glob
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from conf import settings
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from PIL import Image
from target import target_label_dict, filtered_label_dict


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * settings.CHANNEL_SIZE, (0.5,) * settings.CHANNEL_SIZE)])

class UserDataLoader(Dataset):
    def __init__(self, data_path_list, classes, transform=None):
        self.path_list = data_path_list
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        image_path = self.path_list[idx]
        image = Image.open(image_path).convert('RGB')
        # image = Image.open(image_path).convert('L')
        if self.transform is not None:
            image = self.transform(image)

        label_name = image_path.split('/')[-2]
        label = self.classes.index(label_name)

        return image, label, image_path

def get_label(data_path_list):
    return [path.split('/')[-2] for path in data_path_list]
    # return [os.path.dirname(path).split('/')[-1] for path in data_path_list]


def integrated_trainloader(args, node_id, n_train_dataset):
    # target_labels = target_label_dict[node_id]
    target_labels = filtered_label_dict[node_id] # filterted data load
    
    total_train_data_subset = []

    for cls_label in target_labels:
        for rot_label in settings.LABELS:
            train_data_label_path = glob(f'{os.path.expanduser("~")}/Workspace/DataSet/rotation_data/{args.dataset}/{cls_label}/train/{rot_label}/*[.png, .jpg]')
            train_data_label = UserDataLoader(train_data_label_path, settings.LABELS, transform=transform)
            train_data_subset = Subset(train_data_label, list(range(n_train_dataset)))
            total_train_data_subset.append(train_data_subset)

        combined_train_dataset = ConcatDataset(total_train_data_subset)

    train_loader = torch.utils.data.DataLoader(
        combined_train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    return train_loader


def rot_trainloader(args, task_label, n_train_dataset):
    total_train_data_subset = []

    for label in settings.LABELS:
        train_data_label_path = glob(f'{os.path.expanduser("~")}/Workspace/DataSet/rotation_data/{args.dataset}/{task_label}/train/{label}/*[.png, .jpg]')
        
        train_data_label = UserDataLoader(train_data_label_path, settings.LABELS, transform=transform)

        if n_train_dataset >= len(train_data_label):
            n_train_dataset = len(train_data_label)

        train_data_subset = Subset(train_data_label, list(range(n_train_dataset)))

        total_train_data_subset.append(train_data_subset)

    combined_train_dataset = ConcatDataset(total_train_data_subset)

    train_loader = torch.utils.data.DataLoader(
        combined_train_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    return train_loader

def rot_testloader(args, task_label, n_test_dataset):
    total_test_data_subset = []

    for label in settings.LABELS:
        test_data_label_path = glob(f'{os.path.expanduser("~")}/Workspace/DataSet/rotation_data/{args.dataset}/{task_label}/test/{label}/*[.png, .jpg]')

        test_data_label = UserDataLoader(test_data_label_path, settings.LABELS, transform=transform)

        if n_test_dataset >= len(test_data_label):
            n_test_dataset = len(test_data_label)
        
        test_data_subset = Subset(test_data_label, list(range(n_test_dataset)))
        
        total_test_data_subset.append(test_data_subset)

    combined_test_dataset = ConcatDataset(total_test_data_subset)

    test_loader = torch.utils.data.DataLoader(
        combined_test_dataset,
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    return test_loader


def testloader():
    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # LABELS = ['apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    # 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    # 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin']
    
    # LABELS = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/genesis_data/CIFAR10_64'
    
    TRAIN_DATA_SET_PATH = glob(f'{base_path}/test/*/*[.png, .jpg]')
    
    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(TRAIN_DATA_SET_PATH, LABELS, transform=transform),
        batch_size=64,
        shuffle=True
    )

    return test_loader
    

