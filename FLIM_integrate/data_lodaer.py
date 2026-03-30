import os
import torch
from torch.utils.data import Dataset
from skimage import io
from glob import glob
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from conf import settings

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * settings.CHANNEL_SIZE, (0.5,) * settings.CHANNEL_SIZE)])
# transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(32, antialias=True), transforms.Normalize((0.5,) * settings.CHANNEL_SIZE, (0.5,) * settings.CHANNEL_SIZE)])

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


def node_dataloader(args, node_id):  
    base_path = f'{os.path.expanduser("~")}/workspace/DataSet/processing_data/{args.dataset}'
    
    TRAIN_DATA_SET_PATH = glob(f'{base_path}/{node_id}/train/*/*[.png, .jpg]')
    TEST_DATA_SET_PATH = glob(f'{base_path}/../../genesis_data/CIFAR10_64/test/*/*[.png, .jpg]')
    # TEST_DATA_SET_PATH = glob(f'{base_path}/../../genesis_data/CIFAR30/test/*/*[.png, .jpg]')
    # TEST_DATA_SET_PATH = glob(f'{base_path}/../../genesis_data/FMNIST/test/*/*[.png, .jpg]')
    
    train_loader = torch.utils.data.DataLoader(
        UserDataLoader(TRAIN_DATA_SET_PATH, settings.LABELS, transform=transform),
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(TEST_DATA_SET_PATH, settings.LABELS, transform=transform),
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    return train_loader, test_loader


def node_rot_dataloader(args, node_id): 
    labels = ['0', '90', '180', '270']
    
    args.dataset = args.dataset.replace('_aug', '')
    base_path = f'{os.path.expanduser("~")}/workspace/DataSet/rotation_data/{args.dataset}'
    # base_path = f'{os.path.expanduser("~")}/workspace/DataSet/rotation_data/cifar10_h_w10_l2'
    
    TRAIN_DATA_SET_PATH = glob(f'{base_path}/{node_id}/train/*/*[.png, .jpg]')
    TEST_DATA_SET_PATH = glob(f'{base_path}/{node_id}/test/*/*[.png, .jpg]')
    
    train_loader = torch.utils.data.DataLoader(
        UserDataLoader(TRAIN_DATA_SET_PATH, labels, transform=transform),
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(TEST_DATA_SET_PATH, labels, transform=transform),
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )

    return train_loader, test_loader


def src_testloader(dataset):
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # labels = ['apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    # 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    # 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin']
    
    # labels = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    base_path = f'{os.path.expanduser("~")}/workspace/DataSet/genesis_data/{dataset}/test'
    
    TEST_DATA_SET_PATH = glob(f'{base_path}/*/*[.png, .jpg]')
    
    test_loader = torch.utils.data.DataLoader(
        UserDataLoader(TEST_DATA_SET_PATH, labels, transform=transform),
        batch_size=settings.BATCH_SIZE,
        shuffle=True
    )
    
    return test_loader