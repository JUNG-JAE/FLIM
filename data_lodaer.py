import os
import torch
from torch.utils.data import Dataset
from skimage import io
from glob import glob
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from conf import settings

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * settings.CHANNEL_SIZE, (0.5,) * settings.CHANNEL_SIZE)])


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
    base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/processing_data/{args.dataset}'
    
    TRAIN_DATA_SET_PATH = glob(f'{base_path}/{node_id}/train/*/*[.png, .jpg]')
    
    # TESTING_DATA_SET_PATH = glob(f'{base_path}/{node_id}/test/*/*[.png, .jpg]')
    TEST_DATA_SET_PATH = glob(f'{base_path}/../../CIFAR10/test/*/*[.png, .jpg]') # test loader loads all test data of cifar10
    
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
