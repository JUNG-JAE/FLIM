# ----------- Learning library ----------- #
import torch
import torch.nn as nn
import torch.optim as optim

# ------------ system library ------------ #
from tqdm import tqdm
import random
import numpy as np

# ------------ custom library ------------ #
from conf import settings
from utils_system import print_log
from utils_learning import get_network
from data_lodaer import node_dataloader, node_rot_dataloader
from models.vgg_rotation import vgg11_bn as vgg_rot

class Node:
    def __init__(self, args, logger, node_id:str):
        self.args = args
        self.node_id = node_id
        self.logger = logger
        self.device = torch.device('cuda')
        self.model = get_network(args).to(torch.device('cpu'))
        self.rot_model = vgg_rot().to(torch.device('cpu'))
        self.other_models = {}
        self.loss_function = nn.CrossEntropyLoss()
        self.train_loader, self.test_loader = node_dataloader(self.args, self.node_id)
        self.rot_train_loader, self.rot_test_loader = node_rot_dataloader(self.args, self.node_id)
        
        self.recv_status = True
        self.total_epoch = 0

    def train(self):      
        self.model.to(torch.device('cuda'))  
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=settings.LEARNING_RATE)
        
        self.total_epoch += settings.CLS_EPOCH
        
        print_log(self.logger, f"Classification - Training epoch: {settings.CLS_EPOCH} | Total epoch: {self.total_epoch}")
        
        avg_train_loss = 0.0
        for _ in np.arange(0, settings.CLS_EPOCH):
            train_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = self.loss_function(outputs, targets)
                train_loss += (loss.item() / len(self.train_loader.dataset))
                loss.backward()
                
                optimizer.step()

            avg_train_loss += train_loss
        print_log(self.logger, f"Avg train loss: {(avg_train_loss / settings.CLS_EPOCH):.2f}")
        self.model.to(torch.device('cpu'))  
    
    @torch.no_grad()
    def evaluate(self):
        self.model.to(torch.device('cuda'))  
        self.model.eval()
        correct = 0.0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)

            _, predicts = outputs.max(1)
            correct += predicts.eq(targets).sum()

        print_log(self.logger, f"Accuracy: {correct.float() * 100 / len(self.test_loader.dataset):.2f}")
        self.model.to(torch.device('cpu'))
        
    @torch.no_grad()
    def source_evaluate(self):
        LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # LABELS = ['apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
        # 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
        # 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin']
        # LABELS = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
        
        loss_function = nn.CrossEntropyLoss()
        class_correct = list(0. for i in range(len(LABELS)))
        class_total = list(0. for i in range(len(LABELS)))
        
        self.model.to(torch.device('cuda'))  
        self.model.eval()

        test_loss = 0.0
        correct = 0.0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = loss_function(outputs, targets)

            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()

            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            test_loss += loss.item()
            _, predicts = outputs.max(1)
            correct += predicts.eq(targets).sum()

        accuracy_per_class = [(LABELS[i], 100 * class_correct[i] / class_total[i]) for i in range(len(LABELS))]
        print_log(self.logger, accuracy_per_class)
        self.model.to(torch.device('cpu'))   
        
    def train_rot(self):
        self.rot_model.to(torch.device('cuda'))  
        self.rot_model.train()
        optimizer = optim.Adam(self.rot_model.parameters(), lr=settings.LEARNING_RATE)
        
        print_log(self.logger, f"Roation pretext task - Training epoch: {settings.ROT_EPOCH}")
        
        avg_train_loss = 0.0
        for _ in np.arange(0, settings.ROT_EPOCH):
            train_loss = 0.0
            for inputs, targets in self.rot_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.rot_model(inputs)
                loss = self.loss_function(outputs, targets)
                train_loss += (loss.item() / len(self.rot_train_loader.dataset))
                loss.backward()
                
                optimizer.step()

            avg_train_loss += train_loss
        print_log(self.logger, f"Avg train loss: {(avg_train_loss / settings.ROT_EPOCH):.2f}")
        self.rot_model.to(torch.device('cpu'))  
        
    @torch.no_grad()
    def evaluate_rot(self):
        self.rot_model.to(torch.device('cuda'))  
        self.rot_model.eval()
        correct = 0.0

        for inputs, targets in self.rot_test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.rot_model(inputs)

            _, predicts = outputs.max(1)
            correct += predicts.eq(targets).sum()

        print_log(self.logger, f"Accuracy: {correct.float() * 100 / len(self.rot_test_loader.dataset):.2f}")
        self.rot_model.to(torch.device('cpu'))  
        
def migrate_cls_to_rot(cls_model, rot_model):
    rotation_model = rot_model
    rotation_model_dict = rotation_model.state_dict()

    for layer, params in cls_model.state_dict().items():
        if 'classifier' not in layer:
            rotation_model_dict[layer] = params

    rotation_model_dict['classifier.0.weight'] = cls_model.state_dict()['classifier.0.weight']
    rotation_model_dict['classifier.0.bias'] = cls_model.state_dict()['classifier.0.bias']

    rotation_model.load_state_dict(rotation_model_dict)

    for name, param in rotation_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = True

    rotation_model_dict['classifier.0.weight'].requires_grad = True
    rotation_model_dict['classifier.0.bias'].requires_grad = True

    for layer in rotation_model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
            layer.weight.requires_grad=True
            layer.bias.requires_grad=True
            layer.track_running_stats=True

    return rotation_model
    
def migrate_rot_to_cls(rot_model, cls_model):
    classification_model = cls_model
    cls_model_dict = classification_model.state_dict()

    for layer, params in rot_model.state_dict().items():
        if 'classifier' not in layer:
            cls_model_dict[layer] = params

    cls_model_dict['classifier.0.weight'] = rot_model.state_dict()['classifier.0.weight']
    cls_model_dict['classifier.0.bias'] = rot_model.state_dict()['classifier.0.bias']

    classification_model.load_state_dict(cls_model_dict)

    for name, param in classification_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = True

    cls_model_dict['classifier.0.weight'].requires_grad = True
    cls_model_dict['classifier.0.bias'].requires_grad = True

    for layer in classification_model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
            layer.weight.requires_grad=True
            layer.bias.requires_grad=True
            layer.track_running_stats=True

    return classification_model
