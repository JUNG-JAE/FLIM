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
from data_lodaer import node_dataloader


class Node:
    def __init__(self, args, logger, node_id:str):
        self.args = args
        self.node_id = node_id
        self.logger = logger
        self.model = get_network(args)
        self.other_models = {}
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer =optim.Adam(self.model.parameters(), lr=settings.LEARNING_RATE)
        self.train_loader, self.test_loader = node_dataloader(self.args, self.node_id)
        self.device = torch.device('cuda' if args.gpu else 'cpu')

    def train(self):
        self.model.train()
        rand_epoch = 1
        # rand_epoch = random.randint(settings.INF_EPOCH, settings.SUP_EPOCH)
        print_log(self.logger, f"Training epoch: {rand_epoch}")
        
        avg_train_loss = 0.0
        for _ in np.arange(0, rand_epoch):
            train_loss = 0.0
            # progress = tqdm(total=len(self.train_loader.dataset), ncols=100)
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                train_loss += (loss.item() / len(self.train_loader))
                loss.backward()
                self.optimizer.step()

                # progress.update(settings.BATCH_SIZE)
            avg_train_loss += train_loss
            # progress.close()
        print_log(self.logger, f"Avg train loss: {(avg_train_loss / rand_epoch):.2f}")
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        test_loss = 0.0
        correct = 0.0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)

            test_loss += loss.item()
            _, predicts = outputs.max(1)
            correct += predicts.eq(targets).sum()

        print_log(self.logger, f"Acc: {correct.float() * 100 / len(self.test_loader.dataset):.2f}, Loss: {test_loss / len(self.test_loader.dataset):.2f}")
        
        
        return correct.float() * 100 / len(self.test_loader.dataset)
