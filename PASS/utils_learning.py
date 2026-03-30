# ----------- System library ----------- #
import sys
import os
from collections import defaultdict
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
import statistics
import copy
from scipy.stats import skew

# ----------- Learning library ----------- #
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from models.vgg import vgg11_bn
from models.vgg_rotation import vgg11_bn as vgg_rotation

# ----------- Custom library ----------- #
from conf import settings
from utils_system import get_beta_distribution, create_directory


def load_cls_model(args, model_id, device) -> torch:
    model = vgg11_bn().to(device)
    model.load_state_dict(torch.load(f'{os.path.expanduser("~")}/Workspace/FLIM/runs/{args.exp}/{args.slot}/{args.node_id}/{model_id}.pt'))

    return model


def save_model(model:torch, base_path, model_name:str) -> None:
    torch.save(model.state_dict(), f"{base_path}/{model_name}.pt")


def train_and_eval(args, model_id, model, train_loader, test_loader, epoch, learning_rate, device) -> list:
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    
    sample_accs = []
    
    for e in range(epoch):
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()            
            outputs = model(inputs)
            
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            
        if epoch - settings.SAMPLE_SIZE <= e:
            acc = evaluate(args, model_id, model, test_loader, device)
            sample_accs.append(acc)
    
    model.to(torch.device('cpu'))
    return sample_accs


def robust_train_and_eval(args, model_id, model, train_loader, test_loader, epoch, learning_rate, tau, device) -> list:
    model.to(torch.device('cuda'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(reduction='none')
    model.train()
    
    sample_accs = []
    
    for e in range(epoch):
        train_loss = 0.0
        for inputs, targets, _ in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            losses = loss_function(outputs, targets)  # 각 데이터에 대한 손실 계산

            # 상위 tau 만큼 데이터의 loss 만 사용
            sorted_losses, indices = torch.sort(losses)
            retain_num = int(len(losses) * tau)
            retained_indices = indices[:retain_num]
            
            retained_inputs = torch.index_select(inputs, 0, retained_indices)
            retained_targets = torch.index_select(targets, 0, retained_indices)
            
            optimizer.zero_grad()            
            retained_outputs = model(retained_inputs)
            retained_loss = loss_function(retained_outputs, retained_targets).mean()

            train_loss += retained_loss.item()
            retained_loss.backward()

            optimizer.step()
    
        if epoch - settings.SAMPLE_SIZE <= e:
            acc = evaluate(args, model_id, model, test_loader, device)
            sample_accs.append(acc)
    
    model.to(torch.device('cpu'))
    
    return sample_accs
    
    
def maximize_distinction_learning(args, model_id, model, train_loader, test_loader, epoch, learning_rate, device) -> list:
    model.to(torch.device('cuda'))
    loss_function = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    reliability_toggle = False    
    beta_posterior = {'negative_counter':0, 'positive_counter':0, 'cdf_probs':[], 'relied_epoch':0}
    
    log = {'sample_accs':[], 'negative_target_ratio':[], 'positive_target_ratio':[], 'whole_target_ratio':[], 'train_data_sizes':[]}
    
    for e in range(epoch):
        tracker = {'negative_target_mean_ratio':[], 'positive_target_mean_ratio':[], 'whole_target_mean_ratio':[], 'train_data':0}

        for inputs, targets, paths in train_loader:
            data_counter = {'negative_target_counter':0, 'positive_target_counter':0}
            
            model.eval()
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            losses = loss_function(outputs, targets)
            
            np_losses = copy.deepcopy(losses.cpu().detach().numpy())
            loss_mean, loss_std, loss_median, loss_skewness = np.mean(np_losses), np.std(np_losses), np.median(np_losses), skew(np_losses)
            
            tau_gap = 0.4
            
            if loss_skewness > 0 and loss_median + tau_gap < loss_mean:
                beta_posterior['positive_counter'] += 1
            else:
                beta_posterior['negative_counter'] += 1
            
            data_buffer = [] # save skewness, loss, actural label
            for input, target, label_path, loss in zip(inputs, targets, paths, np_losses):
                actual_label = label_path.split('/')[-1].split('_')[1]
                
                skewness_value = (loss - loss_mean) ** 3 / (loss_std**3)
                
                if actual_label == args.cls_label:
                    if skewness_value <= 0:
                        data_counter['negative_target_counter'] += 1
                    else:
                        data_counter['positive_target_counter'] += 1

                data_buffer.append((skewness_value, input, target))
            
            if reliability_toggle:
                retained_inputs = inputs
                retained_targets = targets
            else:
                data_buffer.sort(key=lambda x: x[0]) # skewness based ascending sort
                filtered_data = [item for item in data_buffer if item[0] <= 0]
                retained_inputs = torch.stack([item[1] for item in filtered_data])
                retained_targets = torch.stack([item[2] for item in filtered_data])
                    
            tracker['train_data'] += len(retained_inputs)
            
            model.train()
            optimizer.zero_grad()
            outputs = model(retained_inputs)
            loss = loss_function(outputs, retained_targets).mean()
            loss.backward()
            optimizer.step()
            
            tracker['negative_target_mean_ratio'].append(data_counter['negative_target_counter'] / targets.size(0) * 100)
            tracker['positive_target_mean_ratio'].append(data_counter['positive_target_counter'] / targets.size(0) * 100)
            tracker['whole_target_mean_ratio'].append((data_counter['negative_target_counter'] + data_counter['positive_target_counter']) / targets.size(0) * 100)
            
            
        in_dist_prob = get_beta_distribution(beta_posterior['negative_counter'], beta_posterior['positive_counter'])
        beta_posterior['cdf_probs'].append(in_dist_prob)
        
        if in_dist_prob >= 0.95:
            reliability_toggle = True
            beta_posterior['relied_epoch'] = e
        
        # print(f"Epoch: {e} [{beta_posterior['negative_counter']}|{beta_posterior['positive_counter']}] In-distribution prob: {in_dist_prob:.2f} Using data: {tracker['train_data']} ({np.mean(tracker['negative_target_mean_ratio']):.2f}|{np.mean(tracker['positive_target_mean_ratio']):.2f}) -> {np.mean(tracker['whole_target_mean_ratio']):.2f}")
        
        if epoch - settings.SAMPLE_SIZE <= e:
            acc = evaluate(args, model_id, model, test_loader, device)
            log['sample_accs'].append(acc)

        log['negative_target_ratio'].append(np.mean(tracker['negative_target_mean_ratio']))
        log['positive_target_ratio'].append(np.mean(tracker['positive_target_mean_ratio']))
        log['whole_target_ratio'].append(np.mean(tracker['whole_target_mean_ratio']))
        log['train_data_sizes'].append(tracker['train_data'])
     
    model.to(torch.device('cpu'))
    
    return log['sample_accs'], beta_posterior['cdf_probs'], beta_posterior['relied_epoch'], log
            

@torch.no_grad()
def evaluate(args, model_id, model, test_loader, device):
    model.eval()
    correct = 0.0
            
    for inputs, targets, _ in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)        
        outputs = model(inputs)
        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum().item()

    accuracy = (correct / len(test_loader.dataset)) * 100
    
    return accuracy


def load_CNN_layers(model, device):
    # Only load CNN layers from pre-trained model
    rotation_model = vgg_rotation().to(device)
    rotation_model_dict = rotation_model.state_dict()

    for layer, params in model.state_dict().items():
        if 'classifier' not in layer:
            rotation_model_dict[layer] = params
    
    rotation_model.load_state_dict(rotation_model_dict)

    for name, param in rotation_model.named_parameters():
        if 'classifier' not in name:  
            param.requires_grad = False

    for layer in rotation_model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
            layer.weight.requires_grad=True
            layer.bias.requires_grad=True
            layer.track_running_stats=True

    return rotation_model


def load_CNN_with_FC_layers(model, device):
    # Load CNN layers and first FC layer
    rotation_model = vgg_rotation().to(device)
    rotation_model_dict = rotation_model.state_dict()

    for layer, params in model.state_dict().items():
        if 'classifier' not in layer:
            rotation_model_dict[layer] = params

    rotation_model_dict['classifier.0.weight'] = model.state_dict()['classifier.0.weight']
    rotation_model_dict['classifier.0.bias'] = model.state_dict()['classifier.0.bias']

    rotation_model.load_state_dict(rotation_model_dict)

    for name, param in rotation_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    rotation_model_dict['classifier.0.weight'].requires_grad = True
    rotation_model_dict['classifier.0.bias'].requires_grad = True

    for layer in rotation_model.modules():
        if isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
            layer.weight.requires_grad=True
            layer.bias.requires_grad=True
            layer.track_running_stats=True

    return rotation_model


def find_saturation_point(accuracies, patience=5, threshold=0.01):

    no_improvement_count = 0
    max_accuracy = accuracies[0]
    
    for epoch, acc in enumerate(accuracies[1:], 1):  # Start from the second epoch
        if acc - max_accuracy < threshold:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
            max_accuracy = acc
        
        if no_improvement_count >= patience:
            return epoch - patience + 1  # +1 to convert to 1-indexed epoch

    return None  # No saturation point found within the given patience


@torch.no_grad()
def evaluate_softmax(model, test_loader, device):
    model.eval()

    correct = 0.0
    softmax_dict = {i: [] for i in range(4)}  # Assuming 10 classes; adjust this number if different

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        softmax_probabilities = torch.nn.functional.softmax(outputs, dim=1)

        for target, softmax_value in zip(targets, softmax_probabilities):
            softmax_dict[target.item()].append(softmax_value.tolist())

        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum()

    accuracy = (correct.float() * 100 / len(test_loader.dataset)).item()

    averaged_softmax_dict = {k: [round(sum(col) / len(col), 2) for col in zip(*v)] for k, v in softmax_dict.items()}

    return accuracy, averaged_softmax_dict
