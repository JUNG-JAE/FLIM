# ----------- System library ----------- #
import sys
import os
from collections import defaultdict
from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
import statistics
import random

# ----------- Learning library ----------- #
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from models.vgg import vgg11_bn
import shutil
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torchvision.transforms import RandomInvert
from torchvision import transforms
from models.multi_head_model import multi_head_vgg11_bn
# ----------- Custom library ----------- #


def unnormalize(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(tensor.device)
    std = torch.tensor([0.5, 0.5, 0.5]).view(-1, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)


def get_label(dataset):
    labels = os.listdir(f'{os.path.expanduser("~")}/Workspace/DataSet/router_data/{dataset}/train')
    
    sorted_labels = sorted(
        [label for label in labels if 'model' in label],
        key=lambda x: int(x.replace('model', ''))
    )
    return sorted_labels


def set_multi_head_model(args, time_slot, labels, device):
    multi_head_model = multi_head_vgg11_bn(len(labels)).to(device)
    
    base_path = f'{os.path.expanduser("~")}/Workspace/FLIM/runs/{args.project}/{time_slot}/{args.node}'
    
    for idx, model_id in enumerate(labels):
        model = vgg11_bn().to(device)
        model.load_state_dict(torch.load(f'{base_path}/{model_id}.pt'))
        
        multi_head_model.features_list[idx] = model.features

    return multi_head_model


def load_gating_module(args, time_slot, multi_head_model):
    trained_model_path = f'{os.path.expanduser("~")}/Workspace/GateModule/controller/runs/{args.project}/{args.dataset}/{args.exp}/{args.node}/{time_slot}/router.pt'
    
    multi_head_model.load_state_dict(torch.load(trained_model_path))
    
    return


def load_cls_models(args, time_slot, labels, device):
    base_path = f'{os.path.expanduser("~")}/Workspace/FLIM/runs/{args.project}/{time_slot}/{args.node}'
    
    models = []
    
    for model_id in labels:
        model = vgg11_bn().to(device)
        model.load_state_dict(torch.load(f'{base_path}/{model_id}.pt'))
        models.append(model)
        
    return models


def save_model(model, base_path, model_name:str):
    os.makedirs(base_path, exist_ok=True)
    torch.save(model.state_dict(), f"{base_path}/{model_name}.pt")


def train(model, train_loader, epoch, learning_rate, device):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    
    for e in range(epoch):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()            
            outputs = model(inputs)
            
            loss = loss_function(outputs, targets)
            
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + 1e-4 * l1_norm
            
            loss.backward()
            optimizer.step()
            

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0.0
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum().item()

    accuracy = (correct / len(test_loader.dataset)) * 100
    
    return accuracy

def get_model_index(model_name):
    # modelN에서 N을 추출하여 인덱스로 변환
    return int(model_name.replace('model', ''))

@torch.no_grad()
def evaluate_gating_module(gating_module, CLS_LABELS, optimal_expert, test_loader, device):
    gating_module.eval()
    correct = 0.0
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = gating_module(inputs)
        _, predicts = outputs.max(1)

        for i in range(len(targets)):
            target_label = CLS_LABELS[targets[i].item()]
            optimal_model_index = get_model_index(optimal_expert[target_label])
            if predicts[i].item() == optimal_model_index:
                correct += 1

    accuracy = (correct / len(test_loader.dataset)) * 100
    
    return accuracy


@torch.no_grad()
def evaluate_and_save(model, test_loader, labels, iteration, base_path, device, conf_level):
    save_dir = f'{base_path}/confidence/{iteration}'
    
    model.eval()
    correct = 0.0
    confidence_count = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predicted_labels = outputs.max(1)
        confidence, _ = outputs.max(dim=1)

        for idx, (conf, predict, label, image) in enumerate(zip(confidence, predicted_labels, targets, inputs)):
            if conf > conf_level:
                correct += predict.eq(targets[idx]).item()
                confidence_count += 1
                unnormalized_image = unnormalize(image)
            
                label_dir = os.path.join(save_dir, labels[predict.item()])
                
                os.makedirs(label_dir, exist_ok=True)
                img_save_path = os.path.join(label_dir, f'{labels[predict.item()]}_{batch_idx}_{idx}.png')
            
                save_image(unnormalized_image.cpu(), img_save_path)
    
    conf_accuracy = (correct / confidence_count) * 100 if confidence_count > 0 else 0
    
    return conf_accuracy
    

@torch.no_grad()
def evaluate_and_save_correct_high_confidence(model, test_loader, device, conf_level):
    PSEUDO_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    model.eval()
    correct_high_confidence_count = 0  # 0.99 이상인 소프트맥스 값을 가진, 정확하게 예측한 예측의 수
    save_dir = f'./data/confidence_data/{conf_level}'  # 저장할 디렉터리의 기본 경로

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model.forward_softmax(inputs)
        _, predicted_labels = outputs.max(1)
        confidence, _ = outputs.max(dim=1)

        for idx, (conf, pred) in enumerate(zip(confidence, predicted_labels)):
            if conf.item() >= conf_level:
                correct_high_confidence_count += 1

                # 이미지 저장 경로 생성
                label_dir = os.path.join(save_dir, PSEUDO_LABELS[pred.item()])
                os.makedirs(label_dir, exist_ok=True)
                img_save_path = os.path.join(label_dir, f'highconf_{PSEUDO_LABELS[pred.item()]}_{batch_idx}_{idx}.png')

                # save_image 함수를 사용하여 이미지 저장
                save_image(inputs[idx].cpu(), img_save_path)

    accuracy = (correct_high_confidence_count / len(test_loader.dataset)) * 100 if len(test_loader.dataset) > 0 else 0
    
    print(f"Accuracy among high confidence: {accuracy:.2f}%")
    print(f"Total high confidence correct: {correct_high_confidence_count}")
    
    return accuracy, correct_high_confidence_count


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    

def load_expert(args, model_id, device):
    model = vgg11_bn().to(device)
    base_path = f'{os.path.expanduser("~")}/Workspace/FLIM/runs/{args.project}/{args.node}'
    model.load_state_dict(torch.load(f'{base_path}/{model_id}.pt'))
    
    return model


def test_router(args, time_slot, multi_head_model, test_loader, labels, device):
    experts = load_cls_models(args, time_slot, labels, device)
    
    correct = 0.0
    multi_head_model.eval()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = multi_head_model(inputs)
        _, predicts = outputs.max(1)
        
        expert = experts[predicts.item()].eval()
        expert_outputs = expert(inputs)
        _, expert_predicts = expert_outputs.max(1)
        
        correct += expert_predicts.eq(targets).item()
    
    accuracy = (correct / len(test_loader.dataset)) * 100
    
    return accuracy

@torch.no_grad()
def evaluate_correct_num(model, test_loader, device):
    model.eval()
    correct = 0.0
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum().item()

    # accuracy = (correct / len(test_loader.dataset)) * 100
    
    return correct, len(test_loader.dataset)