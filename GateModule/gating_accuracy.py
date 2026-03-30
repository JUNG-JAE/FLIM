# Perfecet, normal, non-IID(2)[CLA, LA(0.2~0.5)], epxert5, non-IID(2~8)

import os
import torch
import argparse
import datetime
from models.vgg import vgg11_bn
from utils_learning import set_multi_head_model, train, evaluate, evaluate_and_save, save_model, test_router, load_gating_module, evaluate_gating_module
from utils_system import get_expert_labels_with_acc, create_gate_module_dataset, remove_gate_module_dataset, print_log, set_logger, get_expert_labels
from data_loader import trainloader, testloader, candidate_dataloader, confidence_dataloader, cls_dataloader, src_testloader
import pickle
import shutil

# expert 가져오는 코드, 각 expert 정확도 측정, 레이블 병 정확도 가장 높은 것 선택
# gating module 가져오는 코드
# 평가

def get_optimal_expert(logger, expert_list):
    optimal_label_expert = {}
    
    for label in CLS_LABELS:
        print_log(logger, f"====== {label} ======")
        test_loader = src_testloader(label)
        
        expert_accuracy = {}
        
        for expert in expert_list:
            # base_path = f'{os.path.expanduser("~")}/Workspace/FLIM/runs/{args.project}/{args.time_slot}/{args.node}'
            base_path = f'{os.path.expanduser("~")}/Workspace/FLIM/runs/{args.project}/14/{args.node}'
            model = vgg11_bn().to(device)
            model.load_state_dict(torch.load(f'{base_path}/{expert}.pt'))
            
            accuracy = evaluate(model, test_loader, device)
            print_log(logger, f"{expert}: {accuracy:.2f}")
            
            expert_accuracy[expert] = accuracy
            
        max_accuracy_expert = max(expert_accuracy, key=expert_accuracy.get)
        
        print_log(logger, f"Select: {max_accuracy_expert}\n")

        optimal_label_expert[label] = max_accuracy_expert
    
    print_log(logger, optimal_label_expert)
    
    return optimal_label_expert


def main():    
    exper_labels, expert_labels_dict = get_expert_labels_with_acc(args, logger, CLS_LABELS, args.time_slot)
    
    print_log(logger, f'Time:{args.time_slot} | Expert model labels: {exper_labels} ({len(exper_labels)})')
     
    # optimal_expert_label = get_optimal_expert(logger, exper_labels)
     
    multi_head_model = set_multi_head_model(args, args.time_slot, exper_labels, device)
    load_gating_module(args, args.time_slot, multi_head_model)

    downstream_test_loader = cls_dataloader()
    
    # gating_accuracy = evaluate_gating_module(multi_head_model, CLS_LABELS, optimal_expert_label, downstream_test_loader, device)
    # print_log(logger, f"Gating accuracy: {gating_accuracy:.2f}")
    
    return
    

if __name__ == '__main__':
    device = torch.device('cuda')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='cifar10_h_w10_l2_aug_node10_sim05_E9_epoch_1_5_batch64', help='project name')
    parser.add_argument('--dataset', type=str, default='cifar10_64_CNN_clusterted', help='dataset name')
    parser.add_argument('--exp', type=str, default='1000_500_Epoch_10_50_0.0_withFC_normal_0.0', help='pass exp name')
    parser.add_argument('--node', type=str, default='node7', help='node id')
    parser.add_argument('--time_slot', type=int, default=14, help='current time slot')
    args = parser.parse_args()
    
    logger = set_logger(f'./runs/{args.project}/{args.dataset}/{args.exp}/{args.node}')
    
    CLS_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    BASE_PATH = f'./runs/{args.project}/{args.dataset}/{args.exp}/{args.node}/{args.time_slot}'

    main()
    