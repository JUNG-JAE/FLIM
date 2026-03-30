import os
import re
import torch
import torch.nn as nn
from models.vgg import vgg11_bn
from collections import Counter
from data_lodaer import src_testloader
import argparse
import logging


def set_logger(node_id):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(filename=f"{node_id}.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def print_log(logger, msg):
    print(msg)
    logger.info(msg)


def get_last_time_slot_logs(base_path):
    start_reading = False
    collected_lines = []

    with open(base_path, 'r') as file:
        for line in file:
            if "============== Time 14-15 ==============" in line:
                start_reading = True
                continue
            if start_reading:
                collected_lines.append(line.strip())

    return collected_lines


def get_node_logs(base_path, node_id):
    logs = get_last_time_slot_logs(base_path)
    node0_logs = []
    node0_found = False
    count = 0

    for line in logs:
        if f'[ {node_id} ]' in line:
            node0_found = True
            node0_logs.append(line)
            continue
        if node0_found and count < NUM_NODE:
            node0_logs.append(line)
            count += 1
        if count == NUM_NODE-1:
            break

    return node0_logs


def get_most_common_node(line):
    node_list = line.split(":")[1].strip().replace('[', '').replace(']', '').replace("'", "").split(', ')
    most_common_node = Counter(node_list).most_common(1)[0][0]
    return most_common_node


def sort_models(model_list):
    def extract_number(filename):
        s = re.findall("\d+", filename)
        return int(s[0]) if s else -1

    model_list.sort(key=extract_number)
    return model_list


def rename_model(base_path, time_slot, model_true_name):
    model_base_path = f'{base_path}/{time_slot}/{NODE}'
    
    os.rename(os.path.join(model_base_path, 'model0.pt'), os.path.join(model_base_path, f'{NODE}.pt'))
    
    model_list = [file_name.replace('.pt', '') for file_name in sort_models(os.listdir(f'{model_base_path}')) if 'node' not in file_name]
    print_log(logger, model_list)  
    
    for model_name in model_list:
        # if model_name in model_true_name:
        os.rename(os.path.join(model_base_path, f'{model_name}.pt'), os.path.join(model_base_path, f'{model_true_name[model_name]}.pt'))
            
    temp_model_list = [file_name.replace('.pt', '') for file_name in sort_models(os.listdir(f'{model_base_path}'))]
    
    for temp_model in temp_model_list:
        print(temp_model)
        number = re.search(r'\d+', temp_model)
        extracted_number = number.group()
        os.rename(os.path.join(model_base_path, f'node{extracted_number}.pt'), os.path.join(model_base_path, f'model{extracted_number}.pt'))
        # os.rename(os.path.join(model_base_path, f'node{temp_model[-1]}.pt'), os.path.join(model_base_path, f'model{temp_model[-1]}.pt'))


# def rename_model(base_path, time_slot, model_true_name):
#     model_base_path = f'{base_path}/{time_slot}/{NODE}'
    
#     # 초기 모델 이름 변경
#     initial_model_path = os.path.join(model_base_path, 'model0.pt')
#     if os.path.exists(initial_model_path):
#         os.rename(initial_model_path, os.path.join(model_base_path, f'{NODE}.pt'))
    
#     # 모델 목록 로드 및 필터링
#     model_list = [file_name.replace('.pt', '') for file_name in os.listdir(model_base_path) if 'node' not in file_name and file_name.endswith('.pt')]
    
#     # 정확한 이름으로 변경
#     for model_name in model_list:
#         if model_name in model_true_name:
#             os.rename(os.path.join(model_base_path, f'{model_name}.pt'), os.path.join(model_base_path, f'{model_true_name[model_name]}.pt'))
    
#     # 최종 모델 목록 로드
#     temp_model_list = [file_name.replace('.pt', '') for file_name in os.listdir(model_base_path) if file_name.endswith('.pt')]
    
#     # 숫자를 포함하는 모델 이름 변경
#     for temp_model in temp_model_list:
#         number_match = re.search(r'\d+', temp_model)
#         if number_match:
#             number = number_match.group()
#             os.rename(os.path.join(model_base_path, f'node{number}.pt'), os.path.join(model_base_path, f'model{number}.pt'))
            

@torch.no_grad()
def source_evaluate(model, model_id):
    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # LABELS = ['apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    # 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    # 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin']
    
    # LABELS = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    
    test_loader = src_testloader('CIFAR10_64')

    loss_function = nn.CrossEntropyLoss()
    class_correct = list(0. for i in range(len(LABELS)))
    class_total = list(0. for i in range(len(LABELS)))

    model.eval()
    device = torch.device('cuda')
    model.to(device)

    test_loss = 0.0
    correct = 0.0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
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
    print_log(logger, accuracy_per_class)
    accuracy_per_class.sort(key=lambda x: x[1], reverse=True)

    top_acc_labels = []
    for label, _ in accuracy_per_class[:2]:
        top_acc_labels.append(label)
        

def main():
    base_path = f'./runs/{PROJECT}'
    log_path = f'{base_path}/logs/result.log'
    
    # 각 node의 model 이름을 통일하기 위해 Log파일을 불러 온다.
    node_logs = get_node_logs(log_path, NODE)

    model_true_name = {}
    for i, line in enumerate(node_logs):
        if i == 0:
            continue
        most_common_node = get_most_common_node(line)
        model_true_name[f'model{i}'] = most_common_node
        print_log(logger, f"model{i} -> {most_common_node}")
        
    print_log(logger, " ")
    
    # model이름을 통일한다.
    rename_model(base_path, TIME_SLOT, model_true_name)
    
    # model의 acc를 확인한다.
    model_list = [file_name.replace('.pt', '') for file_name in sort_models(os.listdir(f'{base_path}/{TIME_SLOT}/{NODE}'))]
    for model_id in model_list:
        print_log(logger, f"{model_id} evaluate")
        model = vgg11_bn().to(torch.device('cuda'))
        model.load_state_dict(torch.load(f'{base_path}/{TIME_SLOT}/{NODE}/{model_id}.pt'))
        
        source_evaluate(model, model_id)
        print_log(logger, " ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--node_id', type=str, default='node7', help="set node ID")
    
    args = parser.parse_args()
    
    PROJECT = 'cifar10_64_random_node10_sim05_E10_epoch3_batch64_lambda7_straggler0'
    NODE = args.node_id
    NUM_NODE = 10
    logger = set_logger(NODE)
    
    # for time in range(0, 15):
    #     TIME_SLOT = time
    #     print_log(logger, f"============ {TIME_SLOT} ============")
    #     main()
    
    TIME_SLOT = 14
    print_log(logger, f"============ {TIME_SLOT} ============")
    main()

