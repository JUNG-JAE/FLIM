import os
import logging
import numpy as np
import pandas as pd
import shutil
import random

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_logger(base_path):
    create_directory(f"{base_path}/logs/")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(filename=f"{base_path}/logs/result.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def print_log(logger, msg):
    print(msg)
    logger.info(msg)


def copy_images(src_dir, dst_dir, cls_label, prefix):
    src_path = os.path.join(src_dir, cls_label)
    if os.path.exists(src_path):        
        for filename in os.listdir(src_path):
            # 이미지 파일만 복사
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                new_filename = f"{prefix}_{filename}"
                shutil.copy(os.path.join(src_path, filename), os.path.join(dst_dir, new_filename))
                
""" """
def get_expert_labels(args, labels, time_slot):
    # Load MASS result and integrate all label info
    top_rows = pd.DataFrame()
    
    counter = 0
    
    for label in labels:
        exp_xlsx_path = f'{os.path.expanduser("~")}/Workspace/PASS/controller/runs/{args.project}/{args.dataset}/{args.exp}/{args.node}/{label}/result/{label}.xlsx' #for legacy
        # exp_xlsx_path = f'{os.path.expanduser("~")}/Workspace/PASS/controller/runs/{args.project}/{args.dataset}/{args.exp}/{args.node}/{time_slot}/{label}/result/{label}.xlsx'
        data = pd.read_excel(exp_xlsx_path)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['value'], inplace=True)
        
        # # filtered_rows = data[(data['user_attrs_predict'] == True) & (data['value'] > 0)] # predict이 True이면서 objective value가 0이상인 값만 가져온다.
        filtered_rows = data[data['user_attrs_predict'] == True] # predict이 True인 경우만 가져온다.
        
        if len(filtered_rows) > 0:
            top_row = filtered_rows.sort_values(by='value', ascending=False).iloc[0]
        else:
            top_row = data.sort_values(by='value', ascending=False).iloc[0]
        
        top_row['Label'] = label
        top_rows = top_rows.append(top_row)
        
    top_rows.reset_index(drop=True, inplace=True)
    
    # Select expert per label
    model_label_dict = {}
    for label, accuracy_dic in zip(top_rows.Label, top_rows.user_attrs_Accuracy):
        accuracy_dic = eval(accuracy_dic)
        max_accuracy_model = max(accuracy_dic, key=accuracy_dic.get)
        
        if max_accuracy_model in model_label_dict:
            model_label_dict[max_accuracy_model].append(label)
        else:
            model_label_dict[max_accuracy_model] = [label]

    return list(model_label_dict.keys()), model_label_dict


def get_expert_labels_with_acc(args, logger, labels, time_slot):
    # Load MASS result and integrate all label info
    top_rows = pd.DataFrame()
    
    correct = 0
    
    for label in labels:
        exp_xlsx_path = f'{os.path.expanduser("~")}/Workspace/PASS/controller/runs/{args.project}/{args.dataset}/{args.exp}/{args.node}/{label}/result/{label}.xlsx'
        # exp_xlsx_path = f'{os.path.expanduser("~")}/Workspace/PASS/controller/runs/{args.project}/{args.dataset}/{args.exp}/{args.node}/{time_slot}/{label}/result/{label}.xlsx'
        
        # exp_xlsx_path = f'{os.path.expanduser("~")}/Workspace/PASS/controller/runs/{args.project}/{args.dataset}_5expert/{args.exp}/{args.node}/{time_slot}/{label}/result/{label}.xlsx'
        data = pd.read_excel(exp_xlsx_path)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['value'], inplace=True)
        
        # raind_index = random.randint(0, 9)
        # top_row = data.sort_values(by='value', ascending=False).iloc[raind_index]
        
        top_row = data.sort_values(by='value', ascending=False).iloc[0]
        
        top_row['Label'] = label
        
        # print_log(logger, f"{label}: {top_row['user_attrs_predict']}")
        
        # if top_row['user_attrs_predict']:
        #     correct += 1
        
        top_rows = top_rows.append(top_row)
        
    top_rows.reset_index(drop=True, inplace=True)
    
    # Select expert per label
    model_label_dict = {}
    for label, accuracy_dic in zip(top_rows.Label, top_rows.user_attrs_Accuracy):
        accuracy_dic = eval(accuracy_dic)
        max_accuracy_model = max(accuracy_dic, key=accuracy_dic.get)
        
        print(max_accuracy_model)
        
        if max_accuracy_model in model_label_dict:
            model_label_dict[max_accuracy_model].append(label)
        else:
            model_label_dict[max_accuracy_model] = [label]

    print_log(logger, f"MASS accuracy: {correct}/10")
    return list(model_label_dict.keys()), model_label_dict


def create_gate_module_dataset(args, logger, time_slot, expert_labels_dict):
    clust_train_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/kmeans_cnn_data/{args.dataset}_40/test' # 상위 30%의 데이터만 사용하기 위해 test 폴더에 있는 데이터를 가져온다
    candidate_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/processing_data/cifar10_unlabeled_40_candidate'
    true_test_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/genesis_data/CIFAR10_64/test'

    # ============================================================================================================
    
    # clust_train_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/kmeans_cnn_data/{args.dataset}/{args.node}/test' # 상위 30%의 데이터만 사용하기 위해 test 폴더에 있는 데이터를 가져온다
    
    # candidate_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/kmeans_cnn_data/{args.dataset}_candidate/{args.node}' # unlabled 데이터를 클러스터링한 후 train이미지만 true label에 맞게 수정한 데이터 셋
    # true_test_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/genesis_data/CIFAR10_64/test'
    
    gate_data_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/router_data/{args.exp}_{args.node}_{time_slot}'
    
    # ============================================================================================================
    
    # clust_train_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/kmeans_cnn_data/{args.dataset}/{args.node}/test' # 상위 30%의 데이터만 사용하기 위해 test 폴더에 있는 데이터를 가져온다
    
    # candidate_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/kmeans_cnn_data/{args.dataset}/{args.node}' # unlabled 데이터를 클러스터링한 후 train이미지만 true label에 맞게 수정한 데이터 셋
    # true_test_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/genesis_data/CIFAR30/test'
    
    # gate_data_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/router_data/{args.exp}_{args.node}_{time_slot}'
    
    
    
    for gate_label, cls_labels in expert_labels_dict.items():
        print_log(logger, f'{gate_label}: {cls_labels}')

        # create [train, candiate, test] directory per classification label
        os.makedirs(f'{gate_data_base_path}/train/{gate_label}', exist_ok=True)
        os.makedirs(f'{gate_data_base_path}/candidate/{gate_label}', exist_ok=True)
        os.makedirs(f'{gate_data_base_path}/test/{gate_label}', exist_ok=True)
        
        # Copy image each classification label
        for cls_label in cls_labels:
            copy_images(clust_train_base_path, f'{gate_data_base_path}/train/{gate_label}', cls_label, 'train')
            copy_images(candidate_base_path, f'{gate_data_base_path}/candidate/{gate_label}', cls_label, 'candidate')
            copy_images(true_test_base_path, f'{gate_data_base_path}/test/{gate_label}', cls_label, 'test')
    
    return


def remove_gate_module_dataset(args, time_slot):
    gate_data_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/router_data/{args.exp}_{args.node}_{time_slot}'
    
    if os.path.exists(gate_data_base_path):
        shutil.rmtree(gate_data_base_path)
        print(f'Directory: {args.exp}_{args.node}_{time_slot} has been deleted.')
    else:
        print(f'Directory: {args.exp}_{args.node}_{time_slot} dose not exist.')
        
    return