import os
import pandas as pd
import shutil
import numpy as np

PROJECT = 'cifar10_h_w10_l2_aug_node10_sim05_E9_epoch_1_5_batch64'
DATASET = 'cifar10_64'
NODE = 'node0'
# PASS = '1000_500_Epoch_10_50_0.0_withFC_MDL_1.0_integrated'
# MODE = 'integrated'
PASS = '1000_500_Epoch_10_50_0.0_withFC_normal_0.0'
MODE = 'clusterted'

if MODE == 'integrated':
    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print('Integrated Mode')
    model_label_dict = {}
    label_data_path = f'{os.path.expanduser("~")}/Workspace/PASS/controller/runs/{PROJECT}/{DATASET}/{PASS}/{NODE}/result/summarize.xlsx'
    
    data = pd.read_excel(label_data_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=['value'], inplace=True)
    top_row = data.sort_values(by='value', ascending=False).iloc[0]
    
    df = pd.DataFrame(columns=[f'model{model_idx}' for model_idx in np.arange(0,10)], index=LABELS)
    
    for label in LABELS:
        for model_idx in range(0, 10):
            df.loc[label, f'model{model_idx}'] = eval(top_row[f'user_attrs_{label}_Accuracy'])[f'model{model_idx}']

    for label in LABELS:
        label_dict = df.loc[label].to_dict()
        argmax_model = max(label_dict, key=lambda k: label_dict[k])
        
        if argmax_model in model_label_dict:
            model_label_dict[argmax_model].append(label)
        else:
            model_label_dict[argmax_model] = [label]

else:
    print("Clusterted Mode")
    label_data_path = f'{os.path.expanduser("~")}/Workspace/PASS/controller/runs/{PROJECT}/{DATASET}/{PASS}/{NODE}/{PASS}_{NODE}.xlsx'
    df = pd.read_excel(label_data_path)
    model_label_dict = {}

    for label, accuracy_dic in zip(df.Label, df.user_attrs_Accuracy):
        accuracy_dic = eval(accuracy_dic)
        max_accuracy_model = max(accuracy_dic, key=accuracy_dic.get)
        
        if max_accuracy_model in model_label_dict:
            model_label_dict[max_accuracy_model].append(label)
        else:
            model_label_dict[max_accuracy_model] = [label]

clust_train_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/kmeans_cnn_data/{DATASET}_40/test'
candidate_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/processing_data/cifar10_unlabeled_40_candidate'
true_test_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/genesis_data/CIFAR10_64/test'

gate_data_base_path = f'{os.path.expanduser("~")}/Workspace/DataSet/router_data/{PASS}_{NODE}'


def copy_images(src_dir, dst_dir, cls_label, prefix):
    src_path = os.path.join(src_dir, cls_label)
    if os.path.exists(src_path):
        for filename in os.listdir(src_path):
            # 이미지 파일만 복사
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                new_filename = f"{prefix}_{filename}"
                shutil.copy(os.path.join(src_path, filename), os.path.join(dst_dir, new_filename))

# 모델별 폴더를 생성하고 이미지를 복사
for gate_label, cls_labels in model_label_dict.items():
    print(gate_label, cls_labels)

    # train, candidate, test 폴더를 모델별로 생성
    train_dir = os.path.join(gate_data_base_path, 'train', gate_label)
    candidate_dir = os.path.join(gate_data_base_path, 'candidate', gate_label)
    test_dir = os.path.join(gate_data_base_path, 'test', gate_label)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(candidate_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 각 레이블별로 이미지를 복사
    for cls_label in cls_labels:
        copy_images(clust_train_base_path, train_dir, cls_label, 'train')
        copy_images(candidate_base_path, candidate_dir, cls_label, 'candidate')
        copy_images(true_test_base_path, test_dir, cls_label, 'test')

print("Copying of images is complete.")