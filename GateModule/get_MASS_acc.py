import os
import pandas as pd
import numpy as np


def main():
    top_rows = pd.DataFrame()
    
    predict_true = 0
    predict_false = 0
    
    for label in LABELS:
        exp_xlsx_path = f'{os.path.expanduser("~")}/Workspace/PASS/controller/runs/{PROJECT}/{DATASET}/{EXP}/{NODE}/14/{label}/result/{label}.xlsx'
        data = pd.read_excel(exp_xlsx_path)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['value'], inplace=True)
        
        filtered_rows = data[data['user_attrs_predict'] == True] # predict이 True인 경우만 가져온다.
        
        if len(filtered_rows) > 0:
            predict_true += 1
        else:
            predict_false += 1
        
    print(f"{predict_true}, {predict_false}")
    print(f"Accuracy: {predict_true/(predict_true + predict_false)*100:.2f}")
    
    return


if __name__ == '__main__':
    PROJECT = 'cifar30_d_w30_node30_sim05_E29_epoch3_batch64_lambda5_straggler0'
    DATASET = 'cifar30_CNN_clusterted'
    EXP = '150_40_Epoch_30_80_withFC_MDL_0.6'
    NODE = 'node2'
    
    LABELS = ['apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
    'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
    'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin']
    
    main()