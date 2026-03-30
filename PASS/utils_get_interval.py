import os
import pandas as pd
import statistics

def main():
    df = pd.read_excel(BASE_PATH)
    acc_with_label_df = df[['user_attrs_Accuracy', 'Label', 'params_train', 'params_test', 'user_attrs_STD', 'params_epoch']]
    
    acc_gap_list = []
    std_sum_list = []
    
    train_size_list = []
    test_size_list = []
    
    epoch_list = []
    
    for acc_with_label in acc_with_label_df.itertuples():
        acc_dict = eval(acc_with_label.user_attrs_Accuracy)
        sorted_keys = sorted(acc_dict, key=acc_dict.get, reverse=True)
        
        std_dict = eval(acc_with_label.user_attrs_STD)

        highest_value = acc_dict[sorted_keys[0]]
        second_highest_value = acc_dict[sorted_keys[1]]
        
        highest_std = std_dict[sorted_keys[0]]
        second_highest_std = std_dict[sorted_keys[1]]
        
        difference = highest_value - second_highest_value
        std_sum = highest_std + second_highest_std
        
        acc_gap_list.append(difference)
        std_sum_list.append(std_sum)
        
        train_size_list.append(acc_with_label.params_train)
        test_size_list.append(acc_with_label.params_test)

        epoch_list.append(acc_with_label.params_epoch)
    
    # print(f"STD sum avg: {statistics.mean(std_sum_list):.2f}")
    
    # print(f"Train data size {statistics.mean(train_size_list):.2f}")
    # print(f"Test data size {statistics.mean(test_size_list):.2f}")
    
    print(f"Avg Epoch: {statistics.mean(epoch_list)}")
        
    return

if __name__ == '__main__':
    EXPERIMENT = 'cifar10_h_w10_l2_aug_node10_sim05_E9_epoch_1_5_batch64'
    DATASET = 'cifar10_64_CNN_clusterted'
    PROJECT = '1000_500_Epoch_10_50_0.0_withFC_MDL_0.6'
    
    for i in [0, 1, 2, 3, 4, 5]:
        NODE = f'node{i}'
    
        BASE_PATH = f'./controller/runs/{EXPERIMENT}/{DATASET}/{PROJECT}/{NODE}/{PROJECT}_{NODE}.xlsx'
    
        main()
        
        print(" ")