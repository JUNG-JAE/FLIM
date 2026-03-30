# ----------- System library ----------- #
import os
import joblib
import numpy as np
import random
import pandas as pd
import statistics
from scipy.stats import kurtosis, rankdata
import logging
from scipy.stats import beta
import re

# ----------- Learning library ----------- #
import torch
import torch.nn as nn
import torch.optim as optim

# ----------- Custom library ----------- #

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_result(study, base_path, file_name):
    path = f'{base_path}/result'
    
    create_directory(path)
    joblib.dump(study, f"{path}/{file_name}.pkl")
    

def load_result(base_path, file_name):
    path = f'{base_path}/result'
    
    study = joblib.load(f"{path}/{file_name}.pkl")
    df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete', 'number', 'duration'], axis=1)
    pd.options.display.max_rows = 100
    df.head(3)
    print(df)
    df.to_excel(f"{path}/{file_name}.xlsx")


def calculate_avg_acc(task_acc_per_trial):
    acc_sum_dict = {key: 0 for key in task_acc_per_trial[0].keys()}
    for task_acc in task_acc_per_trial:
        for key, value in task_acc.items():
            acc_sum_dict[key] += value
    return {key: round(value / len(task_acc_per_trial), 2) for key, value in acc_sum_dict.items()}


def calculate_avg_std_acc(task_acc_per_trial):
    acc_list_dict = {key: [] for key in task_acc_per_trial[0].keys()}
    for task_acc in task_acc_per_trial:
        for key, value in task_acc.items():
            acc_list_dict[key].append(value)
            
    avg_accs = {key: round(statistics.mean(values), 2) for key, values in acc_list_dict.items()}
    std_devs = {key: round(statistics.stdev(values), 2) for key, values in acc_list_dict.items()}
    kurts = {key: round(kurtosis(values), 2) for key, values in acc_list_dict.items()}

    
    return avg_accs, std_devs, kurts


def calculate_interval(avg_acc):
    sorted_values = sorted(avg_acc.values(), reverse=True)
    return sorted_values[0] - sorted_values[1]


def gradient(task_acc, x_1, x_2):
    y_gradient = abs(task_acc[x_2] - task_acc[x_1])
    x_gradient = abs(x_2 - x_1)
    
    return round(y_gradient / x_gradient, 2)


def Mann_Whitney_U(data1, data2):
    combined_data = data1 + data2
    ranks = rankdata([-x for x in combined_data])

    n1, n2 = len(data1), len(data2)

    R1 = sum(ranks[:n1])
    R2 = sum(ranks[n1:])

    U1 = n1 * n2 + (n1 * (n1 + 1)) / 2 - R1
    U2 = n1 * n2 + (n1 * (n2 + 1)) / 2 - R2

    return U1, U2


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


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_beta_distribution(alpha_param, beta_param):
    # alpha: 0보다 큰값, beta: 0보다 작은값
    
    x = np.linspace(0, 1, 1000)
    
    # pdf = beta.pdf(x, alpha_param+1, beta_param+1)
    cdf = beta.cdf(x, alpha_param+1, beta_param+1)

    # CDF가 0.5 이상인 값들의 비율 계산
    probability_over_50 = np.mean(cdf > 0.5)
        
    return probability_over_50


def sort_models(model_list):
    def extract_number(filename):
        s = re.findall("\d+", filename)
        return int(s[0]) if s else -1

    model_list.sort(key=extract_number)
    return model_list