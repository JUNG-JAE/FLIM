# ------------ System library ------------ #
import os
import numpy as np
import pathlib
import logging

# ------------ Custom library ------------ #
from conf import settings

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

def format_title(title, width=40):
    return f" {title} ".center(width, '-')

def format_time_title(title, width=40):
    return f" {title} ".center(width, '=')

def print_log(logger, msg):
    print(msg)
    logger.info(msg)
    

def slicer(s):
    if "hard" in s:
        return s[:s.index("hard")+4]
    elif "soft" in s:
        return s[:s.index("soft")+4]
    else:
        return s

        
def poisson_distribution(n_node:int) -> list:
    LAMBDA = 60 / ((settings.INF_EPOCH+settings.SUP_EPOCH/2) * settings.TRAINING_TIME) # The unit of 60 is 'sec'
    TRANSACTION_PER_MINUTE = LAMBDA * n_node
    
    return [np.random.poisson(TRANSACTION_PER_MINUTE) for _ in range(settings.SIMULATION_TIME)]

def flatten_tuple(tup):
    result = []
    for item in tup:
        if isinstance(item, tuple):
            result.extend(flatten_tuple(item))
        else:
            result.append(item)
    return result

