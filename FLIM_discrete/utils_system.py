# ------------ System library ------------ #
import os
import shutil
import numpy as np
import pathlib
import logging
import matplotlib.pyplot as plt
import random

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

        
def generate_poisson_events(rate, time_duration):
    num_events = max(np.random.poisson(rate), 1)
    event_times = np.sort(np.random.uniform(0, time_duration, num_events))
    # inter_arrival_times = np.diff(event_times)
    
    return num_events, event_times


def plot_poisson_event(base_path, node_id, num_events, event_times):
    plt.title(f'Poisson Process Event Times\nTotal: {num_events} events\n')
    
    plt.step(event_times, np.arange(1, num_events + 1, 1), where='post', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Event Number')
    plt.xticks(np.arange(0, settings.SIMULATION_TIME, 1))
    
    create_directory(f'{base_path}/events')
    plt.savefig(f'{base_path}/events/{node_id}.png')
    plt.clf()
    

def node_event_generator(args, base_path):
    straggler_nodes = random.sample([f'node{node_idx}' for node_idx in range(0, args.n_node)], args.straggler)
    
    node_event_times = {}
    
    for node_idx in range(0, args.n_node):
        if f'node{node_idx}' in straggler_nodes:
            num_events, event_times = generate_poisson_events(1, settings.SIMULATION_TIME)
        else:
            num_events, event_times = generate_poisson_events(args.lamb, settings.SIMULATION_TIME)
        plot_poisson_event(base_path, f'node{node_idx}', num_events, event_times)
        node_event_times[f'node{node_idx}'] = event_times

    all_events = [(time, node_id) for node_id, times in node_event_times.items() for time in times]
    all_events.sort(key=lambda x: x[0])
    
    return straggler_nodes, all_events    


def flatten_tuple(tup):
    result = []
    for item in tup:
        if isinstance(item, tuple):
            result.extend(flatten_tuple(item))
        else:
            result.append(item)
    return result


def copy_prior_time_slot(base_path, minute):
    if minute == 0:
        create_directory(f'{base_path}/0')
    elif minute != 0:
        shutil.copytree(f'{base_path}/{minute-1}', f'{base_path}/{minute}', dirs_exist_ok=True)

