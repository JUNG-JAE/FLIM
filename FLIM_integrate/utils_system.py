import os
import shutil
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

from conf import settings


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_logger(base_path):
    create_directory(f"{base_path}/logs/")

    logger = logging.getLogger(base_path)  # unique name to avoid collision
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(filename=f"{base_path}/logs/result.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def format_title(title, width=50):
    return f" {title} ".center(width, '-')


def print_log(logger, msg):
    print(msg)
    logger.info(msg)


def flatten_tuple(tup):
    result = []
    for item in tup:
        if isinstance(item, (tuple, list)):
            result.extend(flatten_tuple(item))
        else:
            result.append(item)
    return result


def get_beta_distribution(alpha_param, beta_param):
    x = np.linspace(0, 1, 1000)
    cdf = beta_dist.cdf(x, alpha_param + 1, beta_param + 1)
    probability_over_50 = np.mean(cdf > 0.5)
    return probability_over_50


# Visualization for SimPy continuous-time events
def plot_node_events_timeline(base_path, node_events_dict, simulation_time):
    create_directory(f'{base_path}/events')

    fig, ax = plt.subplots(figsize=(14, max(6, len(node_events_dict) * 0.5)))

    node_ids = sorted(node_events_dict.keys(), key=lambda x: int(x.replace('node', '')))

    for y_pos, node_id in enumerate(node_ids):
        events = node_events_dict[node_id]
        ax.scatter(events, [y_pos] * len(events), marker='|', s=200, linewidths=2, zorder=3)
        ax.text(-0.5, y_pos, f'{node_id} ({len(events)})', va='center', ha='right', fontsize=9)

    ax.set_yticks(range(len(node_ids)))
    ax.set_yticklabels(['' for _ in node_ids])
    ax.set_xlabel('Simulation Time (continuous)')
    ax.set_title('FLIM_integrate — Event Timeline')
    ax.set_xlim(-0.2, simulation_time + 0.2)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_path}/events/all_nodes_timeline.png', dpi=150)
    plt.close()


def plot_single_node_events(base_path, node_id, event_times, simulation_time):
    create_directory(f'{base_path}/events')

    num_events = len(event_times)
    plt.figure(figsize=(10, 4))
    plt.title(f'{node_id} - Events\nTotal: {num_events} events')

    if num_events > 0:
        plt.step(event_times, np.arange(1, num_events + 1), where='post', color='blue')

    plt.xlabel('Time (continuous)')
    plt.ylabel('Cumulative Events')
    plt.xlim(0, simulation_time)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{base_path}/events/{node_id}.png', dpi=100)
    plt.close()
