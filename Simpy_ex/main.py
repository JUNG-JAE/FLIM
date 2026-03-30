import sys
import os

# Ensure the FLIM_Simpy directory is on the path
sys.path.insert(0, os.path.dirname(__file__))
# Ensure the parent FLIM directory is on the path for model/data imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import datetime
import random
import simpy
import numpy as np

from conf import settings
from utils_system import print_log, set_logger, format_title, create_directory
from utils_learning import set_seed
from sim_engine import SimulationCoordinator
from node_process import SimPyNode


def main():
    parser = argparse.ArgumentParser(description='FLIM SimPy-based Federated Learning Simulation')
    parser.add_argument('--dataset', type=str, default='cifar10_example', help='dataset name (default: cifar10_example)')
    parser.add_argument('--n_node', type=int, default=10, help='number of nodes in the network (default: 10)')
    parser.add_argument('--sim_th', type=float, default=0.6, help='similarity threshold for clustering (default: 0.6)')
    parser.add_argument('--lamb', type=int, default=6, help='Poisson lambda rate - expected events per simulation (default: 6)')
    parser.add_argument('--straggler', type=int, default=0, help='number of straggler nodes (default: 0)')
    parser.add_argument('--mode', type=str, default='withRot', help='training mode: withRot or noRot (default: withRot)')
    parser.add_argument('--net', type=str, default='vgg11', help='neural network architecture (default: vgg11)')
    parser.add_argument('--gpu', action='store_true', default=True, help='use GPU if available (default: True)')
    parser.add_argument('--sim_time', type=float, default=None, help='override simulation time (default: from settings)')
    parser.add_argument('--batch_window', type=float, default=None, help='override event batch window (default: from settings)')
    args = parser.parse_args()

    # Override settings if provided
    if args.sim_time is not None:
        settings.SIMULATION_TIME = args.sim_time
    if args.batch_window is not None:
        settings.EVENT_BATCH_WINDOW = args.batch_window

    # ----------------------------------------------------------
    # Setup paths and logger
    # ----------------------------------------------------------
    BASE_PATH = (
        f'{settings.LOG_DIR}/{args.dataset}'
        f'_node{args.n_node}'
        f'_sim{str(args.sim_th).replace(".", "")}'
        f'_E{settings.SUP_OTHER_MODEL_SIZE}'
        f'_epoch{settings.CLS_EPOCH}'
        f'_batch{settings.BATCH_SIZE}'
        f'_lambda{args.lamb}'
        f'_straggler{args.straggler}'
        f'_simpy'
    )
    logger = set_logger(BASE_PATH)

    print_log(logger, format_title("FLIM SimPy Simulation"))
    print_log(logger, f"SimPy continuous-time Poisson process simulation")
    print_log(logger, f"Simulation time: {settings.SIMULATION_TIME}s")
    print_log(logger, f"Event batch window: {settings.EVENT_BATCH_WINDOW}s")
    print_log(logger, f"Nodes: {args.n_node} | Lambda: {args.lamb} | Stragglers: {args.straggler}")
    print_log(logger, f"Network: {args.net} | Similarity threshold: {args.sim_th}")
    print_log(logger, "")

    # ----------------------------------------------------------
    # Create SimPy environment
    # ----------------------------------------------------------
    env = simpy.Environment()

    # Create coordinator
    coordinator = SimulationCoordinator(env, args, logger, BASE_PATH)

    # ----------------------------------------------------------
    # Select straggler nodes
    # ----------------------------------------------------------
    all_node_ids = [f'node{i}' for i in range(args.n_node)]
    straggler_nodes = random.sample(all_node_ids, min(args.straggler, args.n_node))
    print_log(logger, f"Straggler nodes: {straggler_nodes}")

    # ----------------------------------------------------------
    # Create SimPy nodes (each starts its own process)
    # ----------------------------------------------------------
    for node_idx in range(args.n_node):
        node_id = f'node{node_idx}'
        is_straggler = node_id in straggler_nodes

        node = SimPyNode(
            env=env,
            args=args,
            logger=logger,
            node_id=node_id,
            coordinator=coordinator,
            is_straggler=is_straggler
        )
        coordinator.register_node(node)

    # ----------------------------------------------------------
    # Run simulation
    # ----------------------------------------------------------
    start_time = datetime.datetime.now()
    print_log(logger, f"Execution started at: {start_time}")
    print_log(logger, "")

    # Run SimPy simulation until SIMULATION_TIME
    env.run(until=settings.SIMULATION_TIME + settings.EVENT_BATCH_WINDOW * 2)

    end_time = datetime.datetime.now()
    execution_time = end_time - start_time

    # ----------------------------------------------------------
    # Post-simulation
    # ----------------------------------------------------------
    print_log(logger, "")
    print_log(logger, format_title("Simulation Complete"))

    summary = coordinator.get_summary()
    print_log(logger, f"Total rounds processed: {summary['total_rounds']}")
    print_log(logger, f"Total events generated: {summary['total_events']}")
    print_log(logger, f"Events per node:")
    for node_id, count in summary['events_per_node'].items():
        is_str = " (straggler)" if node_id in straggler_nodes else ""
        print_log(logger, f"  {node_id}: {count} events{is_str}")

    print_log(logger, "")
    print_log(logger, f"Execution ended at: {end_time}")
    print_log(logger, f"Execution time: {execution_time}")

    # Generate visualizations
    coordinator.generate_visualizations()
    print_log(logger, f"Event timeline plots saved to {BASE_PATH}/events/")

    return 0


if __name__ == '__main__':
    set_seed()
    main()
