import sys
import os

# Path setup: own dir first (highest priority), then parent FLIM
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.join(_HERE, '..')
for p in (_PARENT, _HERE):            # insert order → _HERE ends at index 0
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import datetime
import random
import simpy
import numpy as np

from conf import settings
from utils_system import print_log, set_logger, format_title
from utils_learning import set_seed
from sim_engine import NodeRegistry
from node_process import SimPyNode


def main():
    parser = argparse.ArgumentParser(
        description='FLIM SimPy — Fully Asynchronous Federated Learning Simulation'
    )
    parser.add_argument('--dataset', type=str, default='cifar10_example')
    parser.add_argument('--n_node', type=int, default=10)
    parser.add_argument('--sim_th', type=float, default=0.6)
    parser.add_argument('--lamb', type=int, default=6)
    parser.add_argument('--straggler', type=int, default=0)
    parser.add_argument('--mode', type=str, default='withRot')
    parser.add_argument('--net', type=str, default='vgg11')
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--sim_time', type=float, default=None, help='override SIMULATION_TIME')
    args = parser.parse_args()

    if args.sim_time is not None:
        settings.SIMULATION_TIME = args.sim_time

    # ----------------------------------------------------------
    # Paths & logger
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
        f'_simpy_async'
    )
    logger = set_logger(BASE_PATH)

    print_log(logger, format_title("FLIM SimPy — Async Simulation"))
    print_log(logger, f"Mode           : fully asynchronous (no rounds / ticks)")
    print_log(logger, f"Simulation time: {settings.SIMULATION_TIME}s (continuous)")
    print_log(logger, f"Nodes          : {args.n_node}  |  Lambda: {args.lamb}")
    print_log(logger, f"Stragglers     : {args.straggler}")
    print_log(logger, f"Network        : {args.net}  |  Sim threshold: {args.sim_th}")
    print_log(logger, "")

    # ----------------------------------------------------------
    # SimPy environment + registry
    # ----------------------------------------------------------
    env = simpy.Environment()
    registry = NodeRegistry(logger, BASE_PATH)

    # The peers dict is shared among all nodes so they can find
    # each other's inboxes.  It starts empty and is populated as
    # nodes are created.
    peers = registry.peers_dict      # same dict object

    # ----------------------------------------------------------
    # Straggler selection
    # ----------------------------------------------------------
    all_ids = [f'node{i}' for i in range(args.n_node)]
    straggler_nodes = random.sample(all_ids, min(args.straggler, args.n_node))
    print_log(logger, f"Straggler nodes: {straggler_nodes}")

    # ----------------------------------------------------------
    # Create nodes — each starts its own SimPy process immediately
    # ----------------------------------------------------------
    for idx in range(args.n_node):
        nid = f'node{idx}'
        node = SimPyNode(
            env=env,
            args=args,
            logger=logger,
            node_id=nid,
            peers=peers,             # shared reference
            base_path=BASE_PATH,
            is_straggler=(nid in straggler_nodes),
        )
        registry.register(node)      # also populates `peers`

    # ----------------------------------------------------------
    # Run — every node runs independently until SIMULATION_TIME
    # ----------------------------------------------------------
    start_time = datetime.datetime.now()
    print_log(logger, f"Execution started at: {start_time}")
    print_log(logger, "")

    env.run(until=settings.SIMULATION_TIME)

    end_time = datetime.datetime.now()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print_log(logger, "")
    print_log(logger, format_title("Simulation Complete"))

    summary = registry.get_summary()
    print_log(logger, f"Total events: {summary['total_events']}")
    for nid, cnt in summary['events_per_node'].items():
        tag = " (straggler)" if nid in straggler_nodes else ""
        print_log(logger, f"  {nid}: {cnt} events{tag}")

    print_log(logger, "")
    print_log(logger, f"Execution ended at: {end_time}")
    print_log(logger, f"Execution time: {end_time - start_time}")

    registry.generate_visualizations()
    print_log(logger, f"Timeline plots → {BASE_PATH}/events/")


if __name__ == '__main__':
    set_seed()
    main()
