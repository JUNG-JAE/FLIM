import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.join(_HERE, '..')
for p in (_PARENT, _HERE):
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
from node_process import IntegratedSimPyNode


def main():
    parser = argparse.ArgumentParser(description='FLIM_integrate — Async DFL + MASS + GateModule')
    parser.add_argument('--dataset', type=str, default='cifar10_example')
    parser.add_argument('--n_node', type=int, default=10)
    parser.add_argument('--sim_th', type=float, default=0.6, help='Cosine distance threshold for expert aggregation (τ)')
    parser.add_argument('--lamb', type=int, default=6, help='Poisson lambda (events per simulation time)')
    parser.add_argument('--straggler', type=int, default=0, help='Number of straggler nodes')
    parser.add_argument('--mode', type=str, default='withRot')
    parser.add_argument('--net', type=str, default='vgg11')
    parser.add_argument('--gpu', action='store_true', default=True)
    parser.add_argument('--sim_time', type=float, default=None, help='Override SIMULATION_TIME')
    parser.add_argument('--mass_trials', type=int, default=None, help='Override MASS Bayesian optimization trials')
    parser.add_argument('--stable_count', type=int, default=None, help='Override EXPERT_STABLE_COUNT')
    parser.add_argument('--stable_window', type=int, default=None, help='Override STABILITY_WINDOW')
    args = parser.parse_args()

    # Override settings if provided
    if args.sim_time is not None:
        settings.SIMULATION_TIME = args.sim_time
    if args.mass_trials is not None:
        settings.MASS_N_TRIALS = args.mass_trials
    if args.stable_count is not None:
        settings.EXPERT_STABLE_COUNT = args.stable_count
    if args.stable_window is not None:
        settings.STABILITY_WINDOW = args.stable_window

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
        f'_integrate'
    )
    logger = set_logger(BASE_PATH)

    print_log(logger, format_title("FLIM_integrate — Async DFL + MASS + GateModule"))
    print_log(logger, f"Mode             : fully asynchronous (SimPy continuous)")
    print_log(logger, f"Simulation time  : {settings.SIMULATION_TIME}s")
    print_log(logger, f"Nodes            : {args.n_node}  |  Lambda: {args.lamb}")
    print_log(logger, f"Stragglers       : {args.straggler}")
    print_log(logger, f"Network          : {args.net}  |  Sim threshold: {args.sim_th}")
    print_log(logger, f"Expert capacity  : {settings.SUP_OTHER_MODEL_SIZE}")
    print_log(logger, f"Stability trigger: count>={settings.EXPERT_STABLE_COUNT}, "
                       f"window={settings.STABILITY_WINDOW}")
    print_log(logger, f"MASS trials      : {settings.MASS_N_TRIALS}")
    print_log(logger, f"CLA params       : ξ={settings.CLA_TAU_GAP}, δ={settings.CLA_DELTA}")
    print_log(logger, "")

    # ----------------------------------------------------------
    # SimPy environment + registry
    # ----------------------------------------------------------
    env = simpy.Environment()
    registry = NodeRegistry(logger, BASE_PATH)
    peers = registry.peers_dict

    # ----------------------------------------------------------
    # Straggler selection
    # ----------------------------------------------------------
    all_ids = [f'node{i}' for i in range(args.n_node)]
    straggler_nodes = random.sample(all_ids, min(args.straggler, args.n_node))
    print_log(logger, f"Straggler nodes: {straggler_nodes}")

    # ----------------------------------------------------------
    # Create nodes
    # ----------------------------------------------------------
    for idx in range(args.n_node):
        nid = f'node{idx}'
        node = IntegratedSimPyNode(
            env=env,
            args=args,
            logger=logger,
            node_id=nid,
            peers=peers,
            base_path=BASE_PATH,
            is_straggler=(nid in straggler_nodes),
        )
        registry.register(node)

    # ----------------------------------------------------------
    # Run simulation
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

    # Event summary
    summary = registry.get_summary()
    print_log(logger, f"Total events: {summary['total_events']}")
    for nid, cnt in summary['events_per_node'].items():
        tag = " (straggler)" if nid in straggler_nodes else ""
        print_log(logger, f"  {nid}: {cnt} events{tag}")

    # MC (MASS + GateModule) summary
    print_log(logger, "")
    print_log(logger, format_title("MC Process Results"))
    mc_summary = registry.get_mc_summary()
    for nid, info in mc_summary.items():
        print_log(logger, f"  {nid}:")
        print_log(logger, f"    MC triggered   : {info['mc_triggered']}")
        print_log(logger, f"    Expert count   : {info['expert_count']}")
        print_log(logger, f"    Pool stable    : {info['expert_pool_stable']}")
        if info['mc_triggered']:
            print_log(logger, f"    MASS mapping   : {info['mass_mapping']}")
            print_log(logger, f"    Routing acc    : {info['routing_accuracy']:.2f}%" if info['routing_accuracy'] else "    Routing acc    : N/A")
            print_log(logger, f"    MoE acc        : {info['moe_accuracy']:.2f}%" if info['moe_accuracy'] else "    MoE acc        : N/A")

    print_log(logger, "")
    print_log(logger, f"Execution ended at: {end_time}")
    print_log(logger, f"Execution time: {end_time - start_time}")

    # Visualizations
    registry.generate_visualizations()
    print_log(logger, f"Timeline plots -> {BASE_PATH}/events/")


if __name__ == '__main__':
    set_seed()
    main()
