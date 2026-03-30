import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import simpy
import numpy as np
import shutil
import datetime
import random

from conf import settings
from utils_system import (
    print_log, format_title, format_time_title, create_directory,
    plot_node_events_timeline, plot_single_node_events, copy_prior_time_slot
)
from node_process import SimPyNode
from utils_learning import save_model


class SimulationCoordinator:
    """
    Central coordinator for the SimPy-based FLIM simulation.

    Design:
        Instead of discrete time slots, the simulation runs in continuous time.
        Each node independently generates Poisson events. When events occur,
        the coordinator collects them and processes them in "rounds".

        A round is triggered each time a batch of near-simultaneous events
        (within EVENT_BATCH_WINDOW) is collected. The processing sequence
        mirrors the original FLIM:
            1. Training Step   - participating nodes train their models
            2. Broadcasting    - participants broadcast models to all peers
            3. Clustering      - all nodes cluster the received models
            4. Model Saving    - all nodes save current model state
    """

    def __init__(self, env, args, logger, base_path):
        self.env = env
        self.args = args
        self.logger = logger
        self.base_path = base_path

        self.nodes = {}          # {node_id: SimPyNode}
        self.all_nodes_list = [] # ordered list of SimPyNode
        self.pending_events = [] # [(time, SimPyNode)] events waiting to be processed
        self.round_id = 0        # incremental round counter
        self.event_log = []      # full event history for visualization

        # Create the event processor as a SimPy process
        self.process = env.process(self._event_processor())

    def register_node(self, node):
        """Register a node with the coordinator."""
        self.nodes[node.node_id] = node
        self.all_nodes_list.append(node)

    def register_event(self, time, node):
        """Called by nodes when an event fires. Queues for batch processing."""
        self.pending_events.append((time, node))
        self.event_log.append((time, node.node_id))

    # ============================================================
    # Event Processor - SimPy process that batches and processes events
    # ============================================================

    def _event_processor(self):
        """
        SimPy process that periodically checks for pending events
        and processes them in batches (rounds).

        Uses a small polling interval to collect near-simultaneous events.
        """
        while self.env.now < settings.SIMULATION_TIME:
            # Wait a small interval to collect batched events
            yield self.env.timeout(settings.EVENT_BATCH_WINDOW)

            if not self.pending_events:
                continue

            # Collect all pending events into this round
            batch = list(self.pending_events)
            self.pending_events.clear()

            # Deduplicate: a node may fire multiple events in one batch
            participating_nodes = list({node.node_id: node for _, node in batch}.values())
            event_times = [t for t, _ in batch]

            self._process_round(participating_nodes, event_times)

    # ============================================================
    # Round Processing (mirrors original FLIM main loop)
    # ============================================================

    def _process_round(self, participating_nodes, event_times):
        """
        Process one simulation round with the given participating nodes.
        Follows the same sequence as the original FLIM time slot processing.
        """
        self.round_id += 1
        t_min = min(event_times)
        t_max = max(event_times)

        print_log(self.logger,
                  format_time_title(f"Round {self.round_id} | t=[{t_min:.4f}, {t_max:.4f}]"))

        # Check exceeding nodes
        exceeding_nodes = [n.node_id for n in self.all_nodes_list if not n.recv_status]
        if exceeding_nodes:
            print_log(self.logger, f"Exceeding nodes: {exceeding_nodes}")

        if len(exceeding_nodes) == len(self.all_nodes_list):
            print_log(self.logger,
                      f"All nodes exceeded SUP_OTHER_MODEL_SIZE ({settings.SUP_OTHER_MODEL_SIZE})!")
            return

        part_ids = [n.node_id for n in participating_nodes]
        print_log(self.logger, f"Participating nodes: {part_ids}")

        # ----------------------------------------------------------
        # 1. Training Step
        # ----------------------------------------------------------
        print_log(self.logger, format_title("Training Step"))
        for node in participating_nodes:
            node.train_step()

        # ----------------------------------------------------------
        # 2. Broadcasting Step
        # ----------------------------------------------------------
        print_log(self.logger, format_title("Broadcasting Step"))
        broadcasts = self._compute_broadcasts(participating_nodes)

        for sender_id, receiver_ids in broadcasts.items():
            print_log(self.logger, f"{sender_id} -> {receiver_ids}")
        print_log(self.logger, "")

        # Compute received_from mapping
        received_from = self._compute_received_from(broadcasts)

        print_log(self.logger, format_title("Received From"))
        for node in self.all_nodes_list:
            print_log(self.logger, f"{node.node_id}: {received_from[node.node_id]}")
        print_log(self.logger, "")

        # ----------------------------------------------------------
        # 3. Model Clustering Step
        # ----------------------------------------------------------
        print_log(self.logger, format_title("Received Model Clustering"))

        part_models = {n.node_id: n.model for n in participating_nodes}

        for node in self.all_nodes_list:
            recv_models = {
                sender_id: part_models[sender_id]
                for sender_id in received_from[node.node_id]
                if sender_id in part_models
            }
            node.model_clustering(self.base_path, self.round_id, recv_models)

        print_log(self.logger, "")

        # ----------------------------------------------------------
        # 4. Save all models
        # ----------------------------------------------------------
        for node in self.all_nodes_list:
            save_model(self.base_path, self.round_id, node.node_id, node.model, 'model0')

    # ============================================================
    # Broadcasting Logic (same as original FLIM)
    # ============================================================

    def _compute_broadcasts(self, participating_nodes):
        """
        Determine broadcast targets: each participating node sends
        its model to all other nodes that are still accepting models.
        """
        broadcasts = {}
        for bcast_node in participating_nodes:
            receivers = [
                n.node_id for n in self.all_nodes_list
                if n.recv_status and n != bcast_node
            ]
            broadcasts[bcast_node.node_id] = receivers
        return broadcasts

    def _compute_received_from(self, broadcasts):
        """Invert the broadcast mapping to get received_from for each node."""
        received_from = {n.node_id: [] for n in self.all_nodes_list}
        for sender_id, receiver_ids in broadcasts.items():
            for recv_id in receiver_ids:
                received_from[recv_id].append(sender_id)
        return received_from

    # ============================================================
    # Post-simulation Analysis
    # ============================================================

    def generate_visualizations(self):
        """Generate event timeline plots after simulation completes."""
        node_events_dict = {}
        for node in self.all_nodes_list:
            node_events_dict[node.node_id] = node.event_times
            plot_single_node_events(
                self.base_path, node.node_id,
                node.event_times, settings.SIMULATION_TIME
            )

        plot_node_events_timeline(self.base_path, node_events_dict, settings.SIMULATION_TIME)

    def get_summary(self):
        """Return a summary of the simulation."""
        summary = {
            'total_rounds': self.round_id,
            'total_events': len(self.event_log),
            'events_per_node': {
                n.node_id: len(n.event_times) for n in self.all_nodes_list
            },
            'simulation_time': settings.SIMULATION_TIME,
        }
        return summary
