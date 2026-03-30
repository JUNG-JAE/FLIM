# 1. Holds references to all nodes (peer registry)
# 2. Tracks event history for post-simulation analysis
# 3. Generates visualizations after the simulation ends

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conf import settings
from utils_system import (
    print_log, plot_node_events_timeline, plot_single_node_events
)


class NodeRegistry:
    """
    Lightweight registry that holds references to all SimPyNodes
    and provides post-simulation analysis / visualization.

    This is NOT a coordinator — nodes do not interact through it.
    Nodes interact directly by appending to each other's inboxes.
    """

    def __init__(self, logger, base_path):
        self.logger = logger
        self.base_path = base_path
        self.nodes = {}          # {node_id: SimPyNode}

    def register(self, node):
        """Add a node to the registry."""
        self.nodes[node.node_id] = node

    @property
    def peers_dict(self):
        """Return the shared peers dict for nodes to reference each other."""
        return self.nodes

    # ==========================================================
    # Post-simulation
    # ==========================================================

    def get_summary(self):
        """Compute simulation summary after env.run() completes."""
        total_events = sum(n.event_count for n in self.nodes.values())
        return {
            'total_events': total_events,
            'events_per_node': {
                nid: n.event_count for nid, n in self.nodes.items()
            },
            'simulation_time': settings.SIMULATION_TIME,
        }

    def generate_visualizations(self):
        """Generate event timeline plots."""
        node_events_dict = {}
        for node in self.nodes.values():
            node_events_dict[node.node_id] = node.event_times
            plot_single_node_events(
                self.base_path, node.node_id,
                node.event_times, settings.SIMULATION_TIME
            )

        plot_node_events_timeline(
            self.base_path, node_events_dict, settings.SIMULATION_TIME
        )
