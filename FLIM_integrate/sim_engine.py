import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conf import settings
from utils_system import (
    print_log, plot_node_events_timeline, plot_single_node_events
)


class NodeRegistry:
    def __init__(self, logger, base_path):
        self.logger = logger
        self.base_path = base_path
        self.nodes = {}

    def register(self, node):
        self.nodes[node.node_id] = node

    @property
    def peers_dict(self):
        return self.nodes

    # Post-simulation
    def get_summary(self):
        total_events = sum(n.event_count for n in self.nodes.values())
        return {
            'total_events': total_events,
            'events_per_node': {
                nid: n.event_count for nid, n in self.nodes.items()
            },
            'simulation_time': settings.SIMULATION_TIME,
        }

    def get_mc_summary(self):
        mc_results = {}
        for nid, node in self.nodes.items():
            mc_results[nid] = {
                'mc_triggered': node.mc_triggered,
                'mass_mapping': node.mass_mapping,
                'routing_accuracy': node.routing_accuracy,
                'moe_accuracy': node.moe_accuracy,
                'expert_count': len(node.other_models),
                'expert_pool_stable': (
                    len(node.cluster_count_history) >= settings.STABILITY_WINDOW
                    and len(set(node.cluster_count_history[-settings.STABILITY_WINDOW:])) == 1
                ) if len(node.cluster_count_history) >= settings.STABILITY_WINDOW else False,
            }
        return mc_results

    def generate_visualizations(self):
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
