# FLIM_Simpy - Node Process
# Each node is a SimPy process that generates events via exponential inter-arrival times (continuous Poisson process).
# Training, broadcasting, and clustering follow the original FLIM logic.

import sys
import os

# Add parent FLIM directory to import original modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import simpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from conf import settings
from utils_system import print_log, create_directory, flatten_tuple

# Import from parent FLIM
from models.vgg_rotation import vgg11_bn as vgg_rot
from utils_learning import (
    get_network, save_model, models_to_matrix,
    aggregation, cosine_similarity_between_models
)
from data_lodaer import node_dataloader, node_rot_dataloader
from node import migrate_rot_to_cls, migrate_cls_to_rot


class SimPyNode:
    """
    A federated learning node that operates as a SimPy process.

    Each node independently generates participation events following
    an exponential inter-arrival time distribution (Poisson process).
    When an event fires, the node trains its model, broadcasts to peers,
    and receivers perform model clustering.
    """

    def __init__(self, env, args, logger, node_id, coordinator, is_straggler=False):
        """
        Args:
            env: SimPy Environment
            args: command-line arguments
            logger: logging instance
            node_id: unique identifier (e.g., 'node0')
            coordinator: SimulationCoordinator instance for peer communication
            is_straggler: if True, uses lower event rate
        """
        self.env = env
        self.args = args
        self.logger = logger
        self.node_id = node_id
        self.coordinator = coordinator
        self.is_straggler = is_straggler

        # Poisson rate: lambda / SIMULATION_TIME gives per-unit-time rate
        # mean inter-arrival = SIMULATION_TIME / lambda
        if is_straggler:
            self.rate = settings.STRAGGLER_LAMBDA / settings.SIMULATION_TIME
        else:
            self.rate = args.lamb / settings.SIMULATION_TIME

        # Device
        self.device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')

        # Models
        self.model = get_network(args).to(torch.device('cpu'))
        self.rot_model = vgg_rot().to(torch.device('cpu'))
        self.other_models = {}
        self.loss_function = nn.CrossEntropyLoss()

        # Data loaders
        self.train_loader, self.test_loader = node_dataloader(self.args, self.node_id)
        self.rot_train_loader, self.rot_test_loader = node_rot_dataloader(self.args, self.node_id)

        # State
        self.recv_status = True
        self.total_epoch = 0
        self.event_times = []           # Record of all event timestamps
        self.inbox = []                 # Models received from other nodes: [(sender_id, model)]

        # Start the SimPy process
        self.process = env.process(self.run())

    # ============================================================
    # SimPy Process - Main event loop
    # ============================================================

    def run(self):
        """
        Main SimPy process generator.
        Generates events via exponential inter-arrival times and
        delegates to the coordinator for synchronized processing.
        """
        while True:
            # Sample inter-arrival time from exponential distribution
            inter_arrival = np.random.exponential(1.0 / self.rate) if self.rate > 0 else float('inf')
            yield self.env.timeout(inter_arrival)

            # Check if simulation time exceeded
            if self.env.now >= settings.SIMULATION_TIME:
                break

            # Record event
            self.event_times.append(self.env.now)

            # Notify coordinator that this node has an event
            self.coordinator.register_event(self.env.now, self)

    # ============================================================
    # Training Methods (same logic as original FLIM)
    # ============================================================

    def train(self):
        """Classification model training."""
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=settings.LEARNING_RATE)

        self.total_epoch += settings.CLS_EPOCH
        print_log(self.logger,
                  f"Classification - Training epoch: {settings.CLS_EPOCH} | Total epoch: {self.total_epoch}")

        avg_train_loss = 0.0
        for _ in range(settings.CLS_EPOCH):
            train_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                train_loss += (loss.item() / len(self.train_loader.dataset))
                loss.backward()
                optimizer.step()
            avg_train_loss += train_loss

        print_log(self.logger, f"Avg train loss: {(avg_train_loss / settings.CLS_EPOCH):.2f}")
        self.model.to(torch.device('cpu'))

    @torch.no_grad()
    def evaluate(self):
        """Classification model evaluation."""
        self.model.to(self.device)
        self.model.eval()
        correct = 0.0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicts = outputs.max(1)
            correct += predicts.eq(targets).sum()

        acc = correct.float() * 100 / len(self.test_loader.dataset)
        print_log(self.logger, f"Accuracy: {acc:.2f}")
        self.model.to(torch.device('cpu'))
        return acc.item()

    def train_rot(self):
        """Rotation pretext task training."""
        self.rot_model.to(self.device)
        self.rot_model.train()
        optimizer = optim.Adam(self.rot_model.parameters(), lr=settings.LEARNING_RATE)

        print_log(self.logger, f"Rotation pretext task - Training epoch: {settings.ROT_EPOCH}")

        avg_train_loss = 0.0
        for _ in range(settings.ROT_EPOCH):
            train_loss = 0.0
            for inputs, targets in self.rot_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.rot_model(inputs)
                loss = self.loss_function(outputs, targets)
                train_loss += (loss.item() / len(self.rot_train_loader.dataset))
                loss.backward()
                optimizer.step()
            avg_train_loss += train_loss

        print_log(self.logger, f"Avg train loss: {(avg_train_loss / settings.ROT_EPOCH):.2f}")
        self.rot_model.to(torch.device('cpu'))

    @torch.no_grad()
    def evaluate_rot(self):
        """Rotation pretext task evaluation."""
        self.rot_model.to(self.device)
        self.rot_model.eval()
        correct = 0.0

        for inputs, targets in self.rot_test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.rot_model(inputs)
            _, predicts = outputs.max(1)
            correct += predicts.eq(targets).sum()

        acc = correct.float() * 100 / len(self.rot_test_loader.dataset)
        print_log(self.logger, f"Rotation Accuracy: {acc:.2f}")
        self.rot_model.to(torch.device('cpu'))
        return acc.item()

    @torch.no_grad()
    def source_evaluate(self):
        """Per-class accuracy evaluation."""
        class_correct = list(0. for _ in range(len(settings.LABELS)))
        class_total = list(0. for _ in range(len(settings.LABELS)))

        self.model.to(self.device)
        self.model.eval()

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()

            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        accuracy_per_class = [
            (settings.LABELS[i], 100 * class_correct[i] / class_total[i])
            for i in range(len(settings.LABELS))
        ]
        print_log(self.logger, accuracy_per_class)
        self.model.to(torch.device('cpu'))

    # ============================================================
    # Training Pipeline (same sequence as original)
    # ============================================================

    def train_step(self):
        """Full training pipeline: rotation → migrate → classification."""
        print_log(self.logger, f"[ {self.node_id} ] @ t={self.env.now:.4f}")
        self.train_rot()
        self.evaluate_rot()
        self.model = migrate_rot_to_cls(self.rot_model, self.model)
        self.train()
        print_log(self.logger, "")

    # ============================================================
    # Model Clustering (same logic as original FLIM)
    # ============================================================

    def model_clustering(self, base_path, round_id, received_models):
        """
        Two-phase clustering of received models.

        Args:
            base_path: path for saving models
            round_id: current round identifier
            received_models: dict {sender_node_id: model}
        """
        print_log(self.logger, f"[ {self.node_id} ] Clustering")

        # Phase 1: Compare received models with own model
        if received_models:
            similar_models = [self.model]
            similar_node_ids = []
            non_similar_models = {}

            for sender_id, recv_model in received_models.items():
                sim = cosine_similarity_between_models(self.model, recv_model)
                distance = np.maximum(1 - sim, 0)

                if distance < self.args.sim_th:
                    similar_models.append(recv_model)
                    similar_node_ids.append(sender_id)
                else:
                    non_similar_models[sender_id] = recv_model

            received_models = non_similar_models

            # Aggregate similar models
            if len(similar_models) > 1:
                print_log(self.logger,
                          f"{self.node_id} model is similar to: [{', '.join(similar_node_ids)}]")
                self.model = aggregation(self.args, similar_models)

        # Phase 2: Cluster non-similar models with existing expert models
        if received_models:
            all_node_ids = list(self.other_models.keys()) + list(received_models.keys())
            all_models = list(self.other_models.values()) + list(received_models.values())

            model_matrix = models_to_matrix(all_models)
            sim_matrix = sklearn_cosine_similarity(model_matrix)
            distance_matrix = np.maximum(1 - sim_matrix, 0)
            db = DBSCAN(eps=self.args.sim_th, min_samples=1, metric='precomputed').fit(distance_matrix)

            clustered_data = {}
            for save_order, cluster_label in enumerate(np.unique(db.labels_), start=1):
                indices = np.where(db.labels_ == cluster_label)[0].tolist()
                clustered_node_ids = [all_node_ids[idx] for idx in indices]
                clustered_models = [all_models[idx] for idx in indices]

                flat_ids = flatten_tuple(clustered_node_ids)
                print_log(self.logger, f"  Cluster {save_order}: {flat_ids}")

                agg_model = aggregation(self.args, clustered_models)
                save_model(base_path, round_id, self.node_id, agg_model, f"model{save_order}")
                clustered_data[tuple(clustered_node_ids)] = agg_model

            self.other_models = clustered_data

        # Check capacity
        if len(self.other_models) > settings.SUP_OTHER_MODEL_SIZE:
            self.recv_status = False

        print_log(self.logger, "")
