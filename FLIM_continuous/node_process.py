
# Lifecycle of one event:
#   1. yield env.timeout(exponential)   ← wait for next arrival
#   2. Process inbox (cluster any models received since last event)
#   3. Train (rotation → migrate → classification)
#   4. Broadcast own model to all peers' inboxes
#   5. Save model snapshot
#   6. goto 1
#
# Nodes never synchronize with each other.
# The only shared state is each node's "inbox" list, which peer nodes append to when they broadcast.
# Because SimPy is single-threaded DES, list appends are safe without locks.
# ============================================================

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from conf import settings
from utils_system import print_log, create_directory, flatten_tuple

# Parent FLIM imports
from models.vgg_rotation import vgg11_bn as vgg_rot
from utils_learning import (
    get_network, save_model, models_to_matrix,
    aggregation, cosine_similarity_between_models
)
from data_lodaer import node_dataloader, node_rot_dataloader
from node import migrate_rot_to_cls, migrate_cls_to_rot


class SimPyNode:
    """
    Fully autonomous federated learning node.

    Each instance runs its own SimPy process. Events arrive via
    exponential inter-arrival times (continuous Poisson process).
    There is NO global clock tick or round — nodes are completely
    asynchronous with respect to each other.
    """

    def __init__(self, env, args, logger, node_id, peers, base_path,
                 is_straggler=False):
        """
        Args:
            env:            SimPy Environment
            args:           command-line arguments
            logger:         logging instance
            node_id:        unique id, e.g. 'node0'
            peers:          dict reference — will be populated after all nodes
                            are created: {node_id: SimPyNode}
            base_path:      root path for saving models / logs
            is_straggler:   if True, lower event rate
        """
        self.env = env
        self.args = args
        self.logger = logger
        self.node_id = node_id
        self.peers = peers            # shared dict, filled externally
        self.base_path = base_path
        self.is_straggler = is_straggler

        # Poisson rate (events per unit time)
        lam = settings.STRAGGLER_LAMBDA if is_straggler else args.lamb
        self.rate = lam / settings.SIMULATION_TIME   # λ / T

        # Device
        self.device = torch.device(
            'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        )

        # Models
        self.model = get_network(args).to(torch.device('cpu'))
        self.rot_model = vgg_rot().to(torch.device('cpu'))
        self.other_models = {}          # expert pool (clustered)
        self.loss_function = nn.CrossEntropyLoss()

        # Data
        self.train_loader, self.test_loader = node_dataloader(args, node_id)
        self.rot_train_loader, self.rot_test_loader = node_rot_dataloader(args, node_id)

        # State
        self.recv_status = True         # accepting models?
        self.total_epoch = 0
        self.event_count = 0            # how many events this node has fired
        self.event_times = []           # timestamps of all events

        # ---- Async inbox ----
        # Other nodes append (sender_id, model) here at any time.
        # This node drains the inbox at the START of its own event.
        self.inbox = []                 # [(sender_id, model), ...]

        # Start the autonomous process
        self.process = env.process(self._run())

    # ==========================================================
    # SimPy Process — fully autonomous loop
    # ==========================================================

    def _run(self):
        """Main loop. No ticks, no rounds — just Poisson arrivals."""
        while True:
            # 1) Wait for next event (exponential inter-arrival)
            inter_arrival = np.random.exponential(1.0 / self.rate)
            yield self.env.timeout(inter_arrival)

            if self.env.now >= settings.SIMULATION_TIME:
                break

            self.event_count += 1
            self.event_times.append(self.env.now)

            print_log(
                self.logger,
                f"[t={self.env.now:8.4f}] {self.node_id} event #{self.event_count}"
            )

            # 2) Process inbox — cluster models received since last event
            self._process_inbox()

            # 3) Train
            self._train_step()

            # 4) Broadcast to all peers
            self._broadcast()

            # 5) Save model snapshot
            save_model(
                self.base_path,
                f"t{self.env.now:.4f}",   # continuous timestamp as id
                self.node_id,
                self.model,
                'model0'
            )

            print_log(self.logger, "")

    # ==========================================================
    # Inbox Processing (async model clustering)
    # ==========================================================

    def _process_inbox(self):
        """
        Drain the inbox and cluster received models.
        Called at the beginning of each node event so that any models
        that arrived asynchronously from other nodes are incorporated.
        """
        if not self.inbox or not self.recv_status:
            if self.inbox and not self.recv_status:
                print_log(self.logger,
                          f"  {self.node_id} inbox has {len(self.inbox)} model(s) "
                          f"but recv_status=False, discarding")
                self.inbox.clear()
            return

        # Drain
        received = {}
        for sender_id, model in self.inbox:
            received[sender_id] = model      # last-write-wins if duplicate
        self.inbox.clear()

        print_log(self.logger,
                  f"  {self.node_id} processing inbox: "
                  f"{list(received.keys())}")

        # ---- Phase 1: similarity filter vs own model ----
        similar_models = [self.model]
        similar_ids = []
        non_similar = {}

        for sid, rmodel in received.items():
            sim = cosine_similarity_between_models(self.model, rmodel)
            dist = max(1 - sim, 0)

            if dist < self.args.sim_th:
                similar_models.append(rmodel)
                similar_ids.append(sid)
            else:
                non_similar[sid] = rmodel

        if len(similar_models) > 1:
            print_log(self.logger,
                      f"  Aggregating similar: [{', '.join(similar_ids)}]")
            self.model = aggregation(self.args, similar_models)

        # ---- Phase 2: DBSCAN clustering of non-similar + existing experts ----
        if non_similar:
            all_ids = list(self.other_models.keys()) + list(non_similar.keys())
            all_models = list(self.other_models.values()) + list(non_similar.values())

            mat = models_to_matrix(all_models)
            sim_mat = sklearn_cosine_similarity(mat)
            dist_mat = np.maximum(1 - sim_mat, 0)
            db = DBSCAN(
                eps=self.args.sim_th, min_samples=1, metric='precomputed'
            ).fit(dist_mat)

            clustered = {}
            for order, label in enumerate(np.unique(db.labels_), start=1):
                idxs = np.where(db.labels_ == label)[0].tolist()
                c_ids = [all_ids[i] for i in idxs]
                c_models = [all_models[i] for i in idxs]
                flat = flatten_tuple(c_ids)
                print_log(self.logger, f"    Cluster {order}: {flat}")

                agg = aggregation(self.args, c_models)
                save_model(
                    self.base_path,
                    f"t{self.env.now:.4f}",
                    self.node_id, agg, f"model{order}"
                )
                clustered[tuple(c_ids)] = agg

            self.other_models = clustered

        # Capacity check
        if len(self.other_models) > settings.SUP_OTHER_MODEL_SIZE:
            self.recv_status = False
            print_log(self.logger,
                      f"  {self.node_id} reached expert capacity, recv_status=False")

    # ==========================================================
    # Broadcasting — push model to every peer's inbox
    # ==========================================================

    def _broadcast(self):
        """Send own model to every peer that is still accepting."""
        sent_to = []
        for pid, peer in self.peers.items():
            if pid == self.node_id:
                continue
            if peer.recv_status:
                peer.inbox.append((self.node_id, self.model))
                sent_to.append(pid)

        print_log(self.logger, f"  Broadcast -> {sent_to}")

    # ==========================================================
    # Training (identical logic to original FLIM)
    # ==========================================================

    def _train_step(self):
        """Full pipeline: rotation → feature migration → classification."""
        self.train_rot()
        self.evaluate_rot()
        self.model = migrate_rot_to_cls(self.rot_model, self.model)
        self.train_cls()

    def train_cls(self):
        """Classification training."""
        self.model.to(self.device)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=settings.LEARNING_RATE)

        self.total_epoch += settings.CLS_EPOCH
        print_log(self.logger,
                  f"  CLS train | epoch {settings.CLS_EPOCH} | "
                  f"total {self.total_epoch}")

        avg_loss = 0.0
        for _ in range(settings.CLS_EPOCH):
            ep_loss = 0.0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, targets)
                ep_loss += loss.item() / len(self.train_loader.dataset)
                loss.backward()
                optimizer.step()
            avg_loss += ep_loss

        print_log(self.logger, f"  CLS avg loss: {avg_loss / settings.CLS_EPOCH:.4f}")
        self.model.to(torch.device('cpu'))

    @torch.no_grad()
    def evaluate_cls(self):
        """Classification evaluation."""
        self.model.to(self.device)
        self.model.eval()
        correct = 0.0
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            _, preds = self.model(inputs).max(1)
            correct += preds.eq(targets).sum()
        acc = correct.float() * 100 / len(self.test_loader.dataset)
        print_log(self.logger, f"  CLS accuracy: {acc:.2f}%")
        self.model.to(torch.device('cpu'))
        return acc.item()

    def train_rot(self):
        """Rotation pretext task training."""
        self.rot_model.to(self.device)
        self.rot_model.train()
        optimizer = optim.Adam(self.rot_model.parameters(), lr=settings.LEARNING_RATE)

        print_log(self.logger, f"  ROT train | epoch {settings.ROT_EPOCH}")

        avg_loss = 0.0
        for _ in range(settings.ROT_EPOCH):
            ep_loss = 0.0
            for inputs, targets in self.rot_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.rot_model(inputs)
                loss = self.loss_function(outputs, targets)
                ep_loss += loss.item() / len(self.rot_train_loader.dataset)
                loss.backward()
                optimizer.step()
            avg_loss += ep_loss

        print_log(self.logger, f"  ROT avg loss: {avg_loss / settings.ROT_EPOCH:.4f}")
        self.rot_model.to(torch.device('cpu'))

    @torch.no_grad()
    def evaluate_rot(self):
        """Rotation evaluation."""
        self.rot_model.to(self.device)
        self.rot_model.eval()
        correct = 0.0
        for inputs, targets in self.rot_test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            _, preds = self.rot_model(inputs).max(1)
            correct += preds.eq(targets).sum()
        acc = correct.float() * 100 / len(self.rot_test_loader.dataset)
        print_log(self.logger, f"  ROT accuracy: {acc:.2f}%")
        self.rot_model.to(torch.device('cpu'))
        return acc.item()

    @torch.no_grad()
    def source_evaluate(self):
        """Per-class accuracy evaluation."""
        class_correct = [0.0] * len(settings.LABELS)
        class_total = [0.0] * len(settings.LABELS)

        self.model.to(self.device)
        self.model.eval()
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            _, predicted = torch.max(self.model(inputs), 1)
            c = (predicted == targets).squeeze()
            for i in range(len(targets)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        result = [
            (settings.LABELS[i], 100 * class_correct[i] / class_total[i])
            for i in range(len(settings.LABELS))
        ]
        print_log(self.logger, f"  Per-class: {result}")
        self.model.to(torch.device('cpu'))
