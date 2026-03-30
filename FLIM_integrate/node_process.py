import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simpy
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from conf import settings
from utils_system import print_log, create_directory, flatten_tuple
from utils_learning import (
    get_network, save_model, models_to_matrix,
    aggregation, cosine_similarity_between_models,
    cosine_distance_between_models,
    migrate_rot_to_cls, migrate_cls_to_rot,
    correlation_matrix_aggregation,
)
from data_lodaer import node_dataloader, node_rot_dataloader

from models.vgg_rotation import vgg11_bn as vgg_rot

from mass_engine import (
    cluster_unlabeled_data,
    split_cluster_train_test,
    run_mass_pipeline,
)
from gate_engine import (
    build_gating_module,
    prepare_gating_data,
    train_gating_module,
    moe_inference,
)


class IntegratedSimPyNode:

    def __init__(self, env, args, logger, node_id, peers, base_path,
                 is_straggler=False):
        self.env = env
        self.args = args
        self.logger = logger
        self.node_id = node_id
        self.peers = peers
        self.base_path = base_path
        self.is_straggler = is_straggler

        # Poisson rate
        lam = settings.STRAGGLER_LAMBDA if is_straggler else args.lamb
        self.rate = lam / settings.SIMULATION_TIME

        # Device
        self.device = torch.device(
            'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
        )

        # --- DE process models ---
        self.model = get_network(args).to(torch.device('cpu'))
        self.rot_model = vgg_rot().to(torch.device('cpu'))
        self.other_models = {}          # expert pool: {expert_id: model}
        self.loss_function = nn.CrossEntropyLoss()

        # --- Data ---
        self.train_loader, self.test_loader = node_dataloader(args, node_id)
        self.rot_train_loader, self.rot_test_loader = node_rot_dataloader(args, node_id)

        # --- State ---
        self.recv_status = True
        self.total_epoch = 0
        self.event_count = 0
        self.event_times = []
        self.inbox = []

        # --- Stability tracking for MC trigger ---
        self.cluster_count_history = []  # track expert cluster count over events
        self.mc_triggered = False        # MASS + Gate has been run
        self.mass_mapping = None         # {cluster_id: expert_id}
        self.gating_model = None         # trained gating module
        self.expert_order = None         # ordered list of expert_ids for gating
        self.routing_accuracy = None
        self.moe_accuracy = None

        # Start SimPy process
        self.process = env.process(self._run())

    # SimPy Process — fully autonomous loop
    def _run(self):
        """Main loop: Poisson arrivals, no ticks/rounds."""
        while True:
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

            # 1) Process inbox (expert aggregation)
            self._process_inbox()

            # 2) Train (DE process)
            self._train_step()

            # 3) Broadcast
            self._broadcast()

            # 4) Check stability and trigger MC process
            self._check_and_run_mc()

            # 5) Save snapshot
            save_model(
                self.base_path,
                f"t{self.env.now:.4f}",
                self.node_id,
                self.model,
                'model0'
            )

            print_log(self.logger, "")

    # Inbox Processing (Algorithm 1: Expert Aggregation)
    def _process_inbox(self):
        """
        Drain inbox and aggregate received models using Algorithm 1.

        Phase 1: Similar models (distance < threshold) -> aggregate with own model
        Phase 2: Non-similar models -> DBSCAN clustering + correlation matrix aggregation
        """
        if not self.inbox or not self.recv_status:
            if self.inbox and not self.recv_status:
                print_log(self.logger,
                          f"  {self.node_id} inbox has {len(self.inbox)} model(s) "
                          f"but recv_status=False, discarding")
                self.inbox.clear()
            return

        # Drain inbox
        received = {}
        for sender_id, model in self.inbox:
            received[sender_id] = model
        self.inbox.clear()

        print_log(self.logger,
                  f"  {self.node_id} processing inbox: {list(received.keys())}")

        # Phase 1: similarity filter vs own model
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

        # Phase 2: DBSCAN clustering of non-similar + existing experts
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

        # Track cluster count for stability
        current_count = len(self.other_models)
        self.cluster_count_history.append(current_count)

        # Capacity check
        if current_count > settings.SUP_OTHER_MODEL_SIZE:
            self.recv_status = False
            print_log(self.logger,
                      f"  {self.node_id} reached expert capacity ({current_count}), "
                      f"recv_status=False")

        print_log(self.logger,
                  f"  Expert pool size: {current_count} "
                  f"(history: {self.cluster_count_history[-5:]})")

    # Stability Check & MC Process Trigger
    def _check_and_run_mc(self):
        if self.mc_triggered:
            return

        if len(self.other_models) < settings.EXPERT_STABLE_COUNT:
            return

        history = self.cluster_count_history
        window = settings.STABILITY_WINDOW

        if len(history) < window:
            return

        # Check if last `window` counts are all the same
        recent = history[-window:]
        if len(set(recent)) != 1:
            return

        # Stability confirmed!
        print_log(self.logger,
                  f"\n  *** {self.node_id}: Expert pool STABLE "
                  f"(count={recent[0]}, window={window}) ***")
        print_log(self.logger,
                  f"  *** Triggering MC process (MASS + GateModule) ***\n")

        self._run_mc_process()

    def _run_mc_process(self):
        self.mc_triggered = True

        # Flatten expert pool: tuple keys -> simple string ids
        experts = {}
        idx = 0
        for key, model in self.other_models.items():
            eid = f"expert{idx}"
            experts[eid] = model
            idx += 1

        # Also include own model as an expert
        experts[f"expert{idx}"] = self.model

        n_experts = len(experts)
        n_classes = len(settings.LABELS)

        print_log(self.logger,
                  f"  [MC] Running with {n_experts} experts, {n_classes} classes")

        # Use own unlabeled data (train_loader acts as unlabeled for user node)
        # In the paper, user nodes only have unlabeled data (X)
        unlabeled_loader = self.train_loader

        try:
            # Step 1: MASS pipeline
            mapping, expert_acc = run_mass_pipeline(
                experts, unlabeled_loader, n_classes, self.device,
                self.logger, n_trials=settings.MASS_N_TRIALS
            )

            self.mass_mapping = mapping
            self.expert_order = sorted(experts.keys())

            # Step 2: Cluster data for GateModule
            from mass_engine import cluster_unlabeled_data, split_cluster_train_test

            cluster_labels, features_all, all_images = cluster_unlabeled_data(
                experts, unlabeled_loader, n_classes, self.device
            )
            cluster_data = split_cluster_train_test(
                cluster_labels, features_all, all_images
            )

            # Step 3: Build and train GateModule
            self.gating_model = build_gating_module(
                experts, self.expert_order, self.device
            )

            train_imgs, train_labels, test_imgs, test_labels = prepare_gating_data(
                cluster_data, mapping, self.expert_order
            )

            if train_imgs is not None:
                self.routing_accuracy = train_gating_module(
                    self.gating_model, train_imgs, train_labels,
                    test_imgs, test_labels, self.device, self.logger
                )

                print_log(self.logger,
                          f"  [MC] Routing accuracy: {self.routing_accuracy:.2f}%")

                # Step 4: MoE inference evaluation
                self.moe_accuracy = moe_inference(
                    self.gating_model, experts, self.expert_order,
                    self.test_loader, self.device
                )

                print_log(self.logger,
                          f"  [MC] MoE classification accuracy: {self.moe_accuracy:.2f}%")

                # Save gating model
                gate_save_path = f"{self.base_path}/t{self.env.now:.4f}/{self.node_id}"
                create_directory(gate_save_path)
                torch.save(self.gating_model.state_dict(),
                           f"{gate_save_path}/gating_module.pt")
            else:
                print_log(self.logger,
                          "  [MC] Warning: No gating training data produced")

        except Exception as e:
            print_log(self.logger, f"  [MC] Error during MC process: {e}")
            import traceback
            traceback.print_exc()

    # Broadcasting
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

    # Training (DE process — identical to FLIM_continuous)
    def _train_step(self):
        """Full DE pipeline: rotation -> feature migration -> classification."""
        self.train_rot()
        self.evaluate_rot()
        self.model = migrate_rot_to_cls(self.rot_model, self.model)
        self.train_cls()

    def train_cls(self):
        """Classification (downstream task) training."""
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
            if class_total[i] > 0
        ]
        print_log(self.logger, f"  Per-class: {result}")
        self.model.to(torch.device('cpu'))
