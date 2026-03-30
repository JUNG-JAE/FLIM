import os
import sys
import copy
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind, skew

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset
from sklearn.cluster import KMeans

from conf import settings
from utils_system import print_log, get_beta_distribution, create_directory
from utils_learning import load_cnn_layers_for_rotation

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# 1. Cluster unlabeled data using CNN features from experts
def cluster_unlabeled_data(experts, unlabeled_loader, n_clusters, device):
    feature_list = []
    image_list = []

    with torch.no_grad():
        for expert_model in experts.values():
            expert_model.to(device)
            expert_model.eval()

        for images, _ in unlabeled_loader:
            images = images.to(device)
            batch_features = []
            for expert_model in experts.values():
                # Extract features from the CNN backbone (before classifier)
                feat = expert_model.features(images)
                feat = feat.view(feat.size(0), -1)
                batch_features.append(feat)

            # Concatenate features from all experts
            concat_feat = torch.cat(batch_features, dim=1)
            feature_list.append(concat_feat.cpu().numpy())
            image_list.append(images.cpu())

        for expert_model in experts.values():
            expert_model.to(torch.device('cpu'))

    features_all = np.vstack(feature_list)
    all_images = torch.cat(image_list, dim=0)

    # k-means++ clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                    max_iter=10000, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(features_all)

    return cluster_labels, features_all, all_images


def split_cluster_train_test(cluster_labels, features_all, all_images, test_ratio=0.1):
    n_clusters = len(np.unique(cluster_labels))
    cluster_data = {}

    for k in range(n_clusters):
        mask = cluster_labels == k
        indices = np.where(mask)[0]
        feats_k = features_all[indices]
        imgs_k = all_images[indices]

        centroid = feats_k.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(feats_k - centroid, axis=1)
        sorted_idx = np.argsort(dists)

        n_test = max(1, int(len(indices) * test_ratio))
        test_idx = sorted_idx[:n_test]
        train_idx = sorted_idx[n_test:]

        cluster_data[k] = {
            'train_images': imgs_k[train_idx],
            'test_images': imgs_k[test_idx],
        }

    return cluster_data


# 2. Rotation transform for pretext task
def create_rotation_dataset(images):
    rot_images = []
    rot_labels = []

    for img in images:
        for r in range(4):
            rotated = torch.rot90(img, r, [1, 2])
            rot_images.append(rotated)
            rot_labels.append(r)

    return torch.stack(rot_images), torch.tensor(rot_labels, dtype=torch.long)


# 3. CLA (Conditional Loss Adjustment) — Algorithm 2
def train_mass_model_with_cla(model, train_loader, test_loader, epoch, learning_rate, device, logger=None):
    model.to(device)
    loss_function = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    reliability_toggle = False
    beta_posterior = {
        'negative_counter': 0,
        'positive_counter': 0,
        'cdf_probs': [],
        'relied_epoch': epoch  # default: never relied
    }

    sample_accs = []
    n_sample = settings.MASS_SAMPLE_SIZE

    for e in range(epoch):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Inference pass to get loss distribution
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                losses = loss_function(outputs, targets)

            np_losses = losses.cpu().numpy()
            loss_mean = np.mean(np_losses)
            loss_std = np.std(np_losses) + 1e-8
            loss_median = np.median(np_losses)
            loss_skewness = skew(np_losses) if len(np_losses) > 2 else 0.0

            # Beta distribution update (Eq. 11-13)
            tau_gap = settings.CLA_TAU_GAP  # ξ
            if loss_skewness > 0 and loss_median + tau_gap < loss_mean:
                beta_posterior['positive_counter'] += 1
            else:
                beta_posterior['negative_counter'] += 1

            # Select training samples
            if reliability_toggle:
                retained_inputs = inputs
                retained_targets = targets
            else:
                # Filter: only use samples with skewness contribution <= 0
                skewness_values = (np_losses - loss_mean) ** 3 / (loss_std ** 3)
                mask = skewness_values <= 0
                if mask.sum() == 0:
                    # If all samples are high-loss, use all (fallback)
                    retained_inputs = inputs
                    retained_targets = targets
                else:
                    retained_inputs = inputs[mask]
                    retained_targets = targets[mask]

            # Training pass
            model.train()
            optimizer.zero_grad()
            outputs = model(retained_inputs)
            loss = loss_function(outputs, retained_targets).mean()
            loss.backward()
            optimizer.step()

        # Check CDF threshold
        in_dist_prob = get_beta_distribution(
            beta_posterior['negative_counter'],
            beta_posterior['positive_counter']
        )
        beta_posterior['cdf_probs'].append(in_dist_prob)

        if in_dist_prob >= settings.CLA_DELTA and not reliability_toggle:
            reliability_toggle = True
            beta_posterior['relied_epoch'] = e

        # Collect sample accuracies from last n epochs
        if epoch - n_sample <= e:
            acc = evaluate_rotation(model, test_loader, device)
            sample_accs.append(acc)

    model.to(torch.device('cpu'))

    return sample_accs, beta_posterior['cdf_probs'], beta_posterior['relied_epoch']


def evaluate_rotation(model, test_loader, device):
    """Evaluate rotation classification accuracy."""
    model.eval()
    correct = 0.0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    return (correct / total) * 100 if total > 0 else 0.0


# 4. MASS with Bayesian Optimization
def run_mass_for_cluster(experts, cluster_train_images, cluster_test_images, device, logger=None, n_trials=None):
    if n_trials is None:
        n_trials = settings.MASS_N_TRIALS

    expert_list = list(experts.items())
    n_models = len(expert_list)

    if n_models == 0:
        return None, {}

    # Create rotation datasets
    rot_train_imgs, rot_train_labels = create_rotation_dataset(cluster_train_images)
    rot_test_imgs, rot_test_labels = create_rotation_dataset(cluster_test_images)

    if not HAS_OPTUNA:
        # Fallback: simple evaluation without Bayesian optimization
        return _mass_simple_eval(expert_list, rot_train_imgs, rot_train_labels,
                                 rot_test_imgs, rot_test_labels, device, logger)

    # -- Bayesian Optimization --
    best_result = {'best_expert': None, 'best_value': -float('inf'), 'expert_accuracies': {}}

    def objective(trial):
        n_train = trial.suggest_int("train", 100, min(settings.MASS_N_TRAIN_MAX, len(rot_train_imgs)), step=50)
        n_test = trial.suggest_int("test", 100, min(settings.MASS_N_TEST_MAX, len(rot_test_imgs)), step=50)
        epoch = trial.suggest_int("epoch", settings.MASS_EPOCH_MIN, settings.MASS_EPOCH_MAX)
        lr = trial.suggest_float("LR", 1e-4, 1e-3, log=True)

        # Build dataloaders with suggested sizes
        train_dataset = TensorDataset(rot_train_imgs[:n_train], rot_train_labels[:n_train])
        test_dataset = TensorDataset(rot_test_imgs[:n_test], rot_test_labels[:n_test])
        train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

        model_sample_acc = {}
        model_sample_mean = {}

        for eid, emodel in expert_list:
            # Build rotation model from expert's CNN
            rot_model = load_cnn_layers_for_rotation(emodel, device)

            sample_accs, _, _ = train_mass_model_with_cla(
                rot_model, train_loader, test_loader,
                epoch, lr, device, logger
            )

            mean_acc = round(np.mean(sample_accs), 2) if sample_accs else 0.0
            model_sample_acc[eid] = sample_accs
            model_sample_mean[eid] = mean_acc

        # Sort by accuracy
        sorted_items = sorted(model_sample_mean.items(), key=lambda x: x[1], reverse=True)
        first_key = sorted_items[0][0]
        second_key = sorted_items[1][0] if len(sorted_items) > 1 else sorted_items[0][0]

        # t-test for significance (Eq. 5)
        if first_key != second_key and len(model_sample_acc[first_key]) > 1:
            t_stat, p_value = ttest_ind(
                model_sample_acc[first_key],
                model_sample_acc[second_key],
                equal_var=False
            )
        else:
            t_stat, p_value = 0.0, 1.0

        trial.set_user_attr("Accuracy", model_sample_mean)

        # Objective: exp(t_stat) if significant, else 0
        if p_value > settings.MASS_P_VALUE:
            return 0

        value = np.exp(round(t_stat, 2))

        # Track best
        if value > best_result['best_value']:
            best_result['best_value'] = value
            best_result['best_expert'] = first_key
            best_result['expert_accuracies'] = model_sample_mean

        return value

    # Run optimization
    sampler = optuna.samplers.GPSampler(n_startup_trials=min(6, n_trials),
                                         deterministic_objective=False, seed=42)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return best_result['best_expert'], best_result['expert_accuracies']


def _mass_simple_eval(expert_list, rot_train_imgs, rot_train_labels, rot_test_imgs, rot_test_labels, device, logger=None):
    train_dataset = TensorDataset(rot_train_imgs, rot_train_labels)
    test_dataset = TensorDataset(rot_test_imgs, rot_test_labels)
    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    model_accs = {}
    epoch = settings.MASS_EPOCH_MIN
    lr = settings.LEARNING_RATE

    for eid, emodel in expert_list:
        rot_model = load_cnn_layers_for_rotation(emodel, device)
        sample_accs, _, _ = train_mass_model_with_cla(
            rot_model, train_loader, test_loader,
            epoch, lr, device, logger
        )
        model_accs[eid] = round(np.mean(sample_accs), 2) if sample_accs else 0.0

    best_id = max(model_accs, key=model_accs.get) if model_accs else None
    return best_id, model_accs


# 5. Full MASS pipeline: cluster + map + return mapping
def run_mass_pipeline(experts, unlabeled_loader, n_classes, device, logger=None, n_trials=None):
    if logger:
        print_log(logger, "  [MASS] Starting MASS pipeline...")
        print_log(logger, f"  [MASS] Experts: {list(experts.keys())}")

    # Step 1: Cluster
    cluster_labels, features_all, all_images = cluster_unlabeled_data(
        experts, unlabeled_loader, n_classes, device
    )

    if logger:
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print_log(logger, f"  [MASS] Clusters: {dict(zip(unique.tolist(), counts.tolist()))}")

    # Step 2: Split
    cluster_data = split_cluster_train_test(cluster_labels, features_all, all_images)

    # Step 3: Map each cluster to best expert
    mapping = {}
    expert_accuracies_per_cluster = {}

    for k, cdata in cluster_data.items():
        if logger:
            print_log(logger, f"  [MASS] Evaluating cluster {k}...")

        best_expert, accs = run_mass_for_cluster(
            experts,
            cdata['train_images'],
            cdata['test_images'],
            device, logger, n_trials
        )

        mapping[k] = best_expert
        expert_accuracies_per_cluster[k] = accs

        if logger:
            print_log(logger, f"  [MASS] Cluster {k} -> Expert {best_expert}")

    if logger:
        print_log(logger, f"  [MASS] Final mapping: {mapping}")

    return mapping, expert_accuracies_per_cluster
