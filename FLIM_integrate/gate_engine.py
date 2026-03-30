import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from conf import settings
from utils_system import print_log, create_directory


# 1. Build multi-head gating model
def build_gating_module(experts, expert_order, device):
    from models.multi_head_model import multi_head_vgg11_bn

    n_experts = len(expert_order)
    gating_model = multi_head_vgg11_bn(n_experts).to(device)

    # Load each expert's CNN into the corresponding head
    for idx, eid in enumerate(expert_order):
        expert_model = experts[eid]
        expert_model.to(device)
        gating_model.features_list[idx] = expert_model.features
        expert_model.to(torch.device('cpu'))

    return gating_model



# 2. Prepare gating training data from MASS mapping
def prepare_gating_data(cluster_data, mass_mapping, expert_order):
    expert_to_idx = {eid: idx for idx, eid in enumerate(expert_order)}

    train_imgs_list = []
    train_labels_list = []
    test_imgs_list = []
    test_labels_list = []

    for cluster_id, cdata in cluster_data.items():
        expert_id = mass_mapping.get(cluster_id)
        if expert_id is None or expert_id not in expert_to_idx:
            continue

        label_idx = expert_to_idx[expert_id]

        # Training data
        n_train = len(cdata['train_images'])
        train_imgs_list.append(cdata['train_images'])
        train_labels_list.append(torch.full((n_train,), label_idx, dtype=torch.long))

        # Test data
        n_test = len(cdata['test_images'])
        test_imgs_list.append(cdata['test_images'])
        test_labels_list.append(torch.full((n_test,), label_idx, dtype=torch.long))

    if not train_imgs_list:
        return None, None, None, None

    train_images = torch.cat(train_imgs_list, dim=0)
    train_labels = torch.cat(train_labels_list, dim=0)
    test_images = torch.cat(test_imgs_list, dim=0)
    test_labels = torch.cat(test_labels_list, dim=0)

    return train_images, train_labels, test_images, test_labels



# 3. Train gating module
def train_gating_module(gating_model, train_images, train_labels, test_images, test_labels, device, logger=None):
    # Freeze CNN heads, only train the classifier (FC)
    for param in gating_model.parameters():
        param.requires_grad = False
    for param in gating_model.classifier.parameters():
        param.requires_grad = True

    gating_model.to(device)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, gating_model.parameters()),
        lr=settings.GATE_LEARNING_RATE
    )
    loss_function = nn.CrossEntropyLoss()

    # Phase 1: Train on MASS mapping labels
    for epoch in range(settings.GATE_EPOCH):
        gating_model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = gating_model(inputs)
            loss = loss_function(outputs, targets)

            # L1 regularization (sparse gating)
            l1_norm = sum(p.abs().sum() for p in gating_model.classifier.parameters()
                          if p.requires_grad)
            loss = loss + 1e-4 * l1_norm

            loss.backward()
            optimizer.step()

    # Phase 2: Further training with high-confidence samples from train set
    # (Section 3.2.4 — fine-tune with k^TR samples above softmax 0.99)
    gating_model.eval()
    high_conf_images = []
    high_conf_labels = []

    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            outputs = gating_model(inputs)
            softmax_out = torch.softmax(outputs, dim=1)
            max_conf, preds = softmax_out.max(dim=1)

            mask = max_conf > settings.GATE_CONFIDENCE_LEVEL
            if mask.sum() > 0:
                high_conf_images.append(inputs[mask].cpu())
                high_conf_labels.append(preds[mask].cpu())

    if high_conf_images:
        hc_images = torch.cat(high_conf_images, dim=0)
        hc_labels = torch.cat(high_conf_labels, dim=0)
        hc_dataset = TensorDataset(hc_images, hc_labels)
        hc_loader = DataLoader(hc_dataset, batch_size=settings.BATCH_SIZE, shuffle=True)

        for epoch in range(settings.GATE_EPOCH // 2):
            gating_model.train()
            for inputs, targets in hc_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = gating_model(inputs)
                loss = loss_function(outputs, targets)
                l1_norm = sum(p.abs().sum() for p in gating_model.classifier.parameters()
                              if p.requires_grad)
                loss = loss + 1e-4 * l1_norm
                loss.backward()
                optimizer.step()

    # Evaluate routing accuracy
    routing_acc = evaluate_routing(gating_model, test_loader, device)

    if logger:
        print_log(logger, f"  [Gate] Routing accuracy: {routing_acc:.2f}%")

    gating_model.to(torch.device('cpu'))

    return routing_acc


# 4. Evaluate routing accuracy
def evaluate_routing(gating_model, test_loader, device):
    gating_model.eval()
    correct = 0.0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = gating_model(inputs)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    return (correct / total) * 100 if total > 0 else 0.0



# 5. Full MoE inference
@torch.no_grad()
def moe_inference(gating_model, experts, expert_order, test_loader, device):
    gating_model.to(device)
    gating_model.eval()

    expert_models = []
    for eid in expert_order:
        m = experts[eid]
        m.to(device)
        m.eval()
        expert_models.append(m)

    correct = 0.0
    total = 0

    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Route through gating module
        gate_outputs = gating_model(inputs)
        _, selected = gate_outputs.max(1)

        # Forward through selected experts (per-sample routing)
        for i in range(inputs.size(0)):
            expert_idx = selected[i].item()
            expert_output = expert_models[expert_idx](inputs[i:i+1])
            _, pred = expert_output.max(1)
            correct += pred.eq(targets[i:i+1]).sum().item()
            total += 1

    # Move back to CPU
    gating_model.to(torch.device('cpu'))
    for m in expert_models:
        m.to(torch.device('cpu'))

    return (correct / total) * 100 if total > 0 else 0.0
