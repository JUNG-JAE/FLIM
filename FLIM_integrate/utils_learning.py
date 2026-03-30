# ----------- System library ----------- #
import sys
import os
import numpy as np
import copy
from collections import defaultdict
import random

# ----------- Learning library ----------- #
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

# ----------- Custom library ----------- #
from utils_system import create_directory, print_log
from conf import settings


def get_network(args):
    if args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def aggregation(args, models):
    aggregated_model = get_network(args).to(torch.device('cpu'))
    aggregated_model_dict = defaultdict(lambda: 0)

    coefficient = 1 / len(models)

    for model in models:
        for layer, params in model.state_dict().items():
            aggregated_model_dict[layer] += coefficient * params

    aggregated_model.load_state_dict(aggregated_model_dict)

    return aggregated_model


def correlation_matrix_aggregation(args, new_experts, existing_experts, threshold):
    if not new_experts and not existing_experts:
        return {}

    # Build lists
    H_t = new_experts                   # [(id, model), ...]
    E_t = list(existing_experts.items()) # [(id, model), ...]

    n_new = len(H_t)
    n_exist = len(E_t)

    if n_exist == 0:
        # No existing experts: all new experts become the initial set
        return {eid: m for eid, m in H_t}

    # Correlation matrix M: rows = new, cols = existing
    M = np.zeros((n_new, n_exist))

    append_list = []  # new experts that don't match any existing

    for i, (hid, hmodel) in enumerate(H_t):
        distances = []
        for j, (eid, emodel) in enumerate(E_t):
            d = cosine_distance_between_models(hmodel, emodel)
            distances.append(d)

        min_dist = min(distances)
        if min_dist < threshold:
            closest_j = int(np.argmin(distances))
            M[i, closest_j] = 1
        else:
            append_list.append((hid, hmodel))

    # Build aggregated experts
    updated = {}
    for j, (eid, emodel) in enumerate(E_t):
        col_sum = M[:, j].sum()
        if col_sum > 0:
            # Weighted aggregation: E_j^{t+1} = (sum M * H + E) / (1 + sum M)
            agg_state = defaultdict(lambda: 0)
            for i in range(n_new):
                if M[i, j] > 0:
                    for layer, params in H_t[i][1].state_dict().items():
                        agg_state[layer] += M[i, j] * params
            for layer, params in emodel.state_dict().items():
                agg_state[layer] += params

            weight = 1.0 + col_sum
            new_model = get_network(args).to(torch.device('cpu'))
            final_state = {}
            for layer in agg_state:
                final_state[layer] = agg_state[layer] / weight
            new_model.load_state_dict(final_state)
            updated[eid] = new_model
        else:
            updated[eid] = emodel

    # Append truly new experts
    for hid, hmodel in append_list:
        updated[hid] = hmodel

    return updated


def save_model(base_path, minute, node_id, model, model_name):
    save_path = f"{base_path}/{minute}/{node_id}"
    create_directory(save_path)
    torch.save(model.state_dict(), f"{save_path}/{model_name}.pt")


def model_to_vector(model):
    return torch.cat([param.view(-1) for param in model.parameters()])


def models_to_matrix(models):
    return np.vstack([model_to_vector(model).to('cpu').detach().numpy() for model in models])


def cosine_similarity_between_models(model1, model2):
    vec1 = model_to_vector(model1)
    vec2 = model_to_vector(model2)
    similarity = cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return similarity.item()


def cosine_distance_between_models(model1, model2):
    sim = cosine_similarity_between_models(model1, model2)
    return max(1 - sim, 0)


# Model migration (rotation ↔ classification)
def migrate_cls_to_rot(cls_model, rot_model):
    rotation_model = rot_model
    rotation_model_dict = rotation_model.state_dict()

    for layer, params in cls_model.state_dict().items():
        if 'classifier' not in layer:
            rotation_model_dict[layer] = params

    rotation_model_dict['classifier.0.weight'] = cls_model.state_dict()['classifier.0.weight']
    rotation_model_dict['classifier.0.bias'] = cls_model.state_dict()['classifier.0.bias']

    rotation_model.load_state_dict(rotation_model_dict)

    for name, param in rotation_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = True

    rotation_model_dict['classifier.0.weight'].requires_grad = True
    rotation_model_dict['classifier.0.bias'].requires_grad = True

    for layer in rotation_model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True
            layer.track_running_stats = True

    return rotation_model


def migrate_rot_to_cls(rot_model, cls_model):
    classification_model = cls_model
    cls_model_dict = classification_model.state_dict()

    for layer, params in rot_model.state_dict().items():
        if 'classifier' not in layer:
            cls_model_dict[layer] = params

    cls_model_dict['classifier.0.weight'] = rot_model.state_dict()['classifier.0.weight']
    cls_model_dict['classifier.0.bias'] = rot_model.state_dict()['classifier.0.bias']

    classification_model.load_state_dict(cls_model_dict)

    for name, param in classification_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = True

    cls_model_dict['classifier.0.weight'].requires_grad = True
    cls_model_dict['classifier.0.bias'].requires_grad = True

    for layer in classification_model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True
            layer.track_running_stats = True

    return classification_model


# MASS model building: load CNN layers for rotation pretext task
def load_cnn_layers_for_rotation(expert_model, device):
    from models.vgg_rotation import vgg11_bn as vgg_rotation

    rotation_model = vgg_rotation().to(device)
    rotation_model_dict = rotation_model.state_dict()

    for layer, params in expert_model.state_dict().items():
        if 'classifier' not in layer:
            rotation_model_dict[layer] = params

    # Also load first FC layer
    rotation_model_dict['classifier.0.weight'] = expert_model.state_dict()['classifier.0.weight']
    rotation_model_dict['classifier.0.bias'] = expert_model.state_dict()['classifier.0.bias']

    rotation_model.load_state_dict(rotation_model_dict)

    # Freeze CNN layers
    for name, param in rotation_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    # Unfreeze first FC
    rotation_model_dict['classifier.0.weight'].requires_grad = True
    rotation_model_dict['classifier.0.bias'].requires_grad = True

    # Keep batchnorm trainable
    for layer in rotation_model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True
            layer.track_running_stats = True

    return rotation_model
