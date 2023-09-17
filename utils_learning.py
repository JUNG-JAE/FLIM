# ----------- System library ----------- #
import sys
import numpy as np
from collections import defaultdict
import random

# ----------- Learning library ----------- #
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

# ----------- Custom library ----------- #
from utils_system import create_directory, print_log
from conf import settings
# from data_lodaer import source_dataloader


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
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'custom':
        from models.customCNN import CustomCNN
        net = CustomCNN()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:
        net = net.cuda()

    return net


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def aggregation(args, models):
    aggregated_model = get_network(args)
    aggregated_model_dict = defaultdict(lambda: 0)

    coefficient = 1 / len(models)

    for model in models:
        for layer, params in model.state_dict().items():
            aggregated_model_dict[layer] += coefficient * params

    aggregated_model.load_state_dict(aggregated_model_dict)

    return aggregated_model


def save_model(base_path, minute, model, model_name):
    save_path = f"{base_path}/{minute}"
    create_directory(save_path)
    torch.save(model.state_dict(), f"{save_path}/{model_name}.pt")


def avg_cosine_similarity(model1, model2):
    similarities = []

    for layer1, layer2 in zip(model1.parameters(), model2.parameters()):
        similarity = cosine_similarity(layer1.view(-1).unsqueeze(0), layer2.view(-1).unsqueeze(0))
        similarities.append(similarity.item())

    return np.mean(similarities)


def model_to_vector(model):
    return torch.cat([param.view(-1) for param in model.parameters()])


def models_to_matrix(models):
    return np.vstack([model_to_vector(model).to('cpu').detach().numpy() for model in models])


def cosine_similarity_between_models(model1, model2):
    vec1 = model_to_vector(model1)
    vec2 = model_to_vector(model2)
    similarity = cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return similarity.item()


def manhattan_distance(model1, model2):
    vec1 = model_to_vector(model1)
    vec2 = model_to_vector(model2)
    distance = torch.cdist(vec1.unsqueeze(0), vec2.unsqueeze(0), p=1)
    
    return distance.item()




# @torch.no_grad()
# def source_evaluate(args, logger, model):
#     _, test_loader = source_dataloader()

#     loss_function = nn.CrossEntropyLoss()
#     class_correct = list(0. for i in range(10))
#     class_total = list(0. for i in range(10))

#     model.eval()
#     device = torch.device('cuda' if args.gpu else 'cpu')
#     model.to(device)

#     test_loss = 0.0
#     correct = 0.0

#     for inputs, targets in test_loader:
#         inputs, targets = inputs.to(device), targets.to(device)
#         outputs = model(inputs)
#         loss = loss_function(outputs, targets)

#         _, predicted = torch.max(outputs, 1)
#         c = (predicted == targets).squeeze()

#         for i in range(len(targets)):
#             label = targets[i]
#             class_correct[label] += c[i].item()
#             class_total[label] += 1

#         test_loss += loss.item()
#         _, predicts = outputs.max(1)
#         correct += predicts.eq(targets).sum()

#     print_log(logger, 'Evaluating Model ... ')
#     print_log(logger, f"Accuracy {correct.float() * 100 / len(test_loader.dataset):.2f}, Average loss: {test_loss / len(test_loader.dataset):.2f}")
#     print_log(logger, '-------------------------------------')
#     for i in range(10):
#         print_log(logger, 'Accuracy of %3s : %2d %%' % (settings.LABELS[i], 100 * class_correct[i] / class_total[i]))
#     print_log(logger, " ")

#     return correct.float() * 100 / len(test_loader.dataset)

