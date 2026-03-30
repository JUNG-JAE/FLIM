from models.vgg import vgg11_bn
import torch
import os
from data_loader import testloader
import torch.nn as nn
from target import target_label_dict

def label_evaluate(model, test_loader, device):
    LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    loss_function = nn.CrossEntropyLoss()
    class_correct = list(0. for i in range(len(LABELS)))
    class_total = list(0. for i in range(len(LABELS)))
    class_loss = list(0. for i in range(len(LABELS)))  # 각 레이블의 Loss를 저장하기 위한 리스트 추가

    model.eval()
    model.to(device)

    test_loss = 0.0
    correct = 0.0

    for inputs, targets, _ in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        _, predicted = torch.max(outputs, 1)
        c = (predicted == targets).squeeze()

        for i in range(len(targets)):
            label = targets[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            class_loss[label] += loss.item()

        test_loss += loss.item()
        _, predicts = outputs.max(1)
        correct += predicts.eq(targets).sum()

    max_label_length = max(len(label) for label in LABELS)+1
    print(f"Accuracy {correct.float() * 100 / len(test_loader.dataset):.2f}, Average loss: {test_loss / len(test_loader.dataset):.2f}")
    print('-------------------------------------')
    for i in range(len(LABELS)):
        formatted_label = LABELS[i].ljust(max_label_length)
        print(f"Accuracy of {formatted_label}: {100 * class_correct[i] / class_total[i]:.2f}")
    print(" ")
    
test_loader = testloader()
device = torch.device('cuda')

node_id = 'node6'

for model_idx in range(10):
    model = vgg11_bn()
    model.load_state_dict(torch.load(f'{os.path.expanduser("~")}/Workspace/FLIM/runs/cifar10_h_w10_l2_aug_node10_sim05_E9_epoch_1_5_batch64/15/{node_id}/model{model_idx}.pt'))
    print(f"Target Labels: {target_label_dict[f'model{model_idx}']}")
    label_evaluate(model, test_loader, device)
    