import torch
from models.vgg import vgg11_bn
from models.multi_head_model import multi_head_vgg11_bn
from utils_learning import load_cls_models, get_label
from data_loader import cls_dataloader

def test_router(multi_head_model, node, testloader, labels, device):
    test_loader = testloader
    experts = load_cls_models(node, labels, device)
    
    correct = 0.0
    multi_head_model.eval()
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = multi_head_model(inputs)
        _, predicts = outputs.max(1)
        
        expert = experts[predicts.item()].eval()
        expert_outputs = expert(inputs)
        _, expert_predicts = expert_outputs.max(1)
        
        correct += expert_predicts.eq(targets).item()
    
    accuracy = (correct / len(test_loader.dataset)) * 100
    print(accuracy)


def main():
    device = torch.device('cuda')
    EXP = 'cifar10_h_w10_l2_aug_node10_sim05_E9_epoch_1_5_batch64'
    PROJECT = '1000_500_Epoch_10_50_0.0_withFC_robust_0.5'
    NODE = 'node5'
    BASE_PATH = f'./controller/runs/{EXP}/{PROJECT}/{NODE}'
    
    test_loader = cls_dataloader()
    labels = get_label(f"{PROJECT}_{NODE}")
    
    router = multi_head_vgg11_bn(len(labels)).to(device)
    router.load_state_dict(torch.load(f'{BASE_PATH}/router/router10.pt'))
    
    test_router(router, NODE, test_loader, labels, device)
    
if __name__ == '__main__':
    main()