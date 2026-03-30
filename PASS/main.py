# ----------- System library ----------- #
import numpy as np
import optuna
import argparse
import statistics
from scipy.stats import kurtosis
from optuna.integration import SkoptSampler
import pickle
import math
from scipy.stats import ttest_ind

# ----------- Learning library ----------- #
import torch
import torch.nn as nn
import torch.optim as optim

# ----------- Custom library ----------- #
from conf import settings
from utils_system import save_result, load_result, create_directory, set_seed, set_logger, print_log
from data_loader import rot_trainloader, rot_testloader
from utils_learning import load_cls_model, load_CNN_layers, load_CNN_with_FC_layers, train_and_eval, robust_train_and_eval, maximize_distinction_learning
device = torch.device('cuda')

def optimize_model(object_function, n_trials):    
    # sampler = SkoptSampler(skopt_kwargs={'n_random_starts':6, 'acq_func':'EI', 'acq_func_kwargs': {'xi':0.02}})
    sampler = optuna.samplers.GPSampler(n_startup_trials=6, deterministic_objective=False, seed=42) # Bayesian Optimization / Gaussian Process Regression + log Expected Improvement

    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(object_function, n_trials=n_trials)

    return study
 
 
def object_function(trial):
    n_train_sets = trial.suggest_int("train", 100, settings.N_TRAIN_MAX, step=50)
    n_test_sets = trial.suggest_int("test", 100, settings.N_TEST_MAX, step=50)
        
    epoch = trial.suggest_int("epoch", settings.EPOCH_MIN, settings.EPOCH_MAX)
    learning_rate = trial.suggest_float("LR", 1e-4, 1e-3, log=True)
    
    print_log(logger, f"Trial:{trial.number} | Train set: {n_train_sets} Test set: {n_test_sets} Epoch: {epoch} Learning rate: {learning_rate:.4f}")

    train_loader = rot_trainloader(args, args.cls_label, n_train_sets)
    test_loader = rot_testloader(args, args.cls_label, n_test_sets)
    
    model_sample_acc = {}
    model_sample_mean = {}
    model_sample_std = {}
    if args.train_mode == 'MDL':
        model_over_epoch = {}
        
    for model_idx in range(args.n_model):
        cls_model = load_cls_model(args, f'model{model_idx}', device)
        rotation_model = load_CNN_with_FC_layers(cls_model, device) if args.withFC else load_CNN_layers(cls_model, device)
        
        if args.train_mode == 'normal':
            sample_accs = train_and_eval(args, f'model{model_idx}', rotation_model, train_loader, test_loader, epoch, learning_rate, device)
        elif args.train_mode == 'robust':
            sample_accs = robust_train_and_eval(args, f'model{model_idx}', rotation_model, train_loader, test_loader, epoch, learning_rate, args.tau, device)
        elif args.train_mode == 'MDL':
            sample_accs, cdf_probs, relied_epoch, data_log = maximize_distinction_learning(args, f'model{model_idx}', rotation_model, train_loader, test_loader, epoch, learning_rate, device)
        
        model_sample_acc[f'model{model_idx}'] = sample_accs
        model_sample_mean[f'model{model_idx}'] = sample_acc = round(np.mean(sample_accs), 2)
        model_sample_std[f'model{model_idx}'] = sample_std = round(np.std(sample_accs, ddof=1), 2)

        create_directory(f'{BASE_PTAH}/sample_acc/{trial.number}')
        with open(f'{BASE_PTAH}/sample_acc/{trial.number}/model{model_idx}.pkl', 'wb') as file:
            pickle.dump(sample_accs, file)
            
        if args.train_mode == 'MDL':
            create_directory(f'{BASE_PTAH}/beta/{trial.number}')
            create_directory(f'{BASE_PTAH}/data/{trial.number}')
                        
            with open(f'{BASE_PTAH}/beta/{trial.number}/model{model_idx}_prob.pkl', 'wb') as file:
                pickle.dump(cdf_probs, file)
                
            with open(f'{BASE_PTAH}/data/{trial.number}/model{model_idx}_used.pkl', 'wb') as file:
                pickle.dump(data_log, file)
                
        if args.train_mode == 'MDL':                    
            print_log(logger, f"(Model{model_idx}) Accuracy: {sample_acc:.2f} | CDF{max(cdf_probs):.2f} over epoch at: {relied_epoch}")
            model_over_epoch[f'model{model_idx}'] = relied_epoch
        else:
            print_log(logger, f"(Model{model_idx}) Accuracy: {sample_acc:.2f} | STD: {sample_std:.2f}")
        
    sorted_items = sorted(model_sample_mean.items(), key=lambda x: x[1], reverse=True)
    first_key = sorted_items[0][0]
    second_key = sorted_items[1][0]

    print_log(logger, f"First high acc: [{first_key}] Second high acc: [{second_key}]")

    # t-test
    t_stat, p_value = ttest_ind(model_sample_acc[first_key], model_sample_acc[second_key], equal_var=False)

    trial.set_user_attr("Accuracy", model_sample_mean)
    trial.set_user_attr("STD", model_sample_std)
    
    if args.train_mode == 'MDL':
        trial.set_user_attr('over_epoch', model_over_epoch)
        
    print_log(logger, f"T value: {t_stat:.2f} P value: {p_value:.2f}\n")

    interval = np.exp(round(t_stat, 2))

    trial.set_user_attr("t value", t_stat)
    trial.set_user_attr("p value", p_value)

    if p_value > settings.P_VALUE:
        return 0

    return interval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='cifar10_h_w10_l2_aug_node10_sim05_E9_epoch_1_5_batch64', help="input experiment title")
    parser.add_argument('--dataset', type=str, default='cifar10_64_CNN_clusterted', help="Select dataset with used for experiment")
    parser.add_argument('--n_trial', type=int, default=10, help="number of trial")
    parser.add_argument('--cls_label', type=str, default='airplane', help="Select classification label for experiment")
    
    parser.add_argument('--n_model', type=int, default=10, help="Select number of pretrained model")
    parser.add_argument('--node_id', type=str, default='node0', help="input node id")
    parser.add_argument('--slot', type=str, default='15', help="input model slot location")
    
    parser.add_argument('--train_mode', type=str, default='normal', help="normal, robust, MDL")
    parser.add_argument('--tau', type=float, default='0.0', help="noise rate for robust and adaptive")
    
    parser.add_argument('--withFC', action='store_true', default=True, help="T: not only load cnn layers but also load first FC layer")
    args = parser.parse_args()

    set_seed()
    
    withFC = "withFC" if args.withFC else "withoutFC"
    BASE_PTAH = f'{settings.LOG_DIR}/{args.exp}/{args.dataset}/{settings.N_TRAIN_MAX}_{settings.N_TEST_MAX}_Epoch_{settings.EPOCH_MIN}_{settings.EPOCH_MAX}_{withFC}_{args.train_mode}_{args.tau}/{args.node_id}/{args.cls_label}'
    
    create_directory(BASE_PTAH)
    
    logger = set_logger(BASE_PTAH)
    print_log(logger, f"===== CLS Label: {args.cls_label}, Node: {args.node_id}, Sample size: {settings.SAMPLE_SIZE}, P value: {settings.P_VALUE} Train mode: {args.train_mode} =====")
    
    study = optimize_model(object_function, args.n_trial)

    save_result(study, BASE_PTAH, args.cls_label)
    load_result(BASE_PTAH, args.cls_label)

