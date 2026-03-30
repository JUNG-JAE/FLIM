# ------------ System library ------------ #
import argparse
import datetime
import time
import numpy as np
import sys
import random
import itertools
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import shutil
# ------------ Learning library ------------ #

# ------------ Custom library ------------ #
from node import Node, migrate_cls_to_rot, migrate_rot_to_cls
from conf import settings
from utils_system import set_logger, slicer, create_directory, format_title, format_time_title, print_log, flatten_tuple, node_event_generator, copy_prior_time_slot
from utils_learning import set_seed, save_model, models_to_matrix, aggregation, cosine_similarity_between_models


def train_evaluate(logger, node, minute, base_path):
    print_log(logger, f"[ {node.node_id} ]")
    node.train_rot()
    node.evaluate_rot()
    node.model = migrate_rot_to_cls(node.rot_model, node.model)
    node.train()
    # node.source_evaluate() # 추가 
    # node.rot_model = migrate_cls_to_rot(node.model, node.rot_model)
    # save_model(base_path, minute, node.node_id, node.model, f'model0')
    print_log(logger, "\n")
    
# def train_evaluate(logger, node, minute, base_path):
#     print_log(logger, f"[ {node.node_id} ]")
#     node.train_rot()
#     node.evaluate_rot()
#     node.model = migrate_rot_to_cls(node.rot_model, node.model)
#     node.train()
#     node.source_evaluate()
#     node.rot_model = migrate_cls_to_rot(node.model, node.rot_model)
#     print_log(logger, "\n")


def all_node_save_model(nodes, base_path, minute):
    for node in nodes:
        save_model(base_path, minute, node.node_id, node.model, f'model0')


def copy_previous_models(node_id, base_path, minute):
    if minute != 0:
        create_directory(f'{base_path}/{minute}/{node_id}')
        shutil.copytree(f'{base_path}/{minute-1}/{node_id}', f'{base_path}/{minute}/{node_id}', dirs_exist_ok=True)


def nodes_broadcast(part_node, nodes):
    broadcasts = {}
    
    for bcast_node in part_node:
        recv_nodes = [node for node in nodes if node.recv_status and node != bcast_node]
        broadcasts[bcast_node.node_id] = [node.node_id for node in recv_nodes]
        
    return broadcasts

""" 
def get_received_from(broadcasts):
    received_from = {node: [] for node in broadcasts.keys()}
    for sender, receivers in broadcasts.items():
        for receiver in receivers:
            received_from[receiver].append(sender)
    return received_from

def get_received_from(broadcasts):
    received_from = {}
    
    # 모든 수신 노드에 대한 빈 리스트 초기화
    for receivers in broadcasts.values():
        for receiver in receivers:
            if receiver not in received_from:
                received_from[receiver] = []

    # 각 수신 노드에 대해 송신 노드 추가
    for sender, receivers in broadcasts.items():
        for receiver in receivers:
            received_from[receiver].append(sender)

    return received_from
"""

def get_received_from(broadcasts, all_nodes):
    received_from = {node: [] for node in all_nodes}  # Initialize every node with an empty list

    # Populate the dictionary with senders for each receiver
    for sender, receivers in broadcasts.items():
        for receiver in receivers:
            received_from[receiver].append(sender)

    return received_from


def model_clustering(args, logger, base_path, minute, node, part_node, received_from):
    print_log(logger, f"[ {node.node_id} ]")
    part_models = {node.node_id: node.model for node in part_node}
    recv_models = {recv_node_id: part_models[recv_node_id] for recv_node_id in received_from[node.node_id]}

    # 1. It measures whether the model received from other nodes is similar to the model it trained.
    if recv_models:
        similar_models = [node.model]
        similar_node_ids = []
        non_similar_models = {}
        
        for recv_node_id, recv_model in recv_models.items():
            recv_model_similarity = cosine_similarity_between_models(node.model, recv_model)  # similarity between my model and receive model
            similarity_distance = np.maximum(1 - recv_model_similarity, 0)
            
            if similarity_distance < args.sim_th:
                similar_models.append(recv_model)
                similar_node_ids.append(recv_node_id)
            else:
                non_similar_models[recv_node_id] = recv_model

        recv_models = non_similar_models  # Update recv_models with non-similar models

        # aggregate model
        if len(similar_models) > 1:
            print_log(logger, f"{node.node_id} model is similar to models from nodes: [{', '.join(similar_node_ids)}]")
            node.model = aggregation(args, similar_models)

    # 2. Compare the similarity between the received model and other models held by the node.
    # 1 번 과정으로 인해 recv_model이 없을 수 있음. 따라서 다시 확인해야 함
    if recv_models:            
        # Combine prior node_ids and received node_ids
        all_node_ids = list(node.other_models.keys()) + list(recv_models.keys())
        all_models = list(node.other_models.values()) + list(recv_models.values())
        
        # Create a similarity matrix and perform clustering
        model_matrix = models_to_matrix(all_models)
        similarity_matrix = cosine_similarity(model_matrix)
        distance_matrix = np.maximum(1 - similarity_matrix, 0)
        db = DBSCAN(eps=args.sim_th, min_samples=1, metric='precomputed').fit(distance_matrix)
    
        # Create a dictionary to store clustered node_ids and their models
        clustered_data = {}
        for save_order, cluster_label in enumerate(np.unique(db.labels_), start=1):
            indices = np.where(db.labels_ == cluster_label)[0].tolist()
            
            clustered_node_ids = [all_node_ids[idx] for idx in indices]            
            clustered_models = [all_models[idx] for idx in indices]
            
            flat_clustered_node_ids = flatten_tuple(clustered_node_ids)
            print_log(logger, f"{save_order} Clustered nodes: {flat_clustered_node_ids}")

            agg_model = aggregation(args, clustered_models)
            save_model(base_path, minute, node.node_id, agg_model, f"model{save_order}")
            clustered_data[tuple(clustered_node_ids)] = agg_model
            
        # Update node's other_models with the new clustered data
        node.other_models = clustered_data
        
    if len(node.other_models) > settings.SUP_OTHER_MODEL_SIZE:
        node.recv_status = False
    
    print_log(logger, " ")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10_example', required=False, help='select experiment dataset')
    parser.add_argument('--n_node', type=int, default=10, required=False, help='number of node in network')
    parser.add_argument('--sim_th', type=float, default=0.6, required=False, help='similarity threshold')
    parser.add_argument('--lamb', type=int, default=6, required=False, help='poisson process lambda')
    parser.add_argument('--straggler', type=int, default=0, required=False, help='straggler node ration in network')
    parser.add_argument('--mode', type=str, default='withRot', required=False, help='training with rotation')
    parser.add_argument('--net', type=str, default='vgg11', required=False, help='neural network type')
    parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
    args = parser.parse_args()

    
    # Set base path and logger
    BASE_PATH = f'{settings.LOG_DIR}/{args.dataset}_node{args.n_node}_sim{str(args.sim_th).replace(".", "")}_E{settings.SUP_OTHER_MODEL_SIZE}_epoch{settings.CLS_EPOCH}_batch{settings.BATCH_SIZE}_lambda{args.lamb}_straggler{args.straggler}'
    logger = set_logger(BASE_PATH)
    
    # Generate nodes
    nodes = [Node(args, logger, f'node{node_idx}') for node_idx in np.arange(args.n_node)]
    
    # Select the nodes that broadcast the model based on poisson distribution
    straggler_nodes, node_events = node_event_generator(args, BASE_PATH)
    
    print_log(logger, f"Straggler nodes: {straggler_nodes}")
    start_time = datetime.datetime.now()
    print_log(logger, f"Execution started at: {start_time}")
    
    for time_slot in range(0, settings.SIMULATION_TIME):
        print_log(logger, format_time_title(f"Time {time_slot}-{time_slot+1}"))
        
        """ node의 최대 expert 저장 개수가 초과 했는지 확인하는 코드 """
        exceeding_nodes = [node.node_id for node in nodes if node.recv_status == False]
        print_log(logger, f"Exceeding nodes: {exceeding_nodes}")

        # Check if all nodes' other_models length is greater than or equal to settings.SUP_OTHER_MODEL_SIZE
        if len(exceeding_nodes) == len(nodes):
            print_log(logger, f"All nodes.other_models have greater than the SUP_OTHER_MODEL_SIZE!({settings.SUP_OTHER_MODEL_SIZE})")
            raise ValueError(" ")
        """ ============================================== """

        node_events_in_slot = [node_id for time, node_id in node_events if time_slot <= time < time_slot + 1]
        print_log(logger, f"Participate node: {node_events_in_slot}")
        
        # participate_nodes = [node for node in nodes if node.node_id in node_events_in_slot]
        participate_nodes = [node for node in nodes for id in node_events_in_slot if node.node_id == id]
        if len(participate_nodes) == 1:
            # Train and evaluate model
            print_log(logger, format_title("Training Step"))
            for node in participate_nodes:
                train_evaluate(logger, node, time_slot, BASE_PATH)

            # Broadcast nodes
            print_log(logger, format_title("Brocasting Step"))
            broadcasts = nodes_broadcast(participate_nodes, nodes)
            for sender, receivers in broadcasts.items():
                print_log(logger, f"{sender} -> {receivers}")
            print_log(logger, "\n")
            
            # Print receive from
            print_log(logger, format_title("Received From"))
            received_from = get_received_from(broadcasts, [f'node{node_idx}' for node_idx in range(0, args.n_node)])
            for node in nodes:
                print_log(logger, f"{node.node_id}: {received_from[node.node_id]}")
            print_log(logger, "\n")
            
            print_log(logger, f"{participate_nodes[0].node_id}'s previous model copied")
            copy_previous_models(participate_nodes[0].node_id, BASE_PATH, time_slot)
            
            # Received model clustering
            print_log(logger, format_title("Received model clustering"))
            for node in nodes:
                model_clustering(args, logger, BASE_PATH, time_slot, node, participate_nodes, received_from)
            print_log(logger, "\n")
            
            all_node_save_model(nodes, BASE_PATH, time_slot)
        elif len(participate_nodes) != 0:
            # Train and evaluate model
            print_log(logger, format_title("Training Step"))
            for node in participate_nodes:
                train_evaluate(logger, node, time_slot, BASE_PATH)

            # Broadcast nodes
            print_log(logger, format_title("Brocasting Step"))
            broadcasts = nodes_broadcast(participate_nodes, nodes)
            for sender, receivers in broadcasts.items():
                print_log(logger, f"{sender} -> {receivers}")
            print_log(logger, "\n")
            
            # Print receive from
            print_log(logger, format_title("Received From"))
            received_from = get_received_from(broadcasts, [f'node{node_idx}' for node_idx in range(0, args.n_node)])
            for node in nodes:
                print_log(logger, f"{node.node_id}: {received_from[node.node_id]}")
            print_log(logger, "\n")
            
            # Received model clustering
            print_log(logger, format_title("Received model clustering"))
            for node in nodes:
                model_clustering(args, logger, BASE_PATH, time_slot, node, participate_nodes, received_from)
            print_log(logger, "\n")
            
            all_node_save_model(nodes, BASE_PATH, time_slot)
        else:
            print_log(logger, "Copy prior time slot info")
            copy_prior_time_slot(BASE_PATH, time_slot)
            all_node_save_model(nodes, BASE_PATH, time_slot)
    
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    
    print_log(logger, f"Execution ended at: {end_time}")
    print_log(logger, f"Execution time: {execution_time}")
    return 0


if __name__ == '__main__':
    set_seed()
    main()
    