# ------------ System library ------------ #
import argparse
import numpy as np
import sys
import random
import itertools
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# ------------ Learning library ------------ #

# ------------ Custom library ------------ #
from node import Node
from conf import settings
from utils_system import poisson_distribution, set_logger, slicer, create_directory, format_title, format_time_title, print_log
from utils_learning import set_seed, save_model, models_to_matrix, aggregation, cosine_similarity_between_models


def train_evaluate(logger, node, minute, base_path):
    print_log(logger, f"[ {node.node_id} ]")
    node.train()
    node.evaluate()
    save_model(base_path, minute, node.model, node.node_id)
    print_log(logger, "\n")


def nodes_broadcast(part_node, n_peer_set):
    # n_peer_set = min(n_peer_set, len(part_node) - 1)
    
    broadcasts = {}
    for bcast_node in part_node:
        # filtered_nodes = [node for node in part_node if len(node.other_models) < settings.SUP_OTHER_MODEL_SIZE]
        # print(f"Filtered nodes: {[node.node_id for node in filtered_nodes]}")
        # recv_node = random.sample([node for node in filtered_nodes if node != bcast_node], min(n_peer_set, len(filtered_nodes)-1))
        # recv_node = random.sample([node for node in part_node if node != bcast_node], n_peer_set)
        filtered_nodes = [node for node in part_node if len(node.other_models) < settings.SUP_OTHER_MODEL_SIZE and node != bcast_node]
        # print(f"Filtered nodes: {[node.node_id for node in filtered_nodes]}")
        recv_node = random.sample(filtered_nodes, min(n_peer_set, len(filtered_nodes)))
        
        broadcasts[bcast_node.node_id] = [node.node_id for node in recv_node]
    return broadcasts


def get_received_from(broadcasts):
    received_from = {node: [] for node in broadcasts.keys()}
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
        all_node_ids = [item for sublist in all_node_ids for item in (sublist if isinstance(sublist, tuple) else [sublist])]  # Flatten the list
    
        # Flatten the models from node.other_models and combine with recv_models
        all_models = [model for sublist in node.other_models.values() for model in sublist] + list(recv_models.values())
    
        # Create a similarity matrix and perform clustering
        model_matrix = models_to_matrix(all_models)
        similarity_matrix = cosine_similarity(model_matrix)
        distance_matrix = np.maximum(1 - similarity_matrix, 0)
        db = DBSCAN(eps=args.sim_th, min_samples=1, metric='precomputed').fit(distance_matrix)
    
        # Create a dictionary to store clustered node_ids and their models
        clustered_data = {}
        for save_order, cluster_label in enumerate(np.unique(db.labels_)):
            indices = np.where(db.labels_ == cluster_label)[0].tolist()
            clustered_node_ids = [all_node_ids[idx] for idx in indices]
            clustered_models = [all_models[idx] for idx in indices]
            print_log(logger, f"Clustered nodes: {clustered_node_ids}")
            agg_model =aggregation(args, clustered_models)
            save_model(base_path, minute, agg_model, f"{node.node_id}_other_{save_order}")
            clustered_data[tuple(clustered_node_ids)] = [agg_model]
        
        # Update node's other_models with the new clustered data
        node.other_models = clustered_data
        # print(f"keys: {node.other_models.keys()}")
        # print(len(node.other_models))   
    print_log(logger, " ")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10_hard_w10', required=False, help='select experiment dataset')
    parser.add_argument('--n_node', type=int, default=5, required=False, help='number of node in network')
    parser.add_argument('--b_rate', type=float, default=0.5, required=False, help='broadcast rate')
    parser.add_argument('--sim_th', type=float, default=0.5, required=False, help='similarity threshold')
    parser.add_argument('--net', type=str, default='vgg11', required=False, help='neural network type')
    parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
    args = parser.parse_args()

    # Set base path and logger
    BASE_PATH = f'{settings.LOG_DIR}/{slicer(args.dataset)}_node{args.n_node}_Brate{str(args.b_rate).replace(".", "")}_Stask{settings.SUP_OTHER_MODEL_SIZE}_epoch_{settings.INF_EPOCH}_{settings.SUP_EPOCH}_batch{settings.BATCH_SIZE}'
    logger = set_logger(BASE_PATH)
    
    # Generate nodes
    nodes = [Node(args, logger, f'node{node_idx}') for node_idx in np.arange(args.n_node)]
    
    # Select the nodes that broadcast the model based on poisson distribution
    poisson = [min(random_variable, args.n_node) for random_variable in poisson_distribution(args.n_node)]

    for minute, n_part_node in enumerate(poisson, start=1):
        print_log(logger, format_time_title(f"Time {minute}/min"))
        
        exceeding_nodes = [node.node_id for node in nodes if len(node.other_models) >= settings.SUP_OTHER_MODEL_SIZE]
        print_log(logger, f"Exceeding nodes: {exceeding_nodes}")

        # Check if all nodes' other_models length is greater than or equal to settings.SUP_OTHER_MODEL_SIZE
        if len(exceeding_nodes) == len(nodes):
            print_log(logger, f"All nodes.other_models have greater than the SUP_OTHER_MODEL_SIZE!({settings.SUP_OTHER_MODEL_SIZE})")
            sys.exit()

        # part_node = random.sample(filtered_nodes, min(n_part_node, len(filtered_nodes)))
        part_node = random.sample(nodes, n_part_node)
        print_log(logger, f"Participate node: {[node.node_id for node in part_node]} \n")
        
        # Train and evaluate model
        print_log(logger, format_title("Training Step"))
        for node in part_node:
            train_evaluate(logger, node, minute, BASE_PATH)

        # Set the number of nodes to broadcast based on broadcasting rate
        n_peer_set = round(n_part_node * args.b_rate)
        
        # Broadcast nodes
        print_log(logger, format_title("Brocasting Step"))
        broadcasts = nodes_broadcast(part_node, n_peer_set)
        for sender, receivers in broadcasts.items():
            print_log(logger, f"{sender} -> {receivers}")
        print_log(logger, "\n")
        
        # Print receive from
        print_log(logger, format_title("Received From"))
        received_from = get_received_from(broadcasts)
        for bcast_node in part_node:
            print_log(logger, f"{bcast_node.node_id}: {received_from[bcast_node.node_id]}")
        print_log(logger, "\n")
        
        # Received model clustering
        print_log(logger, format_title("Received model clustering"))
        for node in part_node:
            model_clustering(args, logger, BASE_PATH, minute, node, part_node, received_from)
        print_log(logger, "\n")
        
    return 0


if __name__ == '__main__':
    set_seed()
    main()