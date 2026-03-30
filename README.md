# SOMFed: A Self-Organized MoE Framework for Distributed Federated Learning

## Overview
Distributed federated learning is an attractive approach for privacy-preserving collaboration without relying on a central server, but its performance often degrades under asynchronous updates and highly non-IID local data. Existing approaches often compress diverse client knowledge into a single shared model, which can reduce specialization and lead to performance loss in heterogeneous environments. To address this limitation, the proposed framework uses multiple models, allowing each node to construct a personalized model. The proposed MASS procedure uses self-supervised signals to identify suitable experts even when only unlabeled local data is available. Experiments show that the framework remains robust under challenging heterogeneous settings, including pathological non-IID scenarios on CIFAR10.

**[Access the research paper](https://www.sciencedirect.com/science/article/pii/S0167739X25000937?casa_token=Rbeqvg3f8pQAAAAA:f5uRaHpClEz6Nmjbnruh4eYNqjuGsZCv3BuOuHew6EQgLERx8ztQkRxHYIWRLQo4MSzNmem9NJw)**

### Key Contributions
- Introduces a decentralized MoE-based federated learning framework that preserves client specialization instead of forcing all local knowledge into one global model.
- Proposes MASS, a self-supervised expert selection strategy for unlabeled local data, combining expert comparison, Bayesian search, and adaptive training control.
- Demonstrates strong robustness in difficult distributed environments, outperforming multiple federated learning baselines under severe non-IID settings.

### Architecture
Figure link

- **Expert trainer node**: Responsible for training and distributing experts, and is assumed to have access to only a portion of labeled data.
- **User**: Collects the distributed experts and builds a Mixture-of-Experts model, and is assumed to have access only to unlabeled data.

### Proposed Algorithm
Figure link

- An algorithm based on Bayesian optimization and self-supervised learning.

## Performance under data distribution

| :---: | :---: | :---: |
| ![Model Poisoning Accuracy](https://github.com/user-attachments/assets/d8b90c7b-7c32-485f-85af-0f98ee621bd5) | ![Data Poisoning Accuracy](https://github.com/user-attachments/assets/9e24dbde-eea3-499e-8409-73aaa81d9b74) | ![Label Swapping Accuracy](https://github.com/user-attachments/assets/452533ff-879b-4118-9465-ef3423f07330) |
| **IID** | **Non-IID** | **Pathological non-IID** |

### Data settings
- **IID**: Data is evenly distributed across clients using a normal distribution.
- **Non-IID**: Each client is biased toward two randomly selected classes, while samples from the remaining classes are assigned in small proportions according to a Dirichlet distribution.
- **Pathological non-IID**: Each client contains data from only two randomly selected classes.

## Code Explanation
### Contents
The uploaded code is a simplified version of the experimental simulator used in the paper.

- **FLIM_discrete**: A tick-based discrete simulator that generates events based on a Poisson process.
- **FLIM_continuous**: A SimPy-based continuous simulator that generates events based on an exponential process.
- **GateModule**: The gating module for the Mixture-of-Experts model.
- **PASS**: Includes the MASS algorithm and CLA.

### How to Run

```bash
pip install -r requirement.txt

# The execution method is the same for discrete, continuous, and integrate
cd FLIM_continuous

# Generate example data files
python3 example_data_setup.py

# Run the main script
python3 main.py --n_node 10 --lamb 6 --sim_th 0.6