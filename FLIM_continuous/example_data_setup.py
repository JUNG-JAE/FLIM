"""
CIFAR-10 Example Data Setup for FLIM (Federated Learning Simulation)
=====================================================================
이 스크립트는 FLIM 시뮬레이션 실행에 필요한 데이터셋을 준비합니다.

- CIFAR-10 데이터를 다운로드 (없는 경우)
- 10개 노드(node0~node9)에 Non-IID 방식으로 데이터 분배
  (각 노드는 랜덤하게 선택된 2개의 클래스만 보유)
- 테스트 데이터셋 준비 (전체 10개 클래스)
- Rotation pretext task용 데이터 준비

생성되는 디렉토리 구조:
  ~/workspace/DataSet/processing_data/cifar10_example/node{0-9}/train/{class_name}/*.png
  ~/workspace/DataSet/genesis_data/CIFAR10_64/test/{class_name}/*.png
  ~/workspace/DataSet/rotation_data/cifar10_example/node{0-9}/train/{0,90,180,270}/*.png
  ~/workspace/DataSet/rotation_data/cifar10_example/node{0-9}/test/{0,90,180,270}/*.png
"""

import os
import random
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as transforms

# ===== Configuration ===== #
NUM_NODES = 10
LABELS_PER_NODE = 2
CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
ROTATION_ANGLES = [0, 90, 180, 270]
IMAGE_SIZE = 64  # CIFAR10_64 -> 64x64로 리사이즈

BASE_DIR = os.path.expanduser("~/workspace/DataSet")
PROCESSING_DATA_DIR = os.path.join(BASE_DIR, "processing_data", "cifar10_example")
GENESIS_DATA_DIR = os.path.join(BASE_DIR, "genesis_data", "CIFAR10_64")
ROTATION_DATA_DIR = os.path.join(BASE_DIR, "rotation_data", "cifar10_example")
CIFAR10_RAW_DIR = os.path.join(BASE_DIR, "raw", "cifar10")


def download_cifar10():
    """CIFAR-10 데이터를 다운로드합니다 (이미 존재하면 스킵)."""
    print("[1/5] CIFAR-10 데이터 다운로드 중...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=CIFAR10_RAW_DIR, train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=CIFAR10_RAW_DIR, train=False, download=True
    )
    print(f"  - 학습 데이터: {len(train_dataset)}장")
    print(f"  - 테스트 데이터: {len(test_dataset)}장")
    return train_dataset, test_dataset


def assign_labels_to_nodes():
    """
    각 노드에 랜덤하게 2개의 레이블을 할당합니다 (Non-IID).
    모든 레이블이 최소 1개 노드에 할당되도록 보장합니다.
    """
    print("[2/5] Non-IID 레이블 할당 중...")
    label_indices = list(range(len(CIFAR10_LABELS)))

    # 모든 레이블이 최소 한 번은 할당되도록 셔플 후 분배
    shuffled = label_indices.copy()
    random.shuffle(shuffled)

    # 10개 레이블을 5쌍으로 나누면 모든 레이블이 커버됨
    node_labels = {}
    for i in range(NUM_NODES):
        if i < 5:
            # 처음 5개 노드: 셔플된 레이블에서 2개씩 순차 할당
            node_labels[f"node{i}"] = sorted([shuffled[2 * i], shuffled[2 * i + 1]])
        else:
            # 나머지 5개 노드: 랜덤하게 2개 선택
            chosen = sorted(random.sample(label_indices, LABELS_PER_NODE))
            node_labels[f"node{i}"] = chosen

    for node_id, labels in node_labels.items():
        label_names = [CIFAR10_LABELS[l] for l in labels]
        print(f"  - {node_id}: {label_names}")

    return node_labels


def save_image(img, path):
    """PIL Image를 64x64로 리사이즈하여 PNG로 저장합니다."""
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    img_resized.save(path)


def setup_training_data(train_dataset, node_labels):
    """
    각 노드의 학습 데이터를 Non-IID로 분배합니다.
    경로: ~/workspace/DataSet/processing_data/cifar10_example/node{i}/train/{class_name}/*.png
    """
    print("[3/5] 학습 데이터 분배 중...")

    # 클래스별 인덱스 분류
    class_indices = {i: [] for i in range(len(CIFAR10_LABELS))}
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)

    for node_id, labels in node_labels.items():
        for label_idx in labels:
            class_name = CIFAR10_LABELS[label_idx]
            save_dir = os.path.join(PROCESSING_DATA_DIR, node_id, "train", class_name)
            os.makedirs(save_dir, exist_ok=True)

            indices = class_indices[label_idx]
            # 각 노드에 해당 클래스의 데이터를 균등 분배
            # 같은 레이블을 가진 노드 수 계산
            nodes_with_this_label = [n for n, lbls in node_labels.items() if label_idx in lbls]
            node_position = nodes_with_this_label.index(node_id)
            total_nodes_for_label = len(nodes_with_this_label)

            # 데이터를 균등 분할
            chunk_size = len(indices) // total_nodes_for_label
            start = node_position * chunk_size
            end = start + chunk_size if node_position < total_nodes_for_label - 1 else len(indices)
            node_indices = indices[start:end]

            for count, data_idx in enumerate(node_indices):
                img, _ = train_dataset[data_idx]
                img_path = os.path.join(save_dir, f"{class_name}_{count:05d}.png")
                save_image(img, img_path)

            print(f"  - {node_id}/{class_name}: {len(node_indices)}장 저장")


def setup_test_data(test_dataset):
    """
    테스트 데이터를 전체 클래스에 대해 준비합니다.
    경로: ~/workspace/DataSet/genesis_data/CIFAR10_64/test/{class_name}/*.png
    """
    print("[4/5] 테스트 데이터 준비 중...")
    test_dir = os.path.join(GENESIS_DATA_DIR, "test")

    for label_idx in range(len(CIFAR10_LABELS)):
        class_name = CIFAR10_LABELS[label_idx]
        save_dir = os.path.join(test_dir, class_name)
        os.makedirs(save_dir, exist_ok=True)

    count_per_class = {i: 0 for i in range(len(CIFAR10_LABELS))}
    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        class_name = CIFAR10_LABELS[label]
        save_dir = os.path.join(test_dir, class_name)
        img_path = os.path.join(save_dir, f"{class_name}_{count_per_class[label]:05d}.png")
        save_image(img, img_path)
        count_per_class[label] += 1

    total = sum(count_per_class.values())
    print(f"  - 총 {total}장 저장 (클래스별 ~{total // len(CIFAR10_LABELS)}장)")


def setup_rotation_data(train_dataset, test_dataset, node_labels):
    """
    Rotation pretext task용 데이터를 준비합니다.
    각 이미지를 0°, 90°, 180°, 270°로 회전하여 저장합니다.
    경로: ~/workspace/DataSet/rotation_data/cifar10_example/node{i}/{train,test}/{angle}/*.png
    """
    print("[5/5] Rotation 데이터 준비 중...")

    # 클래스별 인덱스 분류 (학습 데이터)
    class_indices = {i: [] for i in range(len(CIFAR10_LABELS))}
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)

    for node_id, labels in node_labels.items():
        # --- Train rotation data ---
        for angle in ROTATION_ANGLES:
            save_dir = os.path.join(ROTATION_DATA_DIR, node_id, "train", str(angle))
            os.makedirs(save_dir, exist_ok=True)

        img_count = 0
        for label_idx in labels:
            indices = class_indices[label_idx]
            nodes_with_this_label = [n for n, lbls in node_labels.items() if label_idx in lbls]
            node_position = nodes_with_this_label.index(node_id)
            total_nodes_for_label = len(nodes_with_this_label)

            chunk_size = len(indices) // total_nodes_for_label
            start = node_position * chunk_size
            end = start + chunk_size if node_position < total_nodes_for_label - 1 else len(indices)
            node_indices = indices[start:end]

            # Rotation 데이터는 일부만 사용 (전체의 20%)
            sample_size = max(1, len(node_indices) // 5)
            sampled = random.sample(node_indices, sample_size)

            for data_idx in sampled:
                img, _ = train_dataset[data_idx]
                img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                for angle in ROTATION_ANGLES:
                    rotated = img_resized.rotate(-angle)  # PIL rotate는 반시계방향
                    save_dir = os.path.join(ROTATION_DATA_DIR, node_id, "train", str(angle))
                    img_path = os.path.join(save_dir, f"rot_{img_count:05d}.png")
                    rotated.save(img_path)
                img_count += 1

        # --- Test rotation data ---
        for angle in ROTATION_ANGLES:
            save_dir = os.path.join(ROTATION_DATA_DIR, node_id, "test", str(angle))
            os.makedirs(save_dir, exist_ok=True)

        test_sample_size = min(200, len(test_dataset))
        test_indices = random.sample(range(len(test_dataset)), test_sample_size)

        for count, data_idx in enumerate(test_indices):
            img, _ = test_dataset[data_idx]
            img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
            for angle in ROTATION_ANGLES:
                rotated = img_resized.rotate(-angle)
                save_dir = os.path.join(ROTATION_DATA_DIR, node_id, "test", str(angle))
                img_path = os.path.join(save_dir, f"rot_{count:05d}.png")
                rotated.save(img_path)

        print(f"  - {node_id}: train={img_count}장x4회전, test={test_sample_size}장x4회전")


def print_summary(node_labels):
    """설정 요약을 출력합니다."""
    print("\n" + "=" * 60)
    print("데이터 설정 완료!")
    print("=" * 60)
    print(f"\n[디렉토리 구조]")
    print(f"  학습 데이터: {PROCESSING_DATA_DIR}/")
    print(f"  테스트 데이터: {GENESIS_DATA_DIR}/test/")
    print(f"  회전 데이터: {ROTATION_DATA_DIR}/")
    print(f"\n[노드별 레이블 할당 (Non-IID)]")
    for node_id, labels in node_labels.items():
        label_names = [CIFAR10_LABELS[l] for l in labels]
        print(f"  {node_id}: {label_names}")
    print(f"\n이제 main.py를 실행할 수 있습니다:")
    print(f"  python main.py --dataset cifar10_example")
    print("=" * 60)


def main():
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("FLIM - CIFAR-10 Example Data Setup")
    print("=" * 60 + "\n")

    # 1. CIFAR-10 다운로드
    train_dataset, test_dataset = download_cifar10()

    # 2. 노드별 레이블 할당 (Non-IID: 각 노드 2개 클래스)
    node_labels = assign_labels_to_nodes()

    # 3. 학습 데이터 분배
    setup_training_data(train_dataset, node_labels)

    # 4. 테스트 데이터 준비
    setup_test_data(test_dataset)

    # 5. Rotation 데이터 준비
    setup_rotation_data(train_dataset, test_dataset, node_labels)

    # 요약 출력
    print_summary(node_labels)


if __name__ == "__main__":
    main()
