# ----------- System parameters ----------- #
LOG_DIR = "./runs"

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# LABELS = ['apple', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl',
#     'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair',
#     'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin']
# LABELS = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']


# ----------- Network parameters ----------- #
SIMULATION_TIME = 15 # 1000 sec

# ----------- Node parameters ----------- #
CLS_EPOCH = 3

ROT_EPOCH = 2

LEARNING_RATE = 1e-4

CHANNEL_SIZE = 3

BATCH_SIZE = 64

SUP_OTHER_MODEL_SIZE = 10







