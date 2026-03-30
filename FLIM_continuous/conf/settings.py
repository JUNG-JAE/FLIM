# ============================================================
# FLIM_Simpy Configuration  (Fully Asynchronous)
#
# No rounds, no ticks.  Each node independently generates
# events via exponential inter-arrival times.
# ============================================================

# ----------- System ----------- #
LOG_DIR = "./runs_simpy"

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# ----------- Simulation ----------- #
SIMULATION_TIME = 15.0          # continuous simulation duration (seconds)

# ----------- Node ----------- #
CLS_EPOCH = 3
ROT_EPOCH = 2
LEARNING_RATE = 1e-4
CHANNEL_SIZE = 3
BATCH_SIZE = 64
SUP_OTHER_MODEL_SIZE = 10       # max expert models per node

# ----------- Poisson process ----------- #
# rate per node = lambda / SIMULATION_TIME  (events per unit time)
# mean inter-arrival = SIMULATION_TIME / lambda
DEFAULT_LAMBDA = 6              # normal nodes: ~6 events in total
STRAGGLER_LAMBDA = 1            # straggler nodes: ~1 event in total
