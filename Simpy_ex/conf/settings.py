# ============================================================
# FLIM_Simpy Configuration
# SimPy process-based federated learning simulation settings
# ============================================================

# ----------- System parameters ----------- #
LOG_DIR = "./runs_simpy"

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# ----------- SimPy Simulation parameters ----------- #
SIMULATION_TIME = 15.0          # Total continuous simulation time (seconds)
EVENT_BATCH_WINDOW = 0.01       # Events within this window (seconds) are batched together
                                # This mimics near-simultaneous arrivals being processed together

# ----------- Node parameters ----------- #
CLS_EPOCH = 3                   # Classification training epochs per participation
ROT_EPOCH = 2                   # Rotation pretext task training epochs
LEARNING_RATE = 1e-4            # Adam optimizer learning rate
CHANNEL_SIZE = 3                # RGB image channels
BATCH_SIZE = 64                 # Training batch size
SUP_OTHER_MODEL_SIZE = 10       # Max stored expert models per node

# ----------- Poisson Process parameters ----------- #
# In the SimPy version, events arrive via exponential inter-arrival times.
# The rate parameter (lambda) controls how frequently a node participates.
# Expected inter-arrival time = 1/lambda
# Default lambda=6 means ~6 events in SIMULATION_TIME for normal nodes
# Straggler lambda=1 means ~1 event in SIMULATION_TIME
DEFAULT_LAMBDA = 6              # Default Poisson rate (events per SIMULATION_TIME)
STRAGGLER_LAMBDA = 1            # Straggler Poisson rate
