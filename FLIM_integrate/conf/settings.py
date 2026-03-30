# ============================================================
# FLIM_integrate Configuration
# Continuous SimPy-based DFL + MASS (PASS) + GateModule
# ============================================================

# ----------- System ----------- #
LOG_DIR = "./runs_integrate"

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

# ----------- Simulation ----------- #
SIMULATION_TIME = 15.0          # continuous simulation duration

# ----------- Node (DE process) ----------- #
CLS_EPOCH = 3
ROT_EPOCH = 2
LEARNING_RATE = 1e-4
CHANNEL_SIZE = 3
BATCH_SIZE = 64
SUP_OTHER_MODEL_SIZE = 10       # max expert models per node (|C|)

# ----------- Poisson process ----------- #
DEFAULT_LAMBDA = 6
STRAGGLER_LAMBDA = 1

# ----------- Expert Aggregation (Algorithm 1) ----------- #
COSINE_THRESHOLD = 0.6          # τ: cosine distance threshold for aggregation

# ----------- MASS stability ----------- #
# After collecting >= EXPERT_STABLE_COUNT experts, if the cluster count
# remains the same for STABILITY_WINDOW more aggregation events,
# we consider the expert pool "stable" and trigger MASS + GateModule.
EXPERT_STABLE_COUNT = 10        # minimum experts before stability check
STABILITY_WINDOW = 3            # consecutive events with same cluster count

# ----------- MASS / Bayesian Optimization ----------- #
MASS_N_TRIALS = 10              # Bayesian optimization trials
MASS_N_TRAIN_MAX = 800          # max rotation train dataset size
MASS_N_TEST_MAX = 300           # max rotation test dataset size
MASS_EPOCH_MIN = 10
MASS_EPOCH_MAX = 50
MASS_SAMPLE_SIZE = 5            # last n epochs for mean accuracy
MASS_P_VALUE = 0.05             # t-test significance level

# ----------- CLA (Conditional Loss Adjustment) ----------- #
CLA_TAU_GAP = 0.4               # ξ: skewness gap threshold
CLA_DELTA = 0.95                # δ: Beta CDF threshold for reliable training

# ----------- GateModule ----------- #
GATE_EPOCH = 10                 # gating module training epochs
GATE_LEARNING_RATE = 1e-4
GATE_CONFIDENCE_LEVEL = 0.99   # softmax threshold for high-confidence samples
