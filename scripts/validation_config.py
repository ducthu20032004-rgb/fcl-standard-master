from __future__ import annotations

from collections import OrderedDict

DEFAULT_OUTPUT_ROOT = "scripts/outputs"

# Validation protocol
GENERATOR_ONLY_SEEDS = list(range(20))
FULL_SEEDS = list(range(5))

# Core validation datasets from the plan.
VALIDATION_DATASETS_CORE = [
    "cifar100",
    "clinc150",
    "covertype",
    "uci_har",
]

INTERACTION_DATASETS = [
    "cifar100",
    "clinc150",
    "covertype",
]

FULL_MAIN_DATASETS = [
    "cifar100",
    "tinyimagenet200",
    "clinc150",
    "banking77",
    "covertype",
    "uci_har",
]

STRESS_DATASETS = [
    "imagenet1k",
]

# Validation baseline panel suggested in the note.
# Map display name -> repo method id.
BASELINE_PANEL = OrderedDict(
    [
        ("FedAvg-FT", "fedavg"),
        ("DER++-Fed", "fedderpp"),
        ("FedWeIT", "fedweit"),
        ("FCIL/GLFC", "glfc"),
        ("TARGET", "target"),
        ("AF-FCL", "af_fcl"),
    ]
)

# Fallback panel using common methods that are more likely to exist in your repo.
FALLBACK_BASELINE_PANEL = OrderedDict(
    [
        ("FedAvg", "fedavg"),
        ("FedProx", "fedprox"),
        ("FedDER++", "fedderpp"),
        ("FedEWC", "fedewc"),
        ("FedLwF", "fedlwf"),
        ("TARGET", "target"),
        ("FedALA", "fedala"),
        ("FedAS", "fedas"),
        ("FedL2P", "fedl2p"),
        ("TagFed", "tagfed"),
    ]
)

REGIME_GRID = OrderedDict(
    [
        ("mild", {"alpha": 1.0, "psi": 0.0}),
        ("skew-only", {"alpha": 0.05, "psi": 0.0}),
        ("order-only", {"alpha": 1.0, "psi": 0.75}),
        ("joint-hard", {"alpha": 0.05, "psi": 0.75}),
    ]
)

INTERACTION_ALPHA_GRID = [1.0, 0.3, 0.05]
INTERACTION_PSI_GRID = [0.0, 0.35, 0.75]

ALPHA_SWEEP = [0.05, 0.1, 0.3, 1.0, 10.0]
PSI_SWEEP = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]

LAMBDA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
MATCH_TOLERANCE = 0.03

DEFAULT_NUM_CLIENTS = 5
DEFAULT_CLIENT_FRACTION = 1.0
DEFAULT_ROUNDS_PER_TASK = 50
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_NUM_WORKERS = 0

DATASET_DEFAULT_CLASSES_PER_TASK = {
    "cifar10": 2,
    "cifar100": 10,
    "tinyimagenet200": 10,
    "clinc150": 10,
    "banking77": 7,
    "covertype": 1,
    "uci_har": 1,
    "imagenet1k": 10,
}

DATASET_DEFAULT_BATCH_SIZE = {
    "cifar10": 64,
    "cifar100": 128,
    "tinyimagenet200": 128,
    "clinc150": 32,
    "banking77": 32,
    "covertype": 256,
    "uci_har": 256,
    "imagenet1k": 256,
}

DATASET_DEFAULT_EVAL_BATCH_SIZE = {
    "cifar10": 256,
    "cifar100": 256,
    "tinyimagenet200": 256,
    "clinc150": 128,
    "banking77": 128,
    "covertype": 512,
    "uci_har": 512,
    "imagenet1k": 256,
}