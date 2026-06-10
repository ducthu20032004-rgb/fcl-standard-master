# Federated Continual Learning Benchmark

A clean, extensible, argument-driven benchmark for:

- **Federated Class-Incremental Learning (Class-IL)**
- **Federated Task-Incremental Learning (Task-IL)**
- **Federated Domain-Incremental Learning (Domain-IL)**
- **Personalized Federated Continual Learning**
- **Replay-, regularization-, distillation-, and adaptive-aggregation-based methods**

The repository is intentionally designed around a few simple principles:

- one main entry point: `main.py`
- no YAML/config files
- explicit **server/client** method split
- explicit **dataset / backbone / evaluation** modules
- clean local logging and figure generation
- easy addition of new methods and new benchmark scenarios

---

# Table of Contents

1. [Overview](#overview)  
2. [Main Goals of the Repository](#main-goals-of-the-repository)  
3. [Repository Structure](#repository-structure)  
4. [Supported Scenarios](#supported-scenarios)  
5. [Implemented Datasets](#implemented-datasets)  
6. [Implemented Methods](#implemented-methods)  
7. [Installation](#installation)  
8. [Data Preparation](#data-preparation)  
9. [Quick Start](#quick-start)  
10. [How Scenarios Work](#how-scenarios-work)  
11. [Metrics and Outputs](#metrics-and-outputs)  
12. [Important Arguments](#important-arguments)  
13. [Recommended Experimental Workflow](#recommended-experimental-workflow)  
14. [How to Add a New Dataset](#how-to-add-a-new-dataset)  
15. [How to Add a New Method](#how-to-add-a-new-method)  
16. [How to Add a New Backbone](#how-to-add-a-new-backbone)  
17. [Practical Advice for Reliable Benchmarking](#practical-advice-for-reliable-benchmarking)  
18. [Troubleshooting](#troubleshooting)  
19. [Current Scope and Caveats](#current-scope-and-caveats)  
20. [Citation / Acknowledgment Placeholder](#citation--acknowledgment-placeholder)  

---

# Overview

This repository is a modular benchmark for federated continual learning research.

It supports three benchmark scenarios:

- **Class-IL**: tasks are class splits, and evaluation uses the full classifier over all classes.
- **Task-IL**: tasks are still class splits, but evaluation uses a task oracle, meaning prediction is restricted to the classes of the current task.
- **Domain-IL**: tasks are domains, while the class space stays shared across tasks.

The codebase began with CIFAR-based Class-IL experiments and has been extended to support:

- local continual-learning baselines aggregated with FedAvg
- personalized federated continual methods
- replay and distillation methods
- repeated-task streams for specific settings such as traceable continual learning
- PACS as a domain-incremental benchmark dataset

Everything is controlled through command-line arguments and executed from `main.py`.

---

# Main Goals of the Repository

The repository tries to balance four things at the same time:

## 1. Clarity
You should be able to read the code and understand where each part lives:
- dataset creation
- partitioning
- method-specific client logic
- method-specific server logic
- metric computation
- plotting and logging

## 2. Extensibility
You should be able to add:
- a new dataset
- a new scenario
- a new method
- a new backbone

without rewriting the whole codebase.

## 3. Benchmark consistency
Different methods should run under the same high-level protocol whenever possible:
- same task construction
- same partitioning logic
- same output format
- same metrics

## 4. Practical usability
The code should be easy to run and easy to debug:
- all outputs saved locally
- no hidden config layers
- explicit arguments
- readable experiment folders

---

# Repository Structure

```text
.
├── data/
├── datasets/
│   ├── __init__.py
│   ├── base.py
│   ├── cifar.py
│   ├── pacs.py
│   ├── partitioners.py
│   └── registry.py
├── methods/
│   ├── __init__.py
│   ├── registry.py
│   ├── fedl2p_modules.py
│   ├── tagfed_modules.py
│   ├── target_modules.py
│   ├── client/
│   │   ├── __init__.py
│   │   ├── base_client.py
│   │   ├── common.py
│   │   ├── fedavg_client.py
│   │   ├── fedprox_client.py
│   │   ├── fedala_client.py
│   │   ├── fedas_client.py
│   │   ├── fedl2p_client.py
│   │   ├── target_client.py
│   │   ├── tagfed_client.py
│   │   ├── fedewc_client.py
│   │   ├── fedlwf_client.py
│   │   └── fedderpp_client.py
│   └── server/
│       ├── __init__.py
│       ├── base_server.py
│       ├── fedavg_server.py
│       ├── fedprox_server.py
│       ├── fedala_server.py
│       ├── fedas_server.py
│       ├── fedl2p_server.py
│       ├── target_server.py
│       └── tagfed_server.py
├── backbones/
│   ├── __init__.py
│   ├── registry.py
│   ├── resnet.py
│   └── resnet_imagenet.py
├── evaluations/
│   ├── __init__.py
│   ├── metrics.py
│   ├── tracker.py
│   └── plots.py
├── utils/
│   ├── __init__.py
│   ├── io.py
│   ├── logger.py
│   ├── misc.py
│   └── seed.py
├── scripts/
│   └── download_pacs.py
├── outputs/
├── .gitignore
├── main.py
└── requirements.txt
```

---

# Supported Scenarios

The benchmark currently supports three scenarios.

## 1. Class-IL
In Class-IL:
- tasks are built by splitting the global class order into chunks
- the classifier head remains global
- evaluation is done over the **full label space**

Example:
- CIFAR-100 with `classes_per_task = 10`
- total classes = 100
- total tasks = 10
- task 0 = 10 classes
- task 1 = another 10 classes
- and so on

This is the default scenario in many class-continual benchmarks.

---

## 2. Task-IL
In Task-IL:
- tasks are also built from class chunks
- the difference from Class-IL is in **evaluation**
- prediction is restricted to the classes of the current task

This means the benchmark assumes a task oracle at evaluation time.

In practice:
- training still uses the same class-based task construction
- testing masks logits to the classes of the target task

This is the cleanest and most standard way to implement Task-IL on top of the same class-partitioned benchmark.

---

## 3. Domain-IL
In Domain-IL:
- tasks are domains
- classes remain shared across all tasks
- each task corresponds to one domain

This is useful for benchmarks such as PACS, where:
- the label space is fixed
- the data distribution changes by domain

Example for PACS:
- task 0 = art_painting
- task 1 = cartoon
- task 2 = photo
- task 3 = sketch

### Repeated domain streams
The code also supports repeated domain streams:
- if `num_tasks > number_of_domains`
- the benchmark extends the client stream by repeating domain task IDs

This is useful if you want to study:
- revisiting domains
- cyclic domain exposure
- repeated-domain adaptation

Importantly, repeated domains are handled as **stream revisits**, not as fake duplicated new tasks.  
This keeps evaluation and forgetting logic much cleaner.

---

# Implemented Datasets

## CIFAR-10
A standard small-scale image benchmark.

Typical usage:
- Class-IL with 5 tasks × 2 classes
- Task-IL with the same task split

## CIFAR-100
The main image benchmark in the current codebase.

Typical usage:
- Class-IL with 10 tasks × 10 classes
- Task-IL with 10 tasks × 10 classes

## PACS
PACS is a domain generalization benchmark and is used here for **Domain-IL**.

PACS domains:
- art_painting
- cartoon
- photo
- sketch

PACS classes are shared across domains.

The benchmark loader assumes PACS is stored under:

```text
data/pacs/
├── art_painting/
├── cartoon/
├── photo/
└── sketch/
```

where each domain folder contains class folders.

PACS uses:
- image resizing
- ImageNet normalization
- an ImageNet-style backbone by default

---

# Implemented Methods

The repository currently includes several method families.

## A. Basic federated optimization baselines
- `fedavg`
- `fedprox`

Use these as optimization-only baselines.

## B. Local continual-learning wrappers + FedAvg aggregation
These methods do continual learning locally, but still aggregate client models with FedAvg.

- `fedewc`
- `fedlwf`
- `fedderpp`

This family is very useful when you want:
> federated learning on the outside, continual learning on the inside

## C. Personalized / adaptive / replay / distillation methods
- `fedala`
- `fedas`
- `fedl2p`
- `target`
- `tagfed`

These methods need additional client/server logic beyond plain FedAvg.

## Important implementation note
Some methods in this repository are:
- faithful implementations of the main training idea
- benchmark-friendly adaptations of papers that originally used more specialized code

The main design target of this repository is:
- methodological comparability
- structural clarity
- ease of extension

rather than reproducing every engineering detail of every original code release.

---

# Installation

## Step 1: Create environment

```bash
conda create -n fcl python=3.10 -y
conda activate fcl
```

You can also use `venv` if preferred.

## Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Verify the installation

```bash
python main.py --help
```

If that works, your environment is ready.

---

# Data Preparation

## CIFAR-10 / CIFAR-100
These datasets are handled directly by the benchmark through torchvision or the built-in tensor cache logic.

Usually you only need:

```bash
python main.py --dataset cifar100 --method fedavg ...
```

and the dataset will be downloaded automatically if needed.

## PACS
PACS should be exported into `data/pacs/`.

Use:

```bash
python scripts/download_pacs.py
```

This helper script downloads PACS and exports the folder structure expected by `datasets/pacs.py`.

If you want to overwrite an existing export:

```bash
python scripts/download_pacs.py --overwrite
```

If you want strict integrity checks:

```bash
python scripts/download_pacs.py --strict-checks
```

### Note
The PACS helper script uses the Hugging Face `datasets` package.  
If it is not already installed in your environment, install it with:

```bash
pip install datasets pillow
```

---

# Quick Start

## 1. Class-IL on CIFAR-100 with FedAvg

```bash
python main.py \
  --dataset cifar100 \
  --scenario class-il \
  --method fedavg \
  --backbone cifar_resnet18 \
  --seed 2023 \
  --num-clients 5 \
  --classes-per-task 10 \
  --dirichlet-alpha 1.0 \
  --order-psi 0.3 \
  --task-label-order random \
  --dirichlet-allocation floor_remainder \
  --schedule-swap-mode scan \
  --rounds-per-task 50 \
  --client-fraction 1.0 \
  --local-epochs 1 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --lr 0.05 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --loss-mode full \
  --use-cifar100-tensor-cache true
```

## 2. Task-IL on CIFAR-100 with FedDER++

```bash
python main.py \
  --dataset cifar100 \
  --scenario task-il \
  --method fedderpp \
  --backbone cifar_resnet18 \
  --seed 2023 \
  --num-clients 5 \
  --classes-per-task 10 \
  --dirichlet-alpha 1.0 \
  --order-psi 0.3 \
  --task-label-order random \
  --dirichlet-allocation floor_remainder \
  --schedule-swap-mode scan \
  --rounds-per-task 50 \
  --client-fraction 1.0 \
  --local-epochs 1 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --lr 0.05 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --loss-mode partial \
  --fedderpp-alpha 0.5 \
  --fedderpp-beta 0.5 \
  --fedderpp-buffer-size 500 \
  --fedderpp-store-per-task 200 \
  --fedderpp-replay-batch-size 64 \
  --use-cifar100-tensor-cache true
```

## 3. Domain-IL on PACS with FedAvg

```bash
python main.py \
  --dataset pacs \
  --scenario domain-il \
  --method fedavg \
  --backbone resnet18_imagenet \
  --use-pretrained-backbone true \
  --seed 2023 \
  --num-clients 5 \
  --num-tasks 4 \
  --dirichlet-alpha 0.5 \
  --order-psi 0.25 \
  --task-label-order random \
  --dirichlet-allocation floor_remainder \
  --schedule-swap-mode scan \
  --rounds-per-task 80 \
  --client-fraction 1.0 \
  --local-epochs 3 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --num-workers 4 \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --loss-mode full \
  --pacs-image-size 224 \
  --pacs-test-ratio 0.1 \
  --pacs-split-seed 2023
```

## 4. Domain-IL with repeated domain stream on PACS

```bash
python main.py \
  --dataset pacs \
  --scenario domain-il \
  --method fedavg \
  --backbone resnet18_imagenet \
  --use-pretrained-backbone true \
  --seed 2023 \
  --num-clients 5 \
  --num-tasks 8 \
  --domain-repeat-mode cycle \
  --dirichlet-alpha 0.5 \
  --order-psi 0.25 \
  --task-label-order random \
  --dirichlet-allocation floor_remainder \
  --schedule-swap-mode scan \
  --rounds-per-task 60 \
  --client-fraction 1.0 \
  --local-epochs 3 \
  --batch-size 64 \
  --eval-batch-size 128 \
  --num-workers 4 \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 5e-4 \
  --loss-mode full \
  --pacs-image-size 224 \
  --pacs-test-ratio 0.1 \
  --pacs-split-seed 2023
```

---

# How Scenarios Work

## Class-IL internals
- build a global class order
- split classes into tasks
- assign client data via label partitioning
- create client-specific task order
- evaluate over the full classifier head

## Task-IL internals
- same class-task construction as Class-IL
- same client partitioning
- evaluation applies task masking to logits

This means:
- the benchmark structure is shared
- only the testing assumption changes

## Domain-IL internals
- task IDs come from dataset domain IDs
- all tasks share the same class space
- test subsets are domain-based
- client data are partitioned within each domain
- task order is applied over domain tasks
- optional stream repetition revisits domains if requested

---

# Metrics and Outputs

Each run writes outputs into:

```text
outputs/<dataset>/<method>/<setting>/<time_id>/
```

Typical content:

```text
outputs/<dataset>/<method>/<setting>/<time_id>/
├── args.json
├── logs/
│   └── run.log
├── tables/
│   ├── partition_summary.json
│   ├── round_metrics.csv
│   ├── task_accuracy_history.csv
│   └── summary.json
└── figures/
    ├── avg_acc_per_round.png
    ├── forgetting_per_round.png
    └── local_global_gap_per_round.png
```

## Main metrics

### `avg_acc`
Client-first average accuracy over tasks a client has seen.

### `forgetting`
Average forgetting over previously seen tasks.

### `local_global_gap`
Difference between the local/personalized model and the global model.

This is especially useful for:
- personalized methods
- replay methods
- local continual-learning baselines

---

# Important Arguments

## Global benchmark setup
- `--dataset`
- `--scenario`
- `--method`
- `--backbone`
- `--seed`

## Federated setup
- `--num-clients`
- `--client-fraction`
- `--rounds-per-task`
- `--local-epochs`

## Task construction
- `--classes-per-task`
- `--task-label-order`
- `--num-tasks` (especially important for Domain-IL stream length)

## Heterogeneity
- `--dirichlet-alpha`
- `--order-psi`
- `--dirichlet-allocation`
- `--schedule-swap-mode`

## Domain-IL specific
- `--domain-repeat-mode`
- `--pacs-image-size`
- `--pacs-test-ratio`
- `--pacs-split-seed`

## Optimization
- `--lr`
- `--momentum`
- `--weight-decay`
- `--batch-size`
- `--eval-batch-size`

## Replay / distillation / regularization specific
Method-specific arguments exist for:
- FedProx
- FedEWC
- FedLwF
- FedDER++
- FedALA
- FedAS
- FedL2P
- TARGET
- TagFed

---

# Recommended Experimental Workflow

A practical and efficient workflow is:

## Phase 1: Sanity checks
Run:
- FedAvg on CIFAR-10
- FedAvg on CIFAR-100

Goal:
- make sure data loading works
- make sure logging works
- make sure outputs look correct

## Phase 2: Basic federated baselines
Run:
- FedAvg
- FedProx

Goal:
- establish optimization-only references

## Phase 3: Local continual-learning baselines
Run:
- FedEWC
- FedLwF
- FedDER++

Goal:
- compare regularization vs distillation vs replay under the same FedAvg server

## Phase 4: More advanced methods
Run:
- FedALA
- FedAS
- FedL2P
- TARGET
- TagFed

Goal:
- test personalization, replay synthesis, traceability, and grouped aggregation

## Phase 5: Scenario expansion
Repeat experiments under:
- Class-IL
- Task-IL
- Domain-IL

Goal:
- understand which methods are robust across different continual-learning assumptions

---

# How to Add a New Dataset

A new dataset usually needs four things.

## Step 1
Create a builder file under `datasets/`.

Examples:
- `datasets/my_dataset.py`
- `datasets/my_timeseries.py`

## Step 2
Return a `DatasetBundle` with:
- train dataset
- test dataset
- train targets
- test targets
- number of classes
- class names
- optional task IDs if needed

## Step 3
If the dataset is naturally domain-based, also provide:
- `train_task_ids`
- `test_task_ids`
- `task_names`
- `default_scenario = "domain-il"`

## Step 4
Register it in `datasets/registry.py`.

---

# How to Add a New Method

A new method should usually include:

## Client side
Create:
```text
methods/client/<method>_client.py
```

## Server side
Create:
```text
methods/server/<method>_server.py
```

## Register it
Add the server/client pair in:
```text
methods/registry.py
```

## Practical rule
If the method is basically:
- local continual learning + standard averaging

then you usually only need a new client file and can reuse `FedAvgServer`.

If the method needs:
- replay generation
- grouped teachers
- server-side distillation
- task boundary hooks

then you likely need a new server file as well.

---

# How to Add a New Backbone

## Step 1
Create the backbone under `backbones/`.

## Step 2
Register it in `backbones/registry.py`.

## Step 3
Expose helper methods if advanced methods need them, such as:
- `extract_features(...)`
- `forward_from_features(...)`
- `backbone_state_dict()`
- `head_state_dict()`

These helpers are especially useful for:
- FedAS
- TARGET
- TagFed
- Domain-IL with pretrained image backbones

---

# Practical Advice for Reliable Benchmarking

## 1. Fix the seed
Always fix `--seed` for reproducibility.

## 2. Keep a method matrix
For serious studies, compare methods under the same:
- dataset
- scenario
- client count
- classes per task
- alpha
- psi
- rounds
- local epochs

## 3. Use the right loss mode
Recommended:
- `class-il`: usually `full`
- `task-il`: usually `partial`
- `domain-il`: usually `full`

## 4. Use the right backbone
Recommended:
- CIFAR: `cifar_resnet18`
- PACS: `resnet18_imagenet`

## 5. Start with small runs
Before launching large experiments:
- reduce rounds
- reduce local epochs
- test one seed
- test one method

## 6. Inspect outputs after every major change
Check:
- `run.log`
- `partition_summary.json`
- figures
- `round_metrics.csv`

---

# Troubleshooting

## Dataset not found
Make sure it is registered in `datasets/registry.py`.

## Method not found
Make sure it is registered in `methods/registry.py`.

## PACS loading fails
Check:
- `data/pacs/` exists
- domain folders are present
- class folders exist inside each domain

## Domain-IL gives strange results
Check:
- you are using `scenario domain-il`
- the dataset has `train_task_ids` and `test_task_ids`
- `num_tasks` is set correctly
- you are not accidentally using Task-IL assumptions

## Task-IL accuracy looks too low
Make sure:
- `scenario` is set to `task-il`
- `loss-mode` is `partial`
- evaluation masking is active

## Global and local metrics look identical
This may happen if:
- the method is not actually personalizing
- local states are overwritten by global states
- the method is effectively shared-only

## PACS runs are too slow
Try:
- smaller image size (only for debugging)
- fewer rounds
- fewer local epochs
- fewer workers if your storage is slow
