# README_RUN.md

# Benchmark Run Plan

This document explains **what to run**, **who runs what**, and **why each run package is necessary** for the alpha-psi benchmark paper.

The benchmark is organized around a **main image-based Class-IL FCIL core** and two **appendix portability checks**:
- **Main benchmark**: image, Class-IL, cross-device-style FCIL
- **Appendix B**: text portability check
- **Appendix B**: graph portability check

The team runs are split **by method ownership**.  
Each person owns a set of methods and is responsible for running those methods on the required run packages.

---

# 1. Benchmark Scope

## 1.1 Main benchmark
- **Modality**: Image
- **Setting**: Class-IL
- **Federated regime**: Cross-device-style
- **Main datasets**:
  - CIFAR-100
  - TinyImageNet

## 1.2 Appendix portability checks
- **Text**:
  - 20NewsGroups
  - Class-IL
  - 10 tasks
- **Graph**:
  - Cora
  - Continual node classification
  - 7 stages

---

# 2. Shared Protocol

These settings must be kept fixed unless a method absolutely requires a method-specific exception.

## 2.1 Main image benchmark
- **Clients**: 20
- **Tasks / stages**: 10
- **Participation fraction**: 0.25
- **Local epochs**: 5
- **Rounds per stage**: 40
- **Backbone**: ResNet-18
- **Inference**: Class-IL, no task identity at test time
- **Metrics to save**:
  - AFA
  - AF
  - B10
- **Also save**:
  - per-stage accuracies
  - per-client final accuracies
  - realized heterogeneity audit
  - raw scores needed for ranking analyses in Section 5

## 2.2 Text appendix
- **Dataset**: 20NewsGroups
- **Setting**: Class-IL
- **Tasks**: 10 tasks, 2 classes per task
- **Clients**: 20
- **Participation fraction**: 0.25
- **Local epochs**: 5
- **Rounds per stage**: 40
- **Encoder / backbone**: frozen sentence embeddings + MLP
- **Metrics to save**:
  - AFA
  - AF
  - B10
  - realized heterogeneity audit

## 2.3 Graph appendix
- **Dataset**: Cora
- **Setting**: continual node classification
- **Stages**: 7
- **Clients**: 20
- **Participation fraction**: 0.25
- **Local epochs**: 5
- **Rounds per stage**: 40
- **Backbone**: 2-layer GCN
- **Metrics to save**:
  - AFA
  - AF
  - B10
  - realized heterogeneity audit

---

# 3. Alpha-Psi Grids

## 3.1 Main image grid
Use this for the main CIFAR-100 and TinyImageNet runs.

- **alpha**: {0.10, 10.00, 100.00}
- **psi**: {0.00, 0.25, 0.50, 0.75, 1.00}
- **main seeds**: {1, 2, 3}

This gives:
- 3 alpha levels
- 5 psi levels
- 3 seeds

So each main image method must cover:

- **15 alpha-psi cells**
- **3 seeds per cell**
- **45 runs per dataset**

Since there are 2 image datasets:
- **90 runs per image method** for the main grid

## 3.2 Extra endpoint seeds for RQ1
These are only for stronger rank-stability analysis in Section 5.1.

- **alpha**: {0.10, 10.00, 100.00}
- **psi**: {0.00, 1.00}
- **extra seeds**: {4, 5}

So each image method must additionally cover:
- 3 alpha levels
- 2 psi endpoints
- 2 extra seeds
- **12 extra runs per dataset**
- **24 extra runs per image method**

## 3.3 Text appendix reduced grid
Use a lighter grid because this is a portability check, not a full leaderboard.

- **alpha**: {0.10, 10.00, 100.00}
- **psi**: {0.00, 0.50, 1.00}
- **seeds**: {1, 2}

So each text method must cover:
- 3 alpha levels
- 3 psi levels
- 2 seeds
- **18 runs per text method**

## 3.4 Graph appendix reduced grid
Same philosophy as text appendix.

- **alpha**: {0.10, 10.00, 100.00}
- **psi**: {0.00, 0.50, 1.00}
- **seeds**: {1, 2}

So each graph method must cover:
- 3 alpha levels
- 3 psi levels
- 2 seeds
- **18 runs per graph method**

---

# 4. Main Comparable Method Pool

## 4.1 Main image methods
- Local-Only
- FedAvg-CL
- GLFC
- MFCL
- TARGET
- LANDER
- Re-Fed
- TagFed
- LGA
- FedCBDR

## 4.2 Appendix text method
- CFeD

## 4.3 Appendix graph methods
- POWER
- MOTION

---

# 5. Team Assignment by Method

Each person owns their methods across all required datasets and grids.

| Team member | Owned methods |
|---|---|
| **Thịnh** | Local-Only, LGA, CFed |
| **Dương** | FedAvg-CL, MFCL, LANDER |
| **Tuấn** | Re-Fed, FedCBDR, TagFed |
| **Nguyệt Anh** | TARGET, POWER |
| **Thảo Nhi** | GLFC, MOTION |

---

# 6. What Each Person Must Run

The key rule is simple:

- If a method belongs to the **main image pool**, run it on:
  - CIFAR-100 main grid
  - TinyImageNet main grid
  - CIFAR-100 extra endpoint seeds
  - TinyImageNet extra endpoint seeds

- If a method belongs to the **text appendix**, run it on:
  - 20NewsGroups reduced grid

- If a method belongs to the **graph appendix**, run it on:
  - Cora reduced grid

---

# 7. Person-by-Person Checklist

## 7.1 Thịnh
### Owned methods
- Local-Only
- LGA
- CFed

### Thịnh must run

#### For `Local-Only`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds
- [ ] 20NewsGroups reduced grid
- [ ] Cora reduced grid

#### For `LGA`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds

#### For `CFed`
- [ ] 20NewsGroups reduced grid

### Notes
- Local-Only is the anchor for collaboration gain and lower-bound behavior across all modalities.
- LGA is a main image benchmark method.
- CFed is used only for the text appendix portability check.

---

## 7.2 Dương
### Owned methods
- FedAvg-CL
- MFCL
- LANDER

### Dương must run

#### For `FedAvg-CL`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds
- [ ] 20NewsGroups reduced grid
- [ ] Cora reduced grid

#### For `MFCL`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds

#### For `LANDER`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds

### Notes
- FedAvg-CL is the federated naive control and must exist in all modalities.
- MFCL represents data-free generative replay.
- LANDER represents semantic data-free transfer.

---

## 7.3 Tuấn
### Owned methods
- Re-Fed
- FedCBDR
- TagFed

### Tuấn must run

#### For `Re-Fed`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds

#### For `FedCBDR`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds

#### For `TagFed`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds

### Notes
- Re-Fed and FedCBDR cover replay-oriented families inside the image benchmark.
- TagFed is the key order-aware method for the main benchmark.

---

## 7.4 Nguyệt Anh
### Owned methods
- TARGET
- POWER

### Nguyệt Anh must run

#### For `TARGET`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds

#### For `POWER`
- [ ] Cora reduced grid

### Notes
- TARGET is the exemplar-free distillation family representative.
- POWER is a graph-only appendix baseline.

---

## 7.5 Thảo Nhi
### Owned methods
- GLFC
- MOTION

### Thảo Nhi must run

#### For `GLFC`
- [ ] CIFAR-100 main grid
- [ ] TinyImageNet main grid
- [ ] CIFAR-100 extra endpoint seeds
- [ ] TinyImageNet extra endpoint seeds

#### For `MOTION`
- [ ] Cora reduced grid

### Notes
- GLFC is a canonical FCIL baseline and must be fully covered on the image benchmark.
- MOTION is a graph-only appendix baseline.

---

# 8. Run Package Format

A run package should be logged in text form like this:

## 8.1 Main image package format
- **Method**: `<method_name>`
- **Dataset**: `CIFAR-100` or `TinyImageNet`
- **Modality**: `image`
- **Setting**: `Class-IL`
- **Tasks**: `10`
- **Clients**: `20`
- **Participation**: `0.25`
- **Local epochs**: `5`
- **Rounds per stage**: `40`
- **Backbone**: `ResNet-18`
- **alpha**: `<0.10 or 10.00 or 100.00>`
- **psi**: `<0.00 / 0.25 / 0.50 / 0.75 / 1.00>`
- **Seed**: `<1 / 2 / 3 / 4 / 5>`
- **Outputs to save**:
  - AFA
  - AF
  - B10
  - per-stage task accuracies
  - per-client final accuracies
  - realized heterogeneity audit
  - raw outputs needed for ranking analyses

## 8.2 Text package format
- **Method**: `Local-Only`, `FedAvg-CL`, or `CFeD`
- **Dataset**: `20NewsGroups`
- **Modality**: `text`
- **Setting**: `Class-IL`
- **Tasks**: `10`
- **Clients**: `20`
- **Participation**: `0.25`
- **Local epochs**: `5`
- **Rounds per stage**: `40`
- **Backbone**: `frozen sentence embeddings + MLP`
- **alpha**: `<0.10 / 10.00 / 100.00>`
- **psi**: `<0.00 / 0.50 / 1.00>`
- **Seed**: `<1 / 2>`
- **Outputs to save**:
  - AFA
  - AF
  - B10
  - realized heterogeneity audit

## 8.3 Graph package format
- **Method**: `Local-Only`, `FedAvg-CL`, `POWER`, or `MOTION`
- **Dataset**: `Cora`
- **Modality**: `graph`
- **Setting**: `continual node classification`
- **Stages**: `7`
- **Clients**: `20`
- **Participation**: `0.25`
- **Local epochs**: `5`
- **Rounds per stage**: `40`
- **Backbone**: `2-layer GCN`
- **alpha**: `<0.10 / 10.00 / 100.00>`
- **psi**: `<0.00 / 0.50 / 1.00>`
- **Seed**: `<1 / 2>`
- **Outputs to save**:
  - AFA
  - AF
  - B10
  - realized heterogeneity audit
