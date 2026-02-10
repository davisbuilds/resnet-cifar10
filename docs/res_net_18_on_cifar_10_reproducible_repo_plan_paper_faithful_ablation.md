# ResNet-18 on CIFAR-10 — Reproducible Repo Plan

**Mode**: Paper-faithful (He et al., 2015) with one baked-in extension (residual ablation)  
**Framework**: PyTorch (MPS)  
**Package manager**: uv

---

## 0. Design Principles
- Stay as close as possible to the original ResNet CIFAR setup
- Minimize “modern conveniences” that change training dynamics
- Make every deviation from the paper explicit and justified
- Enable a controlled residual-vs-non-residual comparison

---

## 1. Repository Layout

```
resnet-cifar10/
├── pyproject.toml
├── uv.lock
├── README.md
├── experiments/
│   ├── baseline.yaml
│   └── no_residual.yaml
├── data/
│   └── cifar10.py
├── models/
│   ├── resnet.py
│   └── blocks.py
├── train.py
├── eval.py
├── utils/
│   ├── seed.py
│   ├── metrics.py
│   └── logging.py
└── results/
```

---

## 2. Environment & Package Management (uv)

### Python
- Python >= 3.10

### Core Dependencies
- torch
- torchvision
- numpy
- pyyaml
- matplotlib

### uv Workflow
- `uv init`
- Declare dependencies in `pyproject.toml`
- Lock with `uv lock`
- Run via `uv run python train.py`

Pin torch/torchvision versions explicitly to avoid backend drift.

---

## 3. Dataset Module (`data/cifar10.py`)

### Dataset
- CIFAR-10 via torchvision

### Normalization (Paper-Standard)
- Mean: (0.4914, 0.4822, 0.4465)
- Std:  (0.2023, 0.1994, 0.2010)

### Augmentation
**Training**:
- RandomCrop(32, padding=4)
- RandomHorizontalFlip

**Test**:
- No augmentation

Augmentations are applied exactly once per sample per epoch.

---

## 4. Model Definition

### 4.1 CIFAR-Style ResNet-18

Paper-faithful modifications relative to ImageNet ResNet:
- Initial conv: 3×3, stride 1, padding 1
- No max-pooling layer
- Channel stages: [64, 128, 256, 512]
- BasicBlock only

### 4.2 Residual Toggle (Key Extension)

All residual behavior is controlled by a single flag:

```
use_residual: true | false
```

Implementation detail:
- Skip connection becomes identity (true) or zero (false)
- Block depth, parameter count, and initialization remain unchanged

This ensures the ablation isolates the effect of residual connections only.

---

## 5. Configuration System (`experiments/*.yaml`)

### baseline.yaml
- use_residual: true
- lr: 0.1
- batch_size: 128
- epochs: 150
- optimizer: sgd
- momentum: 0.9
- weight_decay: 5e-4
- lr_schedule:
  - milestones: [75, 112]
  - gamma: 0.1

### no_residual.yaml
Identical to baseline except:
- use_residual: false

---

## 6. Training Script (`train.py`)

### Responsibilities
- Parse experiment config
- Set seeds (Python / NumPy / Torch)
- Select device (MPS → CPU fallback)
- Instantiate model, optimizer, scheduler
- Run training loop
- Save checkpoints

### Checkpointing
- Save:
  - `best.pt` (by validation accuracy)
  - `final.pt`

### Logging
- Epoch-level metrics only (paper-faithful)
- Write JSONL or CSV logs to `results/`

---

## 7. Evaluation Script (`eval.py`)

- Load trained checkpoint
- Evaluate on CIFAR-10 test set
- Report:
  - Top-1 accuracy
  - Per-class accuracy (optional)

---

## 8. Reproducibility Controls

- Fixed random seeds
- Deterministic CuDNN disabled (not applicable on MPS)
- Log:
  - torch version
  - torchvision version
  - device backend
  - git commit hash

Note expected ±0.2–0.5% variance.

---

## 9. Expected Results

| Experiment        | Expected Accuracy |
|------------------|-------------------|
| ResNet-18        | 93–95%            |
| No-Residual CNN  | 85–88%            |

Residual ablation should show:
- Slower convergence
- Higher training loss
- Lower final generalization

---

## 10. README Structure

- Motivation
- Paper reference
- Environment setup (uv)
- How to run experiments
- Results table
- Discussion: Why residuals matter

---

## 11. Exit Criteria

- Both experiments run end-to-end
- Results fall within expected ranges
- Differences attributable solely to residual connections
- Repo is clean, locked, and reproducible
