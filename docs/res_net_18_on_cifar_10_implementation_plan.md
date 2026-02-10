# ResNet-18 on CIFAR-10 (PyTorch, Apple Silicon)

## Objective
Replicate a canonical computer vision experiment by training **ResNet-18** on **CIFAR-10** using PyTorch on a Mac Mini (Apple Silicon). Target **93–95% test accuracy** with a clean, reproducible setup suitable for a weekend end-to-end run.

---

## Scope and Constraints
- **Hardware**: Mac Mini (Apple Silicon, MPS backend)
- **Framework**: PyTorch + torchvision
- **Dataset**: CIFAR-10 (32×32 RGB)
- **Time budget**: ~1 weekend (8–12 hours total)
- **Non-goals**:
  - No distributed training
  - No large-scale hyperparameter sweeps
  - No ImageNet-scale models

---

## Reference
- He et al., *Deep Residual Learning for Image Recognition* (2015)
- CIFAR-10 benchmark conventions (as used in torchvision and common repos)

---

## High-Level Plan
1. Environment setup and validation
2. Dataset preparation + augmentation
3. Model definition (CIFAR-adapted ResNet-18)
4. Training loop + logging
5. Evaluation and sanity checks
6. Baseline result reproduction
7. Optional extension experiment

---

## 1. Environment Setup

### Python & Tooling
- Python >= 3.10
- Virtual environment (venv or conda)
- Core dependencies:
  - torch
  - torchvision
  - numpy
  - matplotlib

### Device Handling
- Prefer MPS if available
- Fallback to CPU automatically

Validation step:
- Confirm `torch.backends.mps.is_available()`
- Run a single forward/backward pass on MPS

---

## 2. Dataset Preparation

### Dataset
- CIFAR-10 train (50,000) / test (10,000)
- Loaded via `torchvision.datasets.CIFAR10`

### Normalization
Use standard CIFAR-10 statistics:
- Mean: (0.4914, 0.4822, 0.4465)
- Std:  (0.2023, 0.1994, 0.2010)

### Training Augmentations
- RandomCrop(32, padding=4)
- RandomHorizontalFlip
- ToTensor
- Normalize

### Test Transform
- ToTensor
- Normalize

---

## 3. Model Definition

### Architecture Choice
- ResNet-18
- CIFAR variant:
  - Replace initial 7×7 conv with 3×3 conv
  - Remove initial max-pool layer

### Implementation Strategy
- Option A: Modify torchvision’s ResNet implementation
- Option B: Write a minimal ResNet-18 from scratch (preferred for learning)

### Output Layer
- Fully connected layer with 10 outputs

---

## 4. Training Configuration

### Hyperparameters (Baseline)
- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 5e-4
- Batch size: 128
- Epochs: 150
- Loss: CrossEntropyLoss

### Learning Rate Schedule
- Initial LR: 0.1
- Scheduler: MultiStepLR
  - Milestones: [75, 112]
  - Gamma: 0.1

---

## 5. Training Loop

Responsibilities:
- Move data to device
- Forward pass
- Compute loss
- Backward pass
- Optimizer step
- Scheduler step (per epoch)

Metrics tracked per epoch:
- Training loss
- Training accuracy
- Validation accuracy

Checkpoints:
- Save best model by validation accuracy
- Save final model

---

## 6. Evaluation

### Metrics
- Final test accuracy
- Per-class accuracy (optional)

### Sanity Checks
- Training accuracy > validation accuracy
- Loss decreases smoothly
- No divergence or NaNs

### Visualization
- Loss vs. epoch
- Accuracy vs. epoch

---

## 7. Reproducibility

- Set random seeds:
  - Python
  - NumPy
  - PyTorch
- Log:
  - Git commit hash
  - Hyperparameters
  - Device type

Document expected variance across runs.

---

## 8. Optional Extension (Choose One)

### A. Residual Ablation
- Train identical network with skip connections removed
- Compare convergence speed and final accuracy

### B. Data Augmentation Ablation
- Remove RandomCrop / Flip
- Quantify generalization drop

### C. Optimizer Comparison
- SGD vs AdamW (same schedule)

---

## 9. Repository Structure

```
resnet-cifar10/
├── data/
├── models/
│   └── resnet.py
├── train.py
├── eval.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## Expected Outcome
- Test accuracy: 93–95%
- Training time: 1–3 hours on MPS
- Clean, reproducible baseline suitable for further experiments

---

## Exit Criteria
- Model trains end-to-end without manual intervention
- Results fall within expected accuracy range
- One extension experiment completed and documented
