# ResNet-18 on CIFAR-10 (Paper-Faithful, Residual Ablation)

This repository replicates a canonical computer vision experiment from **He et al. (2015)** by training a **CIFAR-style ResNet-18** on the **CIFAR-10** dataset using **PyTorch** on Apple Silicon. In addition to the baseline, it includes a **controlled residual ablation** that isolates the effect of skip connections while keeping model depth and parameter count constant.

The goal is not leaderboard performance, but **reproducibility, interpretability, and faithfulness to the original experiment**.

---

## 1. Reference

> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.  
> *Deep Residual Learning for Image Recognition*.  
> CVPR 2016.  
> https://arxiv.org/abs/1512.03385

---

## 2. What This Repository Demonstrates

- A clean, paper-faithful implementation of **ResNet-18 for CIFAR-10**
- Why **residual connections matter** for optimization and generalization
- How to structure a small but serious ML experiment for local reproducibility

Two experiments are provided:

| Experiment | Description |
|----------|-------------|
| `baseline.yaml` | Standard ResNet-18 with identity skip connections |
| `no_residual.yaml` | Identical network with skip connections removed |

The only difference between these runs is whether the residual path is active.

---

## 3. Design Philosophy

- **Paper-faithful by default**
  - CIFAR-specific ResNet (3×3 conv, no max-pool)
  - SGD + momentum
  - Multi-step learning rate decay
- **Minimal abstraction**
  - Explicit training loops
  - No training frameworks or hidden magic
- **Controlled ablation**
  - Residual connections toggled via a single flag
  - No architectural confounders

Every intentional deviation from the paper is documented.

---

## 4. Repository Structure

```
resnet-cifar10/
├── pyproject.toml        # Dependency specification (uv)
├── uv.lock               # Fully pinned environment
├── .gitignore
├── README.md
├── experiments/
│   ├── baseline.yaml
│   └── no_residual.yaml
├── data/
│   └── cifar10.py        # Dataset + transforms
├── models/
│   ├── blocks.py         # Residual and non-residual blocks
│   └── resnet.py         # CIFAR-style ResNet-18
├── train.py              # Training entrypoint
├── eval.py               # Evaluation entrypoint
├── plot_metrics.py       # Seaborn plots from metrics.csv
├── utils/
│   ├── seed.py
│   ├── metrics.py
│   ├── logging.py
│   └── config.py
├── tests/
│   ├── test_config.py
│   ├── test_data_split.py
│   ├── test_logging.py
│   ├── test_model.py
│   └── test_train_epoch.py
└── results/              # Logs, checkpoints, plots
```

---

## 5. Environment Setup (uv)

### Requirements
- macOS (Apple Silicon recommended)
- Python ≥ 3.10
- `uv` installed

### Setup

```bash
uv sync --dev
```

To run any command:

```bash
uv run python train.py --config experiments/baseline.yaml
```

Torch automatically selects **MPS** when available, with CPU fallback.

### Run Tests

```bash
uv run pytest -q
```

---

## 6. Running the Experiments

### Train Baseline ResNet-18

```bash
uv run python train.py --config experiments/baseline.yaml
```

Recommended for long runs (tmux):

```bash
tmux new -s resnet-baseline
uv run python train.py --config experiments/baseline.yaml
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t resnet-baseline
```

Resume from a checkpoint after interruption:

```bash
uv run python train.py --config experiments/baseline.yaml --resume-from results/<run_dir>/best.pt
```

Quick real-data smoke test:

```bash
uv run python train.py --config experiments/baseline.yaml --max-epochs 1 --limit-train-batches 2 --limit-val-batches 2
```

Training uses a deterministic split of CIFAR-10 train into `45,000 train / 5,000 val`.

### Train No-Residual Ablation

```bash
uv run python train.py --config experiments/no_residual.yaml
```

### Evaluate a Trained Model

```bash
uv run python eval.py --checkpoint results/<run_dir>/best.pt --num-workers 0
```

`eval.py` evaluates only on the CIFAR-10 test split.

### Plot Training Curves (Seaborn)

Single run:

```bash
uv run python plot_metrics.py results/<run_dir>
```

Compare multiple runs:

```bash
uv run python plot_metrics.py results/<run_dir_1> results/<run_dir_2>
```

---

## 7. Training Configuration (Baseline)

- Optimizer: SGD
- Momentum: 0.9
- Weight decay: 5e-4
- Batch size: 128
- Epochs: 150
- Validation split: 5,000 images from CIFAR-10 train (45k/5k split)
- Learning rate: 0.1
- LR schedule: MultiStepLR
  - Milestones: [75, 112]
  - Gamma: 0.1

These values match common CIFAR-10 ResNet setups derived from the original paper.

---

## 8. Expected Results

| Model | Test Accuracy (Top-1) |
|------|----------------------|
| ResNet-18 | 93–95% |
| No-Residual CNN | 85–88% |

The no-residual model should:
- Converge more slowly
- Reach higher training loss
- Generalize substantially worse

This gap is the core empirical result of the repository.

---

## 9. Reproducibility Notes

- All random seeds are fixed (Python, NumPy, PyTorch)
- Dependency versions are fully pinned via `uv.lock`
- Default configs use `num_workers: 0` for broad portability (including restricted environments)
- Remaining variance (~±0.3%) is expected due to nondeterminism in MPS kernels

Log files include:
- Hyperparameters
- Device backend
- Torch / torchvision versions
- Git commit hash

---

## 10. Extensions (Out of Scope)

Potential follow-ons, intentionally excluded from the baseline:
- Adam / AdamW comparison
- CutMix or MixUp augmentation
- Wider or deeper ResNets
- ImageNet-scale training

The emphasis here is **clarity over novelty**.

---

## 11. License

MIT
