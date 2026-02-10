from __future__ import annotations

from data.cifar10 import train_val_split_indices


def test_train_val_split_indices_sizes_and_disjointness() -> None:
    train_indices, val_indices = train_val_split_indices(train_size=50000, val_size=5000, split_seed=42)
    assert len(train_indices) == 45000
    assert len(val_indices) == 5000
    assert not set(train_indices).intersection(val_indices)


def test_train_val_split_indices_is_deterministic() -> None:
    train_a, val_a = train_val_split_indices(train_size=100, val_size=10, split_seed=7)
    train_b, val_b = train_val_split_indices(train_size=100, val_size=10, split_seed=7)
    assert train_a == train_b
    assert val_a == val_b

