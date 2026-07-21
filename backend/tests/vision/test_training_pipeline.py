import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.scripts.train_vision_classifier import (
    FocalLoss,
    calibrate_thresholds,
    create_parser,
    fit_temperature,
    should_optimizer_step,
)


def test_cli_exposes_implemented_lifecycle_modes():
    parser = create_parser()
    for mode in ("cross_validate", "train_final", "evaluate", "promote"):
        args = parser.parse_args(["--mode", mode])
        assert args.mode == mode


def test_focal_loss_is_finite_and_backpropagates():
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
    targets = torch.tensor([0, 1])
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_threshold_selection_minimizes_far_under_frr_constraint():
    probabilities = [0.05, 0.10, 0.20, 0.55, 0.65, 0.80, 0.90]
    labels = [0, 0, 0, 1, 1, 1, 1]
    result = calibrate_thresholds(probabilities, labels, max_frr=0.0)
    assert result["policy_satisfied"] is True
    assert result["frr"] == 0.0
    assert result["far"] == 0.0
    assert 0.20 < result["counterfeit_threshold"] <= 0.55


def test_temperature_scaling_returns_positive_value():
    temperature = fit_temperature(
        [[4.0, -1.0], [-1.0, 4.0], [3.0, -0.5], [-0.5, 3.0]],
        [0, 1, 0, 1],
    )
    assert temperature > 0.0


def test_gradient_accumulation_steps_final_remainder():
    assert should_optimizer_step(1, loader_length=5, accumulation=2)
    assert should_optimizer_step(3, loader_length=5, accumulation=2)
    assert should_optimizer_step(4, loader_length=5, accumulation=2)
    assert not should_optimizer_step(0, loader_length=5, accumulation=2)
    with pytest.raises(ValueError):
        should_optimizer_step(0, loader_length=1, accumulation=0)