import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.scripts.train_vision_classifier import NTXentLoss, FocalLoss, calibrate_thresholds

def test_focal_loss():
    loss_fn = FocalLoss(gamma=2.0)
    
    # Batch size 2, Classes 2
    logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    targets = torch.tensor([0, 1])
    
    loss = loss_fn(logits, targets)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)
    
    # Test behavior on hard vs easy examples
    easy_logits = torch.tensor([[10.0, -10.0]])
    easy_loss = loss_fn(easy_logits, torch.tensor([0]))
    
    hard_logits = torch.tensor([[-10.0, 10.0]])
    hard_loss = loss_fn(hard_logits, torch.tensor([0]))
    
    assert hard_loss.item() > easy_loss.item()
    
def test_calibration_threshold():
    val_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # At t=0.5:
    # 0.1(0) 0.2(0) 0.3(0) 0.4(0) 0.5(1) 0.6(1) 0.7(1) 0.8(1) 0.9(1)
    val_true =  [0,   0,   0,   0,   1,   1,   1,   1,   1]
    
    # Max FRR 0.05
    threshold = calibrate_thresholds(val_probs, val_true, max_frr=0.05)
    
    # Since any threshold <= 0.5 results in 0 False Rejects (genuine predicted counterfeit),
    # and minimizing FAR means picking the highest threshold possible that satisfies FRR,
    # it should pick a threshold near 0.5.
    assert 0.4 <= threshold <= 0.6
