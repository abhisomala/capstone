"""Smoke test for the classifier: correct output shape and finite gradients."""

import torch

from src.models.model import BloomMLP


def test_forward_shape_and_backward():
    model = BloomMLP(in_features=12, hidden=(32, 16), dropout=0.3)
    x = torch.randn(8, 12)
    out = model(x)
    assert out.shape == (8, 1)   # one logit per sample

    loss = torch.nn.BCEWithLogitsLoss()(out, torch.zeros(8, 1))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
