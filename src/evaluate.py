"""Evaluate the trained bloom classifier on the held-out test set — touched once.

Reports ROC-AUC, average precision (PR-AUC), precision, recall, F1, and the confusion
matrix, using the decision threshold tuned on validation during training. Accuracy is
reported only alongside the others, never as the headline, because the classes are
imbalanced. Results are written to a file under logs/.

Run: python -m src.evaluate
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, accuracy_score,
)

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.feature_splits import load_feature_splits, feature_columns
from src.models.model import BloomMLP


def evaluate(config: dict, logger):
    ckpt_path = resolve_path(config["paths"]["models_dir"]) / config["training"]["checkpoint_name"]
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found — run `python -m src.train` first.")
    ckpt = torch.load(ckpt_path, weights_only=False)

    _, _, test_df = load_feature_splits(config)
    feat_cols = ckpt["feature_cols"]
    # Guard against feature drift between training and evaluation.
    assert feat_cols == feature_columns(test_df), "Feature columns changed since training."

    X = test_df[feat_cols].to_numpy(dtype=np.float32)
    X = (X - ckpt["scaler_mean"]) / ckpt["scaler_std"]
    y = test_df["label"].to_numpy(dtype=int)

    model = BloomMLP(len(feat_cols), hidden=tuple(ckpt["hidden"]), dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.tensor(X))).squeeze(1).numpy()

    threshold = ckpt["threshold"]
    preds = (probs >= threshold).astype(int)

    logger.info("=== Held-out TEST evaluation (touched once) ===")
    logger.info("Test samples=%d | bloom=%d (%.1f%%) | stations=%d | threshold=%.2f",
                len(y), int(y.sum()), 100.0 * y.mean() if len(y) else 0.0,
                test_df["station_id"].nunique(), threshold)

    if len(np.unique(y)) < 2:
        logger.warning("Test set has a single class; ROC-AUC / PR-AUC undefined.")
        roc = ap = float("nan")
    else:
        roc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)

    metrics = {
        "ROC_AUC": roc,
        "PR_AUC_avg_precision": ap,
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "F1": f1_score(y, preds, zero_division=0),
        "accuracy": accuracy_score(y, preds),
    }
    for k, v in metrics.items():
        logger.info("  %-22s %.4f", k, v)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    logger.info("Confusion matrix [rows=true 0/1, cols=pred 0/1]:")
    logger.info("  TN=%d FP=%d", cm[0, 0], cm[0, 1])
    logger.info("  FN=%d TP=%d", cm[1, 0], cm[1, 1])

    # Stratify by CyAN resolvability: does the model work on large (CyAN-resolvable)
    # lakes but fail on the small ones? This is the differentiation test — the small
    # lakes are the product, and MODIS 500 m may be too coarse for them (see
    # pathtoselleable.md). This informs the sensor decision, not the headline number.
    if "cyan_class" in test_df.columns:
        logger.info("Test ROC-AUC by CyAN-resolvability class:")
        for cls in ("resolvable", "marginal", "unresolvable"):
            m = (test_df["cyan_class"] == cls).to_numpy()
            yc, pc = y[m], probs[m]
            if m.sum() and len(np.unique(yc)) == 2:
                logger.info("  %-12s n=%d bloom=%d ROC-AUC=%.4f",
                            cls, int(m.sum()), int(yc.sum()), roc_auc_score(yc, pc))
            else:
                logger.info("  %-12s n=%d bloom=%d ROC-AUC=n/a (single class)",
                            cls, int(m.sum()), int(yc.sum()))
    return metrics, cm


def main():
    config = load_config()
    logger = get_run_logger("evaluate", resolve_path(config["paths"]["logs_dir"]))
    evaluate(config, logger)


if __name__ == "__main__":
    main()
