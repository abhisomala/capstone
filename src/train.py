"""Train the bloom classifier with real early stopping and best-checkpoint selection.

  * Features are standardised using TRAIN statistics only (no leakage into val/test).
  * Class imbalance is handled with a positive class weight in the loss.
  * The best checkpoint is the epoch with the highest validation ROC-AUC — not the last
    epoch. Training stops early once validation stops improving.
  * A decision threshold is tuned on the validation set (max F1) and stored in the
    checkpoint, so evaluate.py never needs to touch the test set to pick a threshold.
  * Every epoch is logged to a file under logs/.

The test split is deliberately NOT read here.

Run: python -m src.train
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score

from src.utils.config import load_config, resolve_path
from src.utils.logging_setup import get_run_logger
from src.data.feature_splits import load_feature_splits, feature_columns
from src.models.model import BloomMLP


def _standardize(train_X, *others):
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    std[std == 0] = 1.0
    out = [(train_X - mean) / std] + [(o - mean) / std for o in others]
    return mean, std, out


def _best_f1_threshold(y_true, probs):
    """Threshold in (0,1) maximising F1 on the given (validation) set."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def train(config: dict, logger):
    tcfg = config["training"]
    torch.manual_seed(tcfg["random_seed"])
    np.random.seed(tcfg["random_seed"])

    train_df, val_df, test_df = load_feature_splits(config)
    feat_cols = feature_columns(train_df)
    logger.info("Features (%d): %s", len(feat_cols), feat_cols)
    logger.info("Split sizes — train=%d (bloom=%d) val=%d (bloom=%d) test=%d (held out)",
                len(train_df), int(train_df.label.sum()),
                len(val_df), int(val_df.label.sum()), len(test_df))

    Xtr = train_df[feat_cols].to_numpy(dtype=np.float32)
    Xva = val_df[feat_cols].to_numpy(dtype=np.float32)
    ytr = train_df["label"].to_numpy(dtype=np.float32)
    yva = val_df["label"].to_numpy(dtype=np.float32)

    mean, std, (Xtr, Xva) = _standardize(Xtr, Xva)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr).unsqueeze(1)
    Xva_t = torch.tensor(Xva)

    n_pos = float(ytr.sum()); n_neg = float(len(ytr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
    logger.info("Class weighting: pos_weight=%.2f (train pos=%d neg=%d)",
                pos_weight.item(), int(n_pos), int(n_neg))

    model = BloomMLP(len(feat_cols), hidden=tuple(tcfg["hidden"]), dropout=tcfg["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["learning_rate"],
                                 weight_decay=tcfg["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_has_both = len(np.unique(yva)) == 2
    if not val_has_both:
        logger.warning("Validation set has a single class — selecting on val loss instead of AUC.")

    n = len(Xtr_t); bs = tcfg["batch_size"]
    best_metric, best_state, best_epoch = -np.inf, None, -1
    patience = 0

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            optimizer.zero_grad()
            loss = criterion(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_logits = model(Xva_t)
            val_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(
                val_logits, torch.tensor(yva).unsqueeze(1)).item()
            val_probs = torch.sigmoid(val_logits).squeeze(1).numpy()
        val_auc = roc_auc_score(yva, val_probs) if val_has_both else float("nan")
        selection_metric = val_auc if val_has_both else -val_loss

        if selection_metric > best_metric:
            best_metric, best_epoch, patience = selection_metric, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            logger.info("epoch %3d | train_loss=%.4f val_loss=%.4f val_auc=%s | best_epoch=%d",
                        epoch, epoch_loss, val_loss,
                        f"{val_auc:.4f}" if val_has_both else "n/a", best_epoch)
        if patience >= tcfg["early_stopping_patience"]:
            logger.info("Early stopping at epoch %d (no val improvement for %d epochs).",
                        epoch, patience)
            break

    # Restore the best checkpoint and tune the decision threshold on validation.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(Xva_t)).squeeze(1).numpy()
    threshold, val_f1 = _best_f1_threshold(yva, val_probs) if val_has_both else (0.5, float("nan"))
    logger.info("Best epoch=%d val_selection=%.4f | tuned threshold=%.2f (val F1=%s)",
                best_epoch, best_metric, threshold,
                f"{val_f1:.4f}" if val_has_both else "n/a")

    # Guard against silently shipping a collapsed model. On this dataset the MLP is
    # seed-sensitive and can converge to predicting a single class; when that happens the
    # single-split number is meaningless. The trustworthy metric is `src.cross_validate`.
    frac_pos = float((val_probs >= threshold).mean())
    if frac_pos > 0.98 or frac_pos < 0.02:
        logger.warning("MLP appears DEGENERATE (predicts one class for %.0f%% of val). "
                       "Do not report this single-split model — use `python -m src.cross_validate`.",
                       100 * frac_pos)

    models_dir = resolve_path(config["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = models_dir / tcfg["checkpoint_name"]
    torch.save({
        "state_dict": best_state,
        "feature_cols": feat_cols,
        "scaler_mean": mean, "scaler_std": std,
        "hidden": tcfg["hidden"], "dropout": tcfg["dropout"],
        "threshold": threshold,
        "best_epoch": best_epoch,
    }, ckpt_path)
    logger.info("Saved best checkpoint to %s", ckpt_path)


def main():
    config = load_config()
    logger = get_run_logger("train", resolve_path(config["paths"]["logs_dir"]))
    train(config, logger)


if __name__ == "__main__":
    main()
