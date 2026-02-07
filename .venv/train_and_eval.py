"""
Enhanced Evaluation Module with Comprehensive Metrics
for LDS, Noise Robustness, and Convergence Analysis
"""

import torch
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, data_loader, device, minority_idx=None):
    """
    Comprehensive model evaluation with LDS and noise robustness metrics.

    Returns metrics for:
    1. Medical diagnostics (Sensitivity, Specificity)
    2. LDS analysis (Minority class performance, class imbalance)
    3. Noise robustness (AUC, loss, confidence)
    4. Fairness (Macro-averaged metrics, performance gap)
    """
    model.eval()
    criterion = torch.nn.BCELoss()

    running_loss = 0.0
    total = 0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            all_probs.extend(outputs.detach().cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    # Convert to numpy
    all_targets = np.array(all_targets, dtype=int)
    all_probs = np.array(all_probs, dtype=float)
    all_preds = (all_probs >= 0.5).astype(int)

    # -------------------------------------------------------------------------
    # CLASS DISTRIBUTION ANALYSIS (Critical for LDS)
    # -------------------------------------------------------------------------
    counts = np.bincount(all_targets, minlength=2)

    # Auto-detect minority class
    if minority_idx is None:
        minority_idx = np.argmin(counts) if counts[0] != counts[1] else 1

    majority_idx = 1 - minority_idx

    # Quantify imbalance
    if counts[minority_idx] > 0:
        imbalance_ratio = counts[majority_idx] / counts[minority_idx]
    else:
        imbalance_ratio = float('inf')

    # -------------------------------------------------------------------------
    # PER-CLASS METRICS
    # -------------------------------------------------------------------------
    prec, rec, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1], average=None, zero_division=0
    )

    # Macro-averaged metrics (fairness indicators)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )

    # -------------------------------------------------------------------------
    # AUC METRICS
    # -------------------------------------------------------------------------
    if len(np.unique(all_targets)) > 1:
        roc_auc = roc_auc_score(all_targets, all_probs)
        pr_auc_c1 = average_precision_score(all_targets, all_probs)
    else:
        roc_auc = 0.5
        pr_auc_c1 = 0.0

    # -------------------------------------------------------------------------
    # FAIRNESS & BIAS METRICS (New!)
    # -------------------------------------------------------------------------
    # Performance gap between classes (smaller = more fair)
    performance_gap = abs(f1[0] - f1[1])

    # Balanced accuracy (geometric mean of per-class recalls)
    balanced_acc = balanced_accuracy_score(all_targets, all_preds)

    # Confusion matrix for detailed analysis
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        # =====================================================================
        # BASIC METRICS
        # =====================================================================
        'test_loss': running_loss / total,
        'test_accuracy': accuracy_score(all_targets, all_preds),
        'balanced_accuracy': balanced_acc,  # NEW: Better for imbalanced data

        # =====================================================================
        # MEDICAL METRICS (Domain-specific)
        # =====================================================================
        'sensitivity': rec[1],      # True Positive Rate (Pneumonia detection)
        'specificity': rec[0],      # True Negative Rate (Normal detection)
        'precision_c1': prec[1],    # Positive Predictive Value
        'f1_score_c1': f1[1],       # Harmonic mean for class 1

        # =====================================================================
        # LDS ANALYSIS METRICS ★ (Core contribution)
        # =====================================================================
        'f1_minority': f1[minority_idx],
        'recall_minority': rec[minority_idx],
        'precision_minority': prec[minority_idx],

        'f1_majority': f1[majority_idx],  # NEW: For comparison
        'recall_majority': rec[majority_idx],

        # Bias quantification
        'performance_gap': performance_gap,  # NEW: |F1_c0 - F1_c1|
        'imbalance_ratio': imbalance_ratio,  # NEW: majority/minority count
        'minority_support': int(counts[minority_idx]),  # NEW: Sample count
        'majority_support': int(counts[majority_idx]),  # NEW: Sample count

        # =====================================================================
        # GLOBAL FAIRNESS METRICS
        # =====================================================================
        'f1_macro': f1_macro,
        'precision_macro': prec_macro,  # NEW: Added for completeness
        'recall_macro': rec_macro,      # NEW: Added for completeness

        # =====================================================================
        # NOISE ROBUSTNESS METRICS
        # =====================================================================
        'roc_auc_global': roc_auc,
        'pr_auc_c1': pr_auc_c1,

        # Calibration (how confident is the model?)
        'avg_confidence': float(np.mean(np.maximum(all_probs, 1 - all_probs))),  # NEW

        # =====================================================================
        # DETAILED ANALYSIS
        # =====================================================================
        'confusion_matrix': cm.tolist(),
        'true_positives': int(tp),   # NEW: Explicit CM values
        'true_negatives': int(tn),   # NEW
        'false_positives': int(fp),  # NEW
        'false_negatives': int(fn),  # NEW

        'minority_idx': int(minority_idx)
    }


def train_local_model(model, train_loader, val_loader, optimizer, epochs, device, client_id=""):
    """
    Enhanced training with convergence and stability metrics.

    Returns:
        weights: Model state dict
        history: Extended history with convergence metrics
    """
    model.train()
    criterion = torch.nn.BCELoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_times': []  # NEW: Track time per epoch
    }

    for epoch in range(epochs):
        epoch_start = datetime.now()

        # ---------------------------------------------------------------------
        # Training Phase
        # ---------------------------------------------------------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = (outputs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        history['train_loss'].append(running_loss / total)
        history['train_acc'].append(correct / total)

        # ---------------------------------------------------------------------
        # Validation Phase
        # ---------------------------------------------------------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device).float().view(-1, 1)

                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item() * val_inputs.size(0)
                val_correct += ((val_outputs >= 0.5).float() == val_targets).sum().item()
                val_total += val_targets.size(0)

        history['val_loss'].append(val_loss / val_total)
        history['val_acc'].append(val_correct / val_total)

        # Track epoch time
        epoch_time = (datetime.now() - epoch_start).total_seconds()
        history['epoch_times'].append(epoch_time)

        # ---------------------------------------------------------------------
        # Logging (only final epoch for AWS parallelization)
        # ---------------------------------------------------------------------
        if (epoch + 1) == epochs:
            now = datetime.now().strftime("%H:%M:%S")
            pid = os.getpid()
            log_prefix = f"[{now} | PID: {pid}]"
            if client_id:
                log_prefix += f" {client_id}"

            print(f"{log_prefix} -> Ep {epoch+1}/{epochs} | "
                  f"Val Acc: {history['val_acc'][-1]:.3f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f} | "
                  f"Time: {epoch_time:.1f}s")

    # -------------------------------------------------------------------------
    # POST-TRAINING ANALYSIS (Convergence & Stability)
    # -------------------------------------------------------------------------
    # WICHTIG: Nutzt jetzt die unified Funktion mit dem Modus "local"
    history['convergence_metrics'] = compute_unified_convergence_metrics(
        history,
        mode="local"
    )

    return model.state_dict(), history


def compute_unified_convergence_metrics(
    history,
    mode: str,
    local_epochs: int = None,
    stability_window: int = 5,
    plateau_tol: float = 0.01
):
    """
    Unified convergence metrics for:
    - centralized
    - local
    - federated (round-aware)

    mode ∈ {"central", "local", "federated"}
    """

    val_losses = np.array(history['val_loss'])

    # --- 1. Effektive Vergleichsachse definieren ---
    if mode == "federated":
        if local_epochs is None:
            raise ValueError("local_epochs required for federated mode")
        idx = np.arange(local_epochs - 1, len(val_losses), local_epochs)
        eff_losses = val_losses[idx]
    else:
        eff_losses = val_losses

    n = len(eff_losses)
    window = min(stability_window, n)

    # --- 2. Grundmetriken (für alle gleich interpretiert) ---
    final_loss = float(eff_losses[-1])
    total_improvement = float(eff_losses[0] - eff_losses[-1])
    improvement_per_step = float(total_improvement / max(1, n - 1))
    stability_std = float(np.std(eff_losses[-window:]))

    # --- 3. Konvergenzpunkt (Plateau-Detection) ---
    plateau_step = None
    for i in range(window, n):
        if np.std(eff_losses[i - window:i]) < plateau_tol:
            plateau_step = i
            break

    metrics = {
        "final_loss": final_loss,
        "total_improvement": total_improvement,
        "improvement_per_step": improvement_per_step,
        "stability_std": stability_std,
        "convergence_step": plateau_step,
        "effective_steps": n,
        "converged": plateau_step is not None
    }

    return metrics