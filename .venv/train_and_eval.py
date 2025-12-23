import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


def train_client(model, train_loader, optimizer, client_id, num_epochs, val_loader=None, device="cpu",
                 mode_label="Baseline"):

    criterion = nn.BCEWithLogitsLoss()

    history = {
        "train": {
            "loss": [],
            "accuracy": [],
            "f1_normal": [],
            "f1_krank": []
        }
    }
    if val_loader is not None:
        history["val"] = {
            "loss": [],
            "accuracy": [],
            "f1_normal": [],
            "f1_krank": []
        }

    n_samples = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images = images.to(device)
            # shape entspricht (Batch, 1)
            labels = labels.float().view(-1, 1).to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

            # Prediction: Logits >= 0.0 -> Klasse 1
            preds = (logits >= 0.0).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if epoch == 0:
                n_samples += labels.size(0)

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()

        train_loss = np.mean(epoch_losses)
        train_acc = accuracy_score(all_labels, all_preds) * 100.0
        # F1-Scores pro Klasse [Normal, Krank]
        train_f1s = f1_score(all_labels, all_preds, labels=[0, 1], average=None, zero_division=0)

        history["train"]["loss"].append(train_loss)
        history["train"]["accuracy"].append(train_acc)
        history["train"]["f1_normal"].append(train_f1s[0])
        history["train"]["f1_krank"].append(train_f1s[1])

        val_str = ""
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, device)
            history["val"]["loss"].append(val_metrics["loss"])
            history["val"]["accuracy"].append(val_metrics["accuracy"])
            history["val"]["f1_normal"].append(val_metrics["f1_normal"])
            history["val"]["f1_krank"].append(val_metrics["f1_krank"])

            val_str = (f" | Val Loss: {val_metrics['loss']:.4f} "
                       f"| Val Acc: {val_metrics['accuracy']:.2f}% "
                       f"| Val F1-N: {val_metrics['f1_normal']:.3f}")

        print(
            f"[{client_id} | {mode_label}] "
            f"Ep {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}%"
            f"{val_str}"
        )

    # Gewichte für Aggregation sicher kopieren
    safe_weights = {
        k: v.cpu().detach().clone() for k, v in model.state_dict().items()
    }

    return {
        "weights": safe_weights,
        "n_samples": n_samples,
        "history": history
    }


def evaluate_model(model, loader, device="cpu"):

    # Berechnet Metriken zur Analyse von Label Skew (F1 pro Klasse, Precision, Recall, AUC)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    all_logits, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().view(-1, 1).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    # Zusammenführen der Batches
    all_logits = torch.cat(all_logits).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()

    # Wahrscheinlichkeiten (Sigmoid) und binäre prediction
    probs = 1 / (1 + np.exp(-all_logits))
    preds = (all_logits >= 0.0).astype(float)

    avg_loss = total_loss / len(all_labels)

    # F1 pro Klasse
    f1_per_class = f1_score(all_labels, preds, labels=[0, 1], average=None, zero_division=0)

    # Prüfung auf mehrere Klassen für AUC
    unique_labels = len(np.unique(all_labels))

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, preds) * 100.0,

        "f1_macro": f1_score(all_labels, preds, average="macro", zero_division=0),
        "f1_normal": float(f1_per_class[0]),
        "f1_krank": float(f1_per_class[1]),

        "precision_normal": precision_score(all_labels, preds, pos_label=0, zero_division=0),
        "precision_krank": precision_score(all_labels, preds, pos_label=1, zero_division=0),
        "recall_normal": recall_score(all_labels, preds, pos_label=0, zero_division=0),
        "recall_krank": recall_score(all_labels, preds, pos_label=1, zero_division=0),

        # AUC-Metriken (Sensibel für Konfidenz)
        "auc_roc": roc_auc_score(all_labels, probs) if unique_labels > 1 else np.nan,
        "auc_pr": average_precision_score(all_labels, probs) if unique_labels > 1 else np.nan,

        "confusion_matrix": confusion_matrix(all_labels, preds, labels=[0, 1]),

        "probs": probs,
        "labels": all_labels
    }

    return metrics