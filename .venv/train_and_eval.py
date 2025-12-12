import torch
import numpy as np
from config import *
from sklearn.metrics import roc_auc_score, f1_score, precision_score, average_precision_score, recall_score, accuracy_score, confusion_matrix


def train_client(model, client_train_loader, criterion, optimizer, client_id, num_epochs,
                 client_val_loaders=None, print_last_only=False, last_round=True):
    """
    Trainingsfunktion angepasst für Sigmoid + BCELoss.

    Wichtig:
    - model.forward() muss Sigmoid bereits anwenden (Outputs in [0, 1])
    - criterion muss nn.BCELoss() sein (NICHT BCEWithLogitsLoss)
    - Threshold für Prediction ist jetzt 0.5 (Wahrscheinlichkeit)
    """
    train_losses = []
    train_accuracies = []
    eval_results = {}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n_correct = 0
        n_samples = 0

        for batch_idx, (images, labels) in enumerate(client_train_loader):
            # Input preprocessing: Flatten und auf Device
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device).view(-1)

            # Targets müssen float sein (0.0 oder 1.0) für BCELoss
            targets = labels.float().view(-1, 1)

            # Gradienten zurücksetzen
            optimizer.zero_grad()

            # Forward pass
            # Outputs sind jetzt bereits Wahrscheinlichkeiten in [0, 1] durch Sigmoid
            outputs = model(images)  # Shape: (batch, 1), Wertebereich: [0, 1]

            # Loss berechnen (BCELoss, da Sigmoid bereits in forward())
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Loss akkumulieren
            running_loss += loss.item() * labels.size(0)

            # Predictions: Wahrscheinlichkeit > 0.5 → Klasse 1
            preds = (outputs > 0.5).float().view(-1)

            # Accuracy berechnen
            n_correct += (preds == labels).sum().item()
            n_samples += labels.size(0)

        # Epoch-Metriken berechnen
        if n_samples == 0:
            avg_loss = 0.0
            accuracy = 0.0
        else:
            avg_loss = running_loss / n_samples
            accuracy = 100.0 * n_correct / n_samples

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # Ausgabe und Evaluation
        do_print = not print_last_only or (print_last_only and last_round and (epoch == num_epochs - 1))

        if do_print:
            val_str = ""
            if client_val_loaders is not None and client_id in client_val_loaders:
                eval_out = evaluate_model(model, client_val_loaders[client_id], criterion)
                (val_accuracy, val_loss, val_f1, val_auc, val_auc_pr,
                 val_precision, val_recall, val_cm,
                 val_probs, val_labels, val_class_f1) = eval_out

                if last_round and epoch == num_epochs - 1:
                    eval_results = {
                        "accuracy": val_accuracy, "loss": val_loss, "f1": val_f1,
                        "auc": val_auc, "auc_pr": val_auc_pr, "precision": val_precision,
                        "recall": val_recall, "confusion_matrix": val_cm,
                        "probs": val_probs, "labels": val_labels, **val_class_f1
                    }
                val_str = f" | val acc: {val_accuracy:.2f}% | val loss: {val_loss:.4f}"

            print(f'Epoch [{epoch + 1}/{num_epochs}], client {client_id}, '
                  f'avg loss: {avg_loss:.6f}, train acc: {accuracy:.2f}%{val_str}')

    # Weights sicher kopieren (CPU)
    safe_weights = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

    results = {
        "weights": safe_weights,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies
    }

    if eval_results:
        results["eval_results"] = eval_results

    return results


def evaluate_model(model, test_loader, criterion):
    """
    Evaluiert Modell.
    Modell-Output: Logits (Erwartungswert Pauli-Z Observable in [-1, 1])
    Loss function: BCEWithLogitsLoss
    Labels: [0, 1]
    """
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if images is None or labels is None:
                continue

            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device).view(-1)
            targets = labels.float().view(-1, 1)

            logits = model(images)

            loss = criterion(logits, targets)


            # 1. probabilities: Logits -> Sigmoid -> [0, 1]
            probs_class_1 = torch.sigmoid(logits).view(-1)

            # 2. prediction (Logit > 0 entspricht prob > 0.5)
            preds = (logits > 0.0).float().view(-1)

            total_loss += loss.item() * labels.size(0)
            n_correct += (preds == labels).sum().item()
            n_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs_class_1.cpu().numpy())

    if n_samples == 0:
        return 0, 0, 0, np.nan, np.nan, 0, 0, np.zeros((2, 2)), np.array([]), np.array([]), {}

    accuracy = 100.0 * n_correct / n_samples
    avg_loss = total_loss / n_samples

    y_true = np.array(all_labels).astype(int)
    y_pred = np.array(all_preds).astype(int)
    y_probs = np.array(all_probs)

    # Standard Metriken (Beibehalten)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_probs)
            auc_pr = average_precision_score(y_true, y_probs)
        except ValueError:
            auc = np.nan
            auc_pr = np.nan
    else:
        auc = np.nan
        auc_pr = np.nan

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    class_f1_metrics = {
        "f1_class_0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "f1_class_1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
    }

    return (accuracy, avg_loss, f1, auc, auc_pr, precision, recall,
            cm, y_probs, y_true, class_f1_metrics)