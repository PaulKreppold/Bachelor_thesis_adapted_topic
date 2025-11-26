import torch
import numpy as np
from config import *
from sklearn.metrics import roc_auc_score, f1_score, precision_score, average_precision_score, recall_score, accuracy_score, confusion_matrix


def train_client(model, client_train_loader, criterion, optimizer, client_id, num_epochs,
                 client_val_loaders=None, print_last_only=False, last_round=True):
    """
    Trainingsfunktion angepasst für Softmax / CrossEntropyLoss (2 Output-Neuronen).
    Erwartet Output-Shape [batch_size, 2].
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
            if images is None or labels is None:
                continue

            # Images flatten (wie gehabt)
            images = images.view(images.size(0), -1).to(device)

            # WICHTIG FÜR SOFTMAX: Labels müssen 1D sein und vom Typ Long (Integer)
            # .squeeze() entfernt Dimensionen der Größe 1 (z.B. [16, 1] -> [16])
            labels = labels.to(device).squeeze().long()

            optimizer.zero_grad()

            # 1. Forward Pass (gibt jetzt Shape [batch_size, 2] zurück)
            outputs = model(images)

            # 2. Loss berechnen (CrossEntropyLoss)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            # 3. Accuracy Berechnung (Softmax Logik)
            # Wir brauchen kein echtes Softmax für die Accuracy, nur den Index des Maximums.
            # _, preds gibt den Index (0 oder 1) zurück, der den höheren Wert hat.
            _, preds = torch.max(outputs.detach(), 1)

            n_correct += (preds == labels).sum().item()
            n_samples += labels.size(0)

        if n_samples == 0:
            avg_loss = 0.0
            accuracy = 0.0
        else:
            avg_loss = running_loss / n_samples
            accuracy = 100.0 * n_correct / n_samples

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # --- Validation ---
        val_accuracy = val_loss = None
        if client_val_loaders is not None and client_id in client_val_loaders:
            val_loader = client_val_loaders[client_id]
            # Aufruf der angepassten Eval-Funktion
            (val_accuracy, val_loss, val_f1, val_auc, val_auc_pr,
             val_precision, val_recall, val_cm, val_probs, val_labels, val_class_f1) = evaluate_model(model, val_loader,
                                                                                                      criterion)

            eval_results = {
                "accuracy": val_accuracy, "loss": val_loss, "f1": val_f1, "auc": val_auc,
                "auc_pr": val_auc_pr,
                "precision": val_precision, "recall": val_recall, "confusion_matrix": val_cm,
                "probs": val_probs, "labels": val_labels,
                **val_class_f1
            }

        # --- Druck-Logik ---
        do_print = not print_last_only or (print_last_only and last_round and (epoch == num_epochs - 1))
        if do_print:
            if val_accuracy is not None:
                print(f'Epoch [{epoch + 1}/{num_epochs}], client {client_id}, '
                      f'avg loss: {avg_loss:.6f}, train accuracy: {accuracy:.2f}% | '
                      f'validation accuracy: {val_accuracy:.2f}% | validation loss: {val_loss:.4f}')
            else:
                print(f'Epoch [{epoch + 1}/{num_epochs}], client {client_id}, '
                      f'avg loss: {avg_loss:.6f}, train accuracy: {accuracy:.2f}%')

    results = {
        "weights": model.state_dict(),
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
    }
    if eval_results:
        results["eval_results"] = eval_results

    return results


def evaluate_model(model, test_loader, criterion):
    """
    Evaluierungsfunktion angepasst für Softmax / CrossEntropyLoss (2 Output-Neuronen).
    """
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_samples = 0
    all_preds = []
    all_labels = []
    all_probs = []  # Hier speichern wir nur die Wahrscheinlichkeit für Klasse 1 (Pneumonie)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.view(images.size(0), -1).to(device)

            # Labels anpassen: Long und Shape [batch]
            labels = labels.to(device).squeeze().long()

            # 1. Forward Pass (Shape [batch, 2])
            outputs = model(images)

            # 2. Loss berechnen
            loss = criterion(outputs, labels)

            # 3. Wahrscheinlichkeiten berechnen (Softmax)
            # dim=1 sorgt dafür, dass sich Zeilen zu 1.0 summieren (P(0) + P(1) = 1)
            softmax_probs = torch.softmax(outputs, dim=1)

            # Für AUC und ROC brauchen wir die Wahrscheinlichkeit der "positiven Klasse" (1).
            # Wir nehmen also nur die zweite Spalte (Index 1).
            probs_class_1 = softmax_probs[:, 1]

            # 4. Hard Predictions (Index des Maximums: 0 oder 1)
            _, preds = torch.max(outputs, 1)

            n_correct += (preds == labels).sum().item()
            n_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs_class_1.cpu().numpy())

    # --- Metriken Berechnung ---
    accuracy = 100.0 * n_correct / n_samples
    avg_loss = total_loss / n_samples

    y_true = np.array(all_labels).flatten()
    y_pred = np.array(all_preds).flatten()
    y_probs = np.array(all_probs).flatten()  # Enthält nur P(Klasse=1)

    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs)
        auc_pr = average_precision_score(y_true, y_probs)
    else:
        auc = 0.0
        auc_pr = 0.0

    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0) if len(np.unique(y_true)) > 1 else np.array(
        [f1])
    f1_class_0 = float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0
    f1_class_1 = float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0

    class_f1_metrics = {"f1_class_0": f1_class_0, "f1_class_1": f1_class_1}

    return (accuracy, avg_loss, f1, auc, auc_pr, precision, recall, cm, np.array(all_probs),
            np.array(all_labels), class_f1_metrics)