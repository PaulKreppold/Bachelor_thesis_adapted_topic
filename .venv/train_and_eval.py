import torch
import numpy as np
from config import *
from sklearn.metrics import roc_auc_score, f1_score, precision_score, average_precision_score, recall_score, accuracy_score, confusion_matrix


def train_client(model, client_train_loader, criterion, optimizer, client_id, num_epochs,
                 client_val_loaders=None, print_last_only=False, last_round=True):

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

            images = images.view(images.size(0), -1).to(device)

            # Labels als float für BCELoss
            labels = labels.to(device).float()           # <-- WICHTIG
            labels_2d = labels.unsqueeze(1)               # [batch, 1]

            optimizer.zero_grad()

            # Model Output: [batch,1] with probabilities
            outputs = model(images)                      # [batch,1]

            # BCELoss
            loss = criterion(outputs, labels_2d)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            # Accuracy: Threshold
            preds = (outputs.detach() >= 0.5).long().view(-1)
            labels_long = labels.long()

            n_correct += (preds == labels_long).sum().item()
            n_samples += labels.size(0)

        if n_samples == 0:
            avg_loss = 0.0
            accuracy = 0.0
        else:
            avg_loss = running_loss / n_samples
            accuracy = 100.0 * n_correct / n_samples

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        # Validation
        val_accuracy = val_loss = None
        if client_val_loaders is not None and client_id in client_val_loaders:
            eval_out = evaluate_model(model, client_val_loaders[client_id], criterion)
            (val_accuracy, val_loss, val_f1, val_auc, val_auc_pr,
             val_precision, val_recall, val_cm,
             val_probs, val_labels, val_class_f1) = eval_out

            eval_results = {
                "accuracy": val_accuracy, "loss": val_loss, "f1": val_f1, "auc": val_auc,
                "auc_pr": val_auc_pr, "precision": val_precision, "recall": val_recall,
                "confusion_matrix": val_cm, "probs": val_probs, "labels": val_labels,
                **val_class_f1
            }

        # Logging
        do_print = not print_last_only or (print_last_only and last_round and (epoch == num_epochs - 1))
        if do_print:
            if val_accuracy is not None:
                print(f'Epoch [{epoch + 1}/{num_epochs}], client {client_id}, '
                      f'avg loss: {avg_loss:.6f}, train acc: {accuracy:.2f}% | '
                      f'val acc: {val_accuracy:.2f}% | val loss: {val_loss:.4f}')
            else:
                print(f'Epoch [{epoch + 1}/{num_epochs}], client {client_id}, '
                      f'avg loss: {avg_loss:.6f}, train acc: {accuracy:.2f}%')

    results = {
        "weights": model.state_dict(),
        "train_losses": train_losses,
        "train_accuracies": train_accuracies
    }

    if eval_results:
        results["eval_results"] = eval_results

    return results



def evaluate_model(model, test_loader, criterion):
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

            labels = labels.to(device).float()
            labels_2d = labels.unsqueeze(1)

            outputs = model(images)  # [batch,1], probabilities in [0,1]

            loss = criterion(outputs, labels_2d)

            # predictions
            preds = (outputs >= 0.5).long().view(-1)
            labels_long = labels.long()

            # For AUC: probability of class 1 is outputs itself
            probs_class_1 = outputs.view(-1)

            total_loss += loss.item() * labels.size(0)
            n_correct += (preds == labels_long).sum().item()
            n_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_long.cpu().numpy())
            all_probs.extend(probs_class_1.cpu().numpy())

    if n_samples == 0:
        return 0, 0, 0, 0, 0, 0, 0, np.zeros((2,2)), np.array([]), np.array([]), {}

    accuracy = 100.0 * n_correct / n_samples
    avg_loss = total_loss / n_samples

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

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

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0,1], zero_division=0)
    class_f1_metrics = {
        "f1_class_0": float(f1_per_class[0]),
        "f1_class_1": float(f1_per_class[1])
    }

    return (accuracy, avg_loss, f1, auc, auc_pr, precision, recall,
            cm, y_probs, y_true, class_f1_metrics)
