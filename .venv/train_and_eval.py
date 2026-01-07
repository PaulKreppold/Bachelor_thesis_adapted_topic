import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, seed, device):
    """
    Trainings-Loop für das Super-Modell:
    Gibt nun Train-Loss, Train-Acc, Val-Loss und Val-Acc pro Epoche aus.
    """
    criterion = torch.nn.BCELoss()
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    min_lr_threshold = 1.1e-5

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Accuracy (Schwellenwert 0.5)
            preds = (outputs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        # Metriken berechnen
        train_loss = running_loss / total
        train_acc = correct / total

        # Validierung nach der Epoche
        val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device)

        # LR Scheduler Update (basiert auf Val-Loss)
        scheduler.step(val_loss)

        # Historie speichern
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # ERWEITERTES PRINT-STATEMENT
        print(
            f"Seed {seed:4} | Ep [{epoch + 1:02d}/{epochs}] | "
            f"Train-L: {train_loss:.4f} | Train-A: {train_acc:.4f} | "
            f"Val-L: {val_loss:.4f} | Val-A: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Early Stopping Check
        if optimizer.param_groups[0]['lr'] < min_lr_threshold:
            print(f"\n[Early Stopping] Seed {seed}: Lernrate {optimizer.param_groups[0]['lr']:.7f} unterschritten.")
            break

    return history


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluations-Loop zur Berechnung von Loss, Accuracy und Confusion Matrix.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets_bce = targets.to(device).float().view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets_bce)
            running_loss += loss.item() * inputs.size(0)

            preds = (outputs >= 0.5).float()
            correct += (preds == targets_bce).sum().item()
            total += targets_bce.size(0)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    avg_loss = running_loss / total
    avg_acc = correct / total
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return avg_loss, avg_acc, cm