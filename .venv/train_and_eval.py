import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, seed, device,
                is_weighted=False):
    """
    Trainings-Loop angepasst für:
    - Wahrscheinlichkeits-Outputs (0 bis 1)
    - Optionalen Weighted BCELoss (Klassengewichtung gegen Bias)
    - Early Stopping basierend auf LR-Threshold
    """
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    min_lr_threshold = 1.1e-5

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']

        for inputs, targets in train_loader:
            targets = targets.to(device).float().view(-1, 1)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # --- LOSS BERECHNUNG ---
            if is_weighted:
                # BCELoss muss hier mit reduction='none' initialisiert sein
                raw_loss = criterion(outputs, targets)
                # Gewichtung: Klasse 0 (Gesund) bekommt ca. 2.8, Klasse 1 (Krank) 1.0
                weights = torch.where(targets == 0, 2.8, 1.0).to(device)
                loss = (raw_loss * weights).mean()
            else:
                loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # Accuracy (Schwellenwert 0.5)
            preds = (outputs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        # Validierung nach der Epoche
        val_loss, val_acc, cm = evaluate_model(model, val_loader, criterion, device, is_weighted=is_weighted)

        # LR Scheduler Update (basiert auf Val-Loss)
        scheduler.step(val_loss)

        train_loss = running_loss / total
        train_acc = correct / total

        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(
            f"Seed {seed:4} | Ep [{epoch + 1:02d}/{epochs}] "
            f"L: {train_loss:.4f} | A: {train_acc:.4f} | "
            f"Val-L: {val_loss:.4f} | Val-A: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Early Stopping Check (wenn die LR zu weit sinkt, lernt das Modell nichts mehr)
        next_lr = optimizer.param_groups[0]['lr']
        if next_lr < min_lr_threshold:
            print(f"\n[Early Stopping] Seed {seed}: LR {next_lr:.7f} unterschritten.")
            break

    return history


def evaluate_model(model, data_loader, criterion, device, is_weighted=False):
    """
    Evaluations-Loop:
    - Robust gegenüber skalaren und vektorisierten Loss-Funktionen (weighted)
    - Berechnet Confusion Matrix für finale Analyse
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            targets_bce = targets.to(device).float().view(-1, 1)
            inputs = inputs.to(device)

            outputs = model(inputs)

            # --- LOSS BERECHNUNG (Analog zum Training) ---
            if is_weighted:
                raw_loss = criterion(outputs, targets_bce)
                weights = torch.where(targets_bce == 0, 2.8, 1.0).to(device)
                loss = (raw_loss * weights).mean()
            else:
                loss = criterion(outputs, targets_bce)

            running_loss += loss.item() * inputs.size(0)

            # Vorhersagen (0.5 Schwellenwert)
            preds = (outputs >= 0.5).float()

            correct += (preds == targets_bce).sum().item()
            total += targets_bce.size(0)

            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())

    avg_loss = running_loss / total
    avg_acc = correct / total
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return avg_loss, avg_acc, cm