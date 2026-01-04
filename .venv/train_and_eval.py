import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, seed, device):
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    # Threshold für Early Stopping: Wenn die LR unter 1e-5 fällt,
    # stoppen wir, da keine nennenswerten Updates mehr zu erwarten sind.
    min_lr_threshold = 1.1e-5

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # Aktuelle LR am Anfang der Epoche für das Logging
        current_lr = optimizer.param_groups[0]['lr']

        for inputs, targets in train_loader:
            # BCE braucht Labels als Float (0.0 und 1.0)
            targets = targets.to(device).float().view(-1, 1)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Accuracy (Schwellenwert 0.0 für Logits)
            preds = (outputs > 0).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        # Evaluation nach der Epoche
        val_loss, val_acc, cm = evaluate_model(model, val_loader, criterion, device)

        # LR Scheduler Update basierend auf Val-Loss
        scheduler.step(val_loss)

        # Metriken berechnen
        train_loss = running_loss / total
        train_acc = correct / total

        # History befüllen
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Logging
        print(
            f"Seed {seed:4} | Ep [{epoch + 1:02d}/{epochs}] "
            f"L: {train_loss:.4f} | A: {train_acc:.4f} | "
            f"Val-L: {val_loss:.4f} | Val-A: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # --- EARLY STOPPING CHECK ---
        # Wir prüfen die NEUE Lernrate für die nächste Epoche
        next_lr = optimizer.param_groups[0]['lr']
        if next_lr < min_lr_threshold:
            print(
                f"\n[Early Stopping] Seed {seed}: Lernrate {next_lr:.7f} hat das Minimum unterschritten. Training beendet.")
            break

    return history


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            # Für BCE: Labels zu Float und (N, 1)
            targets_bce = targets.to(device).float().view(-1, 1)
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets_bce)
            running_loss += loss.item() * inputs.size(0)

            # Schwellenwert 0 für Logits
            preds = (outputs > 0).float()

            correct += (preds == targets_bce).sum().item()
            total += targets_bce.size(0)

            # In flache Listen umwandeln für Confusion Matrix
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())  # targets sind hier noch auf CPU/Long

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    return running_loss / total, correct / total, cm