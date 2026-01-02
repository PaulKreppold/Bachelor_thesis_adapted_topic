import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, epochs, seed, device):
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']

        for inputs, targets in train_loader:
            # 1️⃣ BCE braucht Labels als 0.0 und 1.0 (float)
            # targets kommt meist als Long [0, 1] aus dem MedMNIST Loader
            targets = targets.to(device).float().view(-1, 1)
            inputs = inputs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs) # VQC Output Bereich [-1, 1]

            # 2️⃣ BCE With Logits Loss
            # Erwartet (Logits, Targets). Da outputs in [-1, 1] liegt,
            # wird 0.0 als 50% Wahrscheinlichkeit interpretiert.
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # 3️⃣ Accuracy berechnen
            # Schwellenwert 0.0 (Logit 0.0 entspricht Sigmoid 0.5)
            preds = (outputs > 0).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        # Evaluation nach Epoche
        val_loss, val_acc, cm = evaluate_model(model, val_loader, criterion, device)

        # LR Scheduler basierend auf Validation Loss anpassen
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

    return history


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            # Labels für BCE vorbereiten
            targets_bce = targets.to(device).float().view(-1, 1)
            inputs = inputs.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets_bce)
            running_loss += loss.item() * inputs.size(0)

            # Accuracy (Threshold 0.0)
            preds = (outputs > 0).float()
            correct += (preds == targets_bce).sum().item()
            total += targets_bce.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy()) # Originale 0/1 Labels

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    return running_loss / total, correct / total, cm