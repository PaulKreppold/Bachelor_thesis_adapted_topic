import torch
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import time


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, device):
    """
    Trainiert das Modell und behebt den Dtype-Fehler (Labels zu Float).
    """
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

    # Progress Bar Initialisierung
    pbar = tqdm(range(epochs), desc="Initialisierung", unit="epoch", leave=False)

    for epoch in pbar:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            # FIX: Labels zu float konvertieren für BCELoss
            inputs, labels = inputs.to(device), labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(inputs)

            # Form-Check (Sicherheitsmaßnahme gegen Broadcasting-Fehler)
            loss = criterion(outputs, labels.view_as(outputs))

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels.view_as(outputs)).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validierung am Epochenende
        val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device)

        history['loss'].append(float(epoch_loss))
        history['acc'].append(float(epoch_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))

        # Nutzung der epoch-Variable für die Anzeige
        pbar.set_description(f"Epoch {epoch + 1}/{epochs}")
        pbar.set_postfix({
            'Loss': f"{epoch_loss:.3f}",
            'Acc': f"{epoch_acc:.3f}",
            'ValAcc': f"{val_acc:.3f}"
        })

    return history


def evaluate_model(model, loader, criterion, device):
    """
    Evaluiert das Modell und liefert Metriken.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            loss = criterion(outputs, labels.view_as(outputs))

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = np.mean(np.array(all_preds).flatten() == np.array(all_labels).flatten())
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return avg_loss, acc, cm


def save_experiment_results(stats_dir, models_dir, file_name, history, test_metrics, model, duration=None):
    """
    Speichert Metriken als JSON und Modellgewichte als .pth.
    """
    results = {
        "history": history,
        "test_metrics": {
            "loss": float(test_metrics['loss']),
            "acc": float(test_metrics['acc']),
            "cm": test_metrics['cm'].tolist()
        }
    }

    if duration:
        results["duration_seconds"] = duration

    # JSON Speichern (Statistiken)
    with open(os.path.join(stats_dir, f"{file_name}.json"), 'w') as f:
        json.dump(results, f, indent=4)

    # Modell Speichern (Gewichte)
    torch.save(model.state_dict(), os.path.join(models_dir, f"{file_name}.pth"))