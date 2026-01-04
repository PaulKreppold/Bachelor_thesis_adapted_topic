import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, seed, device):
    """
    Trainiert das VQC-Modell unter Verwendung von BCELoss (für Wahrscheinlichkeits-Outputs).
    """
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    final_cm = None

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)

            optimizer.zero_grad()

            # Forward Pass: Liefert Wahrscheinlichkeiten [0, 1]
            outputs = model(inputs)

            # WICHTIG: Nutze nn.BCELoss() im Hauptskript als 'criterion'
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # Klassifizierung: Da Output [0, 1], ist der Schwellenwert 0.5
            preds = (outputs >= 0.5).float()

            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        # Validierung
        val_loss, val_acc, cm = evaluate_model(model, val_loader, criterion, device)

        if epoch == epochs - 1:
            final_cm = cm

        train_loss = running_loss / total
        train_acc = correct / total
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Seed {seed:4} | Ep [{epoch + 1:02d}/{epochs}] "
              f"L: {train_loss:.4f} | A: {train_acc:.4f} | "
              f"Val-L: {val_loss:.4f} | Val-A: {val_acc:.4f}")

    return history, final_cm


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            # Schwellenwert 0.5 für Wahrscheinlichkeiten
            preds = (outputs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / total
    avg_acc = correct / total
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return avg_loss, avg_acc, cm