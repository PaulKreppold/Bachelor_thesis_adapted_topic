import torch

from sklearn.metrics import confusion_matrix
import seaborn as sns


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, seed, device):
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}
    final_cm = None

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            targets_mapped = 2 * targets - 1

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets_mapped)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        # Validierung am Ende jeder Epoche
        val_loss, val_acc, cm = evaluate_model(model, val_loader, criterion, device)

        # In der letzten Epoche speichern wir die Matrix
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
    """
    Evaluiert das Modell und gibt Loss, Acc und Confusion Matrix zurück.
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)

            # Mapping von [0, 1] auf [-1, 1] für den Loss
            targets_mapped = 2 * targets - 1
            outputs = model(inputs)
            loss = criterion(outputs, targets_mapped)

            running_loss += loss.item() * inputs.size(0)

            # Vorhersage: Klasse 1 wenn > 0, sonst Klasse 0
            preds = (outputs > 0).float()

            # Für Metriken sammeln
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / total
    avg_acc = correct / total

    # Confusion Matrix berechnen (0 = Negative, 1 = Positive)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])

    return avg_loss, avg_acc, cm