import torch

from sklearn.metrics import confusion_matrix
import seaborn as sns


# ==========================================
# 4. TRAINING & EVALUATION LOGIC
# ==========================================
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, seed, device):
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

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

        val_loss, val_acc, cm = evaluate_model(model, val_loader, criterion, device)

        train_loss = running_loss / total
        train_acc = correct / total
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(
            f"Seed {seed:4} | Ep [{epoch + 1:02d}/{epochs}] L: {train_loss:.4f} | A: {train_acc:.4f} | Val-L: {val_loss:.4f} | Val-A: {val_acc:.4f}")

    return history


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            targets_mapped = 2 * targets - 1
            outputs = model(inputs)
            loss = criterion(outputs, targets_mapped)

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    return running_loss / total, correct / total, confusion_matrix(all_targets, all_preds, labels=[0, 1])