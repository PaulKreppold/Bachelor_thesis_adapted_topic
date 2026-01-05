import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, seed, device, scenario):
    # Loss-Typ wählen
    criterion = torch.nn.CrossEntropyLoss() if scenario['measurement'] == 'softmax' else torch.nn.BCELoss()
    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    min_lr_threshold = 1.1e-5

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        current_lr = optimizer.param_groups[0]['lr']

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            if scenario['measurement'] == 'softmax':
                targets = targets.to(device).long().view(-1)
            else:
                targets = targets.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if scenario['measurement'] == 'softmax':
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = (outputs >= 0.5).float()

            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        val_loss, val_acc, _ = evaluate_model(model, val_loader, criterion, device, scenario)
        scheduler.step(val_loss)

        history['loss'].append(running_loss / total)
        history['acc'].append(correct / total)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"{scenario['name']} | Ep {epoch + 1:02d} | L: {running_loss / total:.4f} | Val-A: {val_acc:.4f}")
        if current_lr < min_lr_threshold: break

    return history


def evaluate_model(model, data_loader, criterion, device, scenario):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            if scenario['measurement'] == 'softmax':
                targets_eval = targets.to(device).long().view(-1)
            else:
                targets_eval = targets.to(device).float().view(-1, 1)

            outputs = model(inputs)
            loss = criterion(outputs, targets_eval)
            running_loss += loss.item() * inputs.size(0)

            if scenario['measurement'] == 'softmax':
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
            else:
                preds = (outputs >= 0.5).float()
                all_preds.extend(preds.cpu().numpy().flatten())

            total += targets_eval.size(0)
            correct += (preds == targets_eval).sum().item()
            all_targets.extend(targets.numpy().flatten())

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    return running_loss / total, correct / total, cm