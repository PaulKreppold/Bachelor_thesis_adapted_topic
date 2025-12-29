import torch


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, use_bce, patience):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)

            if use_bce:
                t_float = targets.float().view_as(outputs)
                preds = (outputs >= 0.5).long()
            else:
                t_float = targets.float().view_as(outputs) * 2.0 - 1.0
                preds = (outputs >= 0.0).long()

            loss = criterion(outputs, t_float)
            loss.backward()
            optimizer.step()

            correct += (preds.view_as(targets) == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, use_bce)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # KOMPAKTES LOGGING: Nur alle 10 Epochen ODER in der ersten/letzten Epoche
        is_last_epoch = (epoch == epochs - 1)
        if (epoch + 1) % 10 == 0 or epoch == 0 or is_last_epoch:
            print(
                f"    Ep [{epoch + 1:02d}/{epochs}] L: {train_loss:.4f} | Val-L: {val_loss:.4f} | Val-A: {val_acc:.3f}")

        # Early Stopping Check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            # Wenn Early Stopping greift, printen wir den Endstand dieser Runde
            print(f"    >>> Early Stopping in Ep {epoch + 1} (Best Val-L: {best_val_loss:.4f})")
            break

    return history


def evaluate_model(model, data_loader, criterion, use_bce):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            if use_bce:
                t_float = targets.float().view_as(outputs)
                preds = (outputs >= 0.5).long()
            else:
                t_float = targets.float().view_as(outputs) * 2.0 - 1.0
                preds = (outputs >= 0.0).long()

            loss = criterion(outputs, t_float)
            running_loss += loss.item() * inputs.size(0)
            correct += (preds.view_as(targets) == targets).sum().item()
            total += targets.size(0)
    return running_loss / total, correct / total