import torch
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, average_precision_score)


def train_client(model, client_train_loader, criterion, optimizer, client_id, num_epochs,
                 client_val_loaders=None, print_last_only=False, last_round=True, device='cpu'):
    train_losses = []
    train_accuracies = []

    model = model.to(device)
    model.train()

    # WICHTIG: Sicherstellen, dass criterion MSELoss ist
    # criterion = nn.MSELoss() sollte von außen übergeben werden

    for epoch in range(num_epochs):
        running_loss = 0.0

        epoch_preds = []
        epoch_labels = []
        n_samples = 0

        for batch_idx, (images, labels) in enumerate(client_train_loader):
            # 1. Bilder vorbereiten
            images = images.view(images.size(0), -1).to(device)

            # 2. Labels vorbereiten
            # Original: 0 oder 1.
            # Wir brauchen sie aber als Float [batch, 1]
            labels = labels.float().view(-1, 1).to(device)

            # --- ÄNDERUNG FÜR MSE: LABEL-SHIFT ---
            # Wir wandeln Labels von {0, 1} zu {-1, 1} um.
            # Formel: new = (old * 2) - 1
            # 0 -> -1
            # 1 ->  1
            targets = (labels * 2) - 1

            optimizer.zero_grad()

            # Forward Pass (Gibt Werte in [-1, 1] zurück)
            outputs = model(images)

            # Loss berechnen (Vergleicht Output [-1,1] mit Targets [-1,1])
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            # --- METRIKEN ---
            running_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

            # 3. VORHERSAGE LOGIK (Ohne Sigmoid!)
            # Da der Bereich [-1, 1] ist, ist der Threshold 0.0
            # Alles > 0 ist Klasse 1, alles < 0 ist Klasse 0 (bzw. -1)

            # Wir wandeln es direkt zurück in 0/1 für die Metriken,
            # damit accuracy_score funktioniert.
            preds = (outputs > 0.0).float()

            epoch_preds.extend(preds.detach().cpu().numpy().flatten())
            epoch_labels.extend(labels.cpu().numpy().flatten())

        # --- EPOCHEN-ENDE ---
        avg_loss = running_loss / n_samples if n_samples > 0 else 0.0
        accuracy = accuracy_score(epoch_labels, epoch_preds) * 100.0

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        do_print = not print_last_only or (print_last_only and last_round and epoch == num_epochs - 1)
        if do_print:
            val_str = ""
            if client_val_loaders is not None and client_id in client_val_loaders:
                val_results = evaluate_model(model, client_val_loaders[client_id], criterion, device)
                val_acc = val_results[0]
                val_loss = val_results[1]
                val_str = f" | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}"

            print(f"Epoch [{epoch + 1}/{num_epochs}], Client {client_id}: "
                  f"Train Loss: {avg_loss:.6f}, Train Acc: {accuracy:.2f}%{val_str}")

    safe_weights = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

    return {
        "weights": safe_weights,
        "train_losses": train_losses,
        "train_accuracies": train_accuracies
    }



def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.eval()
    model = model.to(device)

    total_loss = 0.0
    n_samples = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if images is None or labels is None:
                continue

            images = images.view(images.size(0), -1).to(device)

            # Original Labels [0, 1] behalten wir für Metriken
            labels = labels.float().view(-1, 1).to(device)

            # Targets [-1, 1] erstellen für den MSE Loss
            targets = (labels * 2) - 1

            # Forward Pass ([-1, 1])
            outputs = model(images)

            # Loss (MSE)
            loss = criterion(outputs, targets)

            # VORHERSAGE LOGIK (Linear)
            # 1. Hard Prediction (0 oder 1)
            preds = (outputs > 0.0).float()

            # 2. "Pseudo"-Wahrscheinlichkeit für AUC berechnen
            # Wir mappen [-1, 1] auf [0, 1] linear um.
            # (-1 -> 0, 0 -> 0.5, 1 -> 1)
            probs = (outputs + 1) / 2.0

            # Sicherheitshalber clippen, falls Werte leicht außerhalb liegen
            probs = torch.clamp(probs, 0.0, 1.0)

            total_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    if n_samples == 0:
        return 0, 0, 0, np.nan, np.nan, 0, 0, np.zeros((2, 2)), np.array([]), np.array([]), {}

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)

    avg_loss = total_loss / n_samples

    # Klassische Metriken
    accuracy = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    if len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_probs)
            auc_pr = average_precision_score(y_true, y_probs)
        except ValueError:
            auc, auc_pr = np.nan, np.nan
    else:
        auc, auc_pr = np.nan, np.nan

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    class_f1_metrics = {
        "f1_class_0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "f1_class_1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0
    }

    return (accuracy, avg_loss, f1, auc, auc_pr, precision, recall,
            cm, y_probs, y_true, class_f1_metrics)