import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, data_loader, device, minority_idx=None):
    """
    Führt eine umfassende Evaluierung des Modells durch.
    Inkludiert Metriken für die Minderheitsklasse und Macro-Durchschnitte.
    """
    model.eval()
    criterion = torch.nn.BCELoss()
    running_loss, total = 0.0, 0
    all_probs, all_preds, all_targets = [], [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            probs = outputs.detach().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))
            all_targets.extend(targets.cpu().numpy().flatten())

    # WICHTIG: Fix für TypeError (Umwandlung in NumPy-Arrays für mathematische Operationen)
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Ermittlung des minority_idx falls nicht gesetzt
    if minority_idx is None:
        counts = np.bincount(all_targets.astype(int), minlength=2)
        minority_idx = np.argmin(counts) if counts[0] != counts[1] else 1

    # Per-Class Metriken (Precision, Recall, F1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=[0, 1], average=None, zero_division=0
    )

    # Macro-Metriken (Ungewichteter Durchschnitt über beide Klassen)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0
    )

    # PR-AUC Berechnung (Fokus auf die Minderheitsklasse für LDS-Studie [cite: 17, 18])
    if minority_idx == 1:
        pr_auc_min = average_precision_score(all_targets, all_probs)
    else:
        # Flippen der Labels/Probs, falls Klasse 0 die Minderheit ist
        pr_auc_min = average_precision_score(1 - all_targets, 1 - all_probs)

    # ROC-AUC (Globale Trennschärfe )
    roc_auc = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.5

    evaluation_results = {
        'Testverlust': running_loss / total,
        'Testgenauigkeit': accuracy_score(all_targets, all_preds),
        'Balancierte_Genauigkeit': balanced_accuracy_score(all_targets, all_preds),

        # Minderheits-Metriken (Zentral für Non-IID / LDS Analyse [cite: 17, 24])
        'F1_Minderheit': f1[minority_idx],
        'Recall_Minderheit': rec[minority_idx],
        'Precision_Minderheit': prec[minority_idx],
        'PR_AUC_Minority': pr_auc_min,

        # Macro-Metriken (Robustheit & Fairness-Indikatoren )
        'F1_Macro': f1_macro,
        'Recall_Macro': rec_macro,
        'Precision_Macro': prec_macro,

        'ROC_AUC_Global': roc_auc,
        'cm': confusion_matrix(all_targets, all_preds, labels=[0, 1]).tolist(),
        'minority_idx': int(minority_idx)
    }
    return evaluation_results


def train_local_model(model, train_loader, val_loader, optimizer, epochs, device, client_id=""):
    """
    Trainiert ein lokales Modell und protokolliert den Fortschritt[cite: 21, 27].
    """
    model.train()
    criterion = torch.nn.BCELoss()
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            preds = (outputs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * inputs.size(0)

        # Training-Stats speichern
        history['train_loss'].append(running_loss / total)
        history['train_acc'].append(correct / total)

        # Validierung pro Epoche (für Konvergenzkurven )
        model.eval()
        v_loss, v_corr, v_total = 0.0, 0, 0
        with torch.no_grad():
            for vi, vt in val_loader:
                vi, vt = vi.to(device), vt.to(device).float().view(-1, 1)
                vo = model(vi)
                v_loss += criterion(vo, vt).item() * vi.size(0)
                v_corr += ((vo >= 0.5).float() == vt).sum().item()
                v_total += vt.size(0)

        history['val_loss'].append(v_loss / v_total)
        history['val_acc'].append(v_corr / v_total)

        # Protokollierung (Wichtig für die Überwachung langer Läufe [cite: 22])
        if (epoch + 1) % 1 == 0 or epoch == epochs - 1:
            print(f"[{client_id}] Ep {epoch + 1}/{epochs} | "
                  f"train acc: {history['train_acc'][-1]:.3f}, val acc: {history['val_acc'][-1]:.3f} | "
                  f"train loss: {history['train_loss'][-1]:.3f}, val loss: {history['val_loss'][-1]:.3f}")

    return model.state_dict(), history