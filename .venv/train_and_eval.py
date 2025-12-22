import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def train_client(model, train_loader, optimizer, client_id, num_epochs, val_loader=None, device="cpu",
                 mode_label="Pure"):
    # BCEWithLogitsLoss ist numerisch stabiler
    criterion = nn.BCEWithLogitsLoss()
    history = {"train_losses": [], "train_accuracies": []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().view(-1, 1).to(device)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Bei Logits: Alles > 0.0 ist Klasse 1
            preds = (logits >= 0.0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_acc = 100.0 * correct / total
        epoch_loss = running_loss / len(train_loader)
        history["train_losses"].append(epoch_loss)
        history["train_accuracies"].append(epoch_acc)

        print(
            f"[{client_id} | {mode_label}] Ep [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

    return history


def evaluate_model(model, loader, device="cpu"):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().view(-1, 1).to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels).flatten()
    all_probs = np.array(all_probs).flatten()
    all_preds = (all_probs >= 0.5).astype(float)

    acc = accuracy_score(all_labels, all_preds) * 100.0

    # PRO-TIP: labels=[0, 1] erzwingt, dass IMMER zwei Werte zurückkommen.
    # Der erste Wert ist immer Klasse 0, der zweite immer Klasse 1.
    f1_scores = f1_score(all_labels, all_preds, labels=[0, 1], average=None, zero_division=0)

    f1_normal = f1_scores[0]
    f1_krank = f1_scores[1]

    return acc, (f1_normal, f1_krank), all_probs, all_labels


def visualize_batch_analysis(model, test_loader, client_id, mode_label, device):
    model.eval()
    scale_param = getattr(model, 'scale', None)
    bias_param = getattr(model, 'bias', None)
    current_scale = scale_param.item() if scale_param is not None else 1.0
    current_bias = bias_param.item() if bias_param is not None else 0.0

    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        # 1. Quanten-Output (Erwartungswert PauliZ: -1 bis 1)
        q_raw = model.qlayer(images.view(images.size(0), -1)).view(-1)
        # 2. Logits berechnen
        logits = (q_raw * current_scale) + current_bias
        # 3. In Wahrscheinlichkeit umrechnen
        probs = torch.sigmoid(logits)

        res = []
        for i in range(len(images)):
            x = q_raw[i].item()
            p = probs[i].item()
            gt = int(labels[i].item())
            pred = 1 if p >= 0.5 else 0

            res.append({
                "Bild": f"Bild {i + 1:02d}",
                "VQC Raw (x)": f"{x:8.4f}",
                "Prob": f"{p * 100:6.1f}%",
                "GT": gt,
                "Pred": pred,
                "Ergebnis": "✅" if pred == gt else "❌"
            })

    df = pd.DataFrame(res)
    print(f"\n" + "=" * 85)
    print(f" BATCH ANALYSE: {client_id} ({mode_label}) ".center(85))
    print("-" * 85)
    print(f" GELERNTE PARAMETER:  Scale = {current_scale:.4f}  |  Bias = {current_bias:.4f}")
    print(f" MATHEMATIK:          Prob = Sigmoid({current_scale:.2f} * Raw_x + {current_bias:.2f})")
    print("-" * 85)
    print(df.to_string(index=False))
    print("=" * 85 + "\n")