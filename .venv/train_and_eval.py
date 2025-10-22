import torch
from config import *

def train_client(model, client_train_loader, criterion, optimizer, client_id, num_epochs,
                 client_val_loaders=None):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        n_correct = 0
        n_samples = 0

        for batch_idx, (images, labels) in enumerate(client_train_loader):
            images = images.view(images.size(0), -1).to(device)
            labels = labels.float().view(-1, 1).to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            n_correct += (preds == labels).sum().item()
            n_samples += labels.size(0)

        avg_loss = running_loss / len(client_train_loader)
        accuracy = 100.0 * n_correct / n_samples

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Client {client_id}, '
              f'Avg Loss: {avg_loss:.6f}, Train Accuracy: {accuracy:.2f}%')

        if client_val_loaders is not None and client_id in client_val_loaders:
            val_loader = client_val_loaders[client_id]
            val_accuracy, val_loss = evaluate_model(model, val_loader, criterion)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)
            print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}% | Validation Loss: {val_loss:.4f}")

    results = {
        "weights": model.state_dict(),
        "losses": train_losses,
        "accuracies": train_accuracies
    }
    return results


def evaluate_model(model, test_loader, criterion):
    model.eval()

    total_loss = 0.0
    n_correct = 0
    n_samples = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.view(images.size(0), -1).to(device)
            labels = labels.float().view(-1, 1).to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            n_correct += (preds == labels).sum().item()
            n_samples += labels.size(0)

            total_loss += loss.item() * labels.size(0)

    accuracy = 100.0 * n_correct / n_samples
    avg_loss = total_loss / n_samples

    return accuracy, avg_loss