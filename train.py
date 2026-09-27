import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

from models import MLP, CNN

torch.manual_seed(42)

# --- Data Pipeline ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = 50000
val_size = 10000
train_set, val_set = random_split(full_train, [train_size, val_size])

BATCH_SIZE = 64
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return history

EPOCHS = 10

print("\n=== Training MLP ===")
mlp_model = MLP()
mlp_history = train_model(mlp_model, train_loader, val_loader, epochs=EPOCHS)

print("\n=== Training CNN ===")
cnn_model = CNN()
cnn_history = train_model(cnn_model, train_loader, val_loader, epochs=EPOCHS)

def plot_history(mlp_hist, cnn_hist, epochs):
    epochs_range = range(1, epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs_range, mlp_hist["train_loss"], label="MLP Train")
    axes[0].plot(epochs_range, mlp_hist["val_loss"], label="MLP Val")
    axes[0].plot(epochs_range, cnn_hist["train_loss"], label="CNN Train")
    axes[0].plot(epochs_range, cnn_hist["val_loss"], label="CNN Val")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy plot
    axes[1].plot(epochs_range, mlp_hist["train_acc"], label="MLP Train")
    axes[1].plot(epochs_range, mlp_hist["val_acc"], label="MLP Val")
    axes[1].plot(epochs_range, cnn_hist["train_acc"], label="CNN Train")
    axes[1].plot(epochs_range, cnn_hist["val_acc"], label="CNN Val")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("\nSaved training_curves.png")
    plt.show()

plot_history(mlp_history, cnn_history, EPOCHS)


def evaluate_on_test(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    return all_labels, all_preds, test_acc

print("\n=== Evaluating CNN on Test Set ===")
true_labels, pred_labels, cnn_test_acc = evaluate_on_test(cnn_model, test_loader)

cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"CNN Confusion Matrix (Test Acc: {cnn_test_acc:.4f})")
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")
plt.show()

# Ən çox qarışan cütü tap (diaqonal xaric ən böyük dəyər)
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
max_idx = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
print(f"\nƏn çox qarışan cüt: True={max_idx[0]}, Predicted={max_idx[1]} ({cm_no_diag[max_idx]} dəfə)")


torch.save(cnn_model.state_dict(), "best_cnn.pth")
print("\nModel saved to best_cnn.pth")