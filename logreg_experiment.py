import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from helpers import plot_decision_boundaries
from models import Logistic_Regression

np.random.seed(42)
torch.manual_seed(42)


def load_split_data():
    # train
    train_df = pd.read_csv("train.csv")
    X_train = train_df[["long", "lat"]].values.astype(np.float32)
    y_train = train_df["country"].values.astype(np.int64)

    # validation
    val_df = pd.read_csv("validation.csv")
    X_val = val_df[["long", "lat"]].values.astype(np.float32)
    y_val = val_df["country"].values.astype(np.int64)

    # test
    test_df = pd.read_csv("test.csv")
    X_test = test_df[["long", "lat"]].values.astype(np.float32)
    y_test = test_df["country"].values.astype(np.int64)

    return X_train, y_train, X_val, y_val, X_test, y_test


def accuracy_from_logits(logits, y_true):
    preds = torch.argmax(logits, dim=1)
    return (preds == y_true).float().mean().item()


def evaluate(model, X, y, criterion, batch_size=32):
    model.eval()
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def train_logreg_for_lr(lr, X_train, y_train, X_val, y_val, X_test, y_test,
                        num_epochs=10, batch_size=32):
    input_dim = 2
    output_dim = 2

    model = Logistic_Regression(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses, test_losses = [], [], []
    train_accs,   val_accs,   test_accs   = [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_correct = 0
        total_samples = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

        avg_train_loss = epoch_loss / total_samples
        avg_train_acc = total_correct / total_samples

        # evaluate on val & test
        val_loss, val_acc = evaluate(model, X_val, y_val, criterion, batch_size)
        test_loss, test_acc = evaluate(model, X_test, y_test, criterion, batch_size)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        train_accs.append(avg_train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"[lr={lr}] Epoch {epoch+1}/{num_epochs} "
              f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.3f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
              f"test_acc={test_acc:.3f}")

    return model, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()
    learning_rates = [0.1, 0.01, 0.001]

    best_val_acc = -1.0
    best_lr = None
    best_model = None
    best_history = None

    for lr in learning_rates:
        model, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = \
            train_logreg_for_lr(lr, X_train, y_train, X_val, y_val, X_test, y_test)


        final_val_acc = val_accs[-1]
        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_lr = lr
            best_model = model
            best_history = (train_losses, val_losses, test_losses,
                            train_accs, val_accs, test_accs)

    print(f"\nBest learning rate (by validation): {best_lr}, "
          f"val acc={best_val_acc:.3f}")

    # Q1: visualize test predictions of best model
    plot_decision_boundaries(best_model, X_test, y_test,
                             title=f"Logistic Regression (best lr={best_lr})")

    # Q2: plot losses over epochs for the best model
    train_losses, val_losses, test_losses, train_accs, val_accs, test_accs = best_history
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='train loss')
    plt.plot(epochs, val_losses,   marker='o', label='validation loss')
    plt.plot(epochs, test_losses,  marker='o', label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Logistic Regression Loss vs Epoch (lr={best_lr})')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()