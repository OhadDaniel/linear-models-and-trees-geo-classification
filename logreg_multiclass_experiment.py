import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from helpers import plot_decision_boundaries
from models import Logistic_Regression

np.random.seed(42)
torch.manual_seed(42)


def load_split_data_multiclass():
    train_df = pd.read_csv("train_multiclass.csv")
    X_train = train_df[["long", "lat"]].values.astype(np.float32)
    y_train = train_df["country"].values.astype(np.int64)

    val_df = pd.read_csv("validation_multiclass.csv")
    X_val = val_df[["long", "lat"]].values.astype(np.float32)
    y_val = val_df["country"].values.astype(np.int64)

    test_df = pd.read_csv("test_multiclass.csv")
    X_test = test_df[["long", "lat"]].values.astype(np.float32)
    y_test = test_df["country"].values.astype(np.int64)

    num_classes = len(np.unique(y_train))
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes


def evaluate(model, X, y, criterion, batch_size=32):
    model.eval()
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

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


def train_logreg_multiclass_for_lr(lr_init, X_train, y_train,
                                   X_val, y_val, X_test, y_test,
                                   num_classes, num_epochs=30, batch_size=32):

    input_dim = 2
    output_dim = num_classes

    model = Logistic_Regression(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.3)  # decay lr every 5 epochs

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    train_losses, val_losses, test_losses = [], [], []
    train_accs, val_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == yb).sum().item()
            total_samples += xb.size(0)

        avg_train_loss = total_loss / total_samples
        avg_train_acc = total_correct / total_samples

        val_loss, val_acc = evaluate(model, X_val, y_val, criterion, batch_size)
        test_loss, test_acc = evaluate(model, X_test, y_test, criterion, batch_size)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        train_accs.append(avg_train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(f"[lr={lr_init}] Epoch {epoch+1}/{num_epochs} "
              f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.3f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
              f"test_acc={test_acc:.3f}, lr={scheduler.get_last_lr()[0]:.5f}")

        scheduler.step()

    return model, train_losses, val_losses, test_losses, train_accs, val_accs, test_accs


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = \
        load_split_data_multiclass()

    learning_rates = [0.01, 0.001, 0.0003]

    results = []
    best_val_acc = -1.0
    best_model = None
    best_lr = None
    best_history = None

    for lr in learning_rates:
        (model,
         train_losses, val_losses, test_losses,
         train_accs, val_accs, test_accs) = train_logreg_multiclass_for_lr(
            lr, X_train, y_train, X_val, y_val, X_test, y_test, num_classes)

        final_val_acc = val_accs[-1]
        final_test_acc = test_accs[-1]
        results.append((lr, final_val_acc, final_test_acc))

        if final_val_acc > best_val_acc:
            best_val_acc = final_val_acc
            best_lr = lr
            best_model = model
            best_history = (train_losses, val_losses, test_losses,
                            train_accs, val_accs, test_accs)

    # Q1: plot val & test accuracy vs learning rate
    lrs = [r[0] for r in results]
    val_final = [r[1] for r in results]
    test_final = [r[2] for r in results]

    plt.figure()
    plt.plot(lrs, val_final, marker='o', label='validation accuracy')
    plt.plot(lrs, test_final, marker='o', label='test accuracy')
    plt.xscale('log')  # אופציונלי, כי אלה לוג-סקייל
    plt.xlabel('initial learning rate')
    plt.ylabel('accuracy (final epoch)')
    plt.title('Multiclass Logistic Regression: accuracy vs learning rate')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\nBest lr (by validation): {best_lr}, val acc={best_val_acc:.3f}")
    best_train_losses, best_val_losses, best_test_losses, \
        best_train_accs, best_val_accs, best_test_accs = best_history

    # Q1 – visualize predictions of best model
    plot_decision_boundaries(best_model, X_test, y_test,
                             title=f"Multiclass Logistic Regression (best lr={best_lr})")

    # Q2 – plot losses over epochs
    epochs = np.arange(1, len(best_train_losses) + 1)

    plt.figure()
    plt.plot(epochs, best_train_losses, marker='o', label='train loss')
    plt.plot(epochs, best_val_losses, marker='o', label='validation loss')
    plt.plot(epochs, best_test_losses, marker='o', label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Multiclass Logistic Regression Loss vs Epoch (lr={best_lr})')
    plt.legend()
    plt.grid(True)
    plt.show()

    # accuracies over epochs
    plt.figure()
    plt.plot(epochs, best_train_accs, marker='o', label='train acc')
    plt.plot(epochs, best_val_accs, marker='o', label='validation acc')
    plt.plot(epochs, best_test_accs, marker='o', label='test acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Multiclass Logistic Regression Accuracy vs Epoch (lr={best_lr})')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()