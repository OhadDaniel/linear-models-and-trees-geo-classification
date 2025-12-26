import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

from helpers import plot_decision_boundaries
from models import Logistic_Regression

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)


def load_split_data_multiclass():
    """
    Load train/validation/test data for the multiclass case.
    Assumes files: train_multiclass.csv, validation_multiclass.csv, test_multiclass.csv
    and columns: long, lat, country.
    """
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


def evaluate_torch_model(model, X, y, criterion, batch_size=32):
    """
    Evaluate a torch model on given data: returns (avg_loss, avg_accuracy).
    """
    model.eval()
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    dataset = TensorDataset(X_t, y_t)
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


def train_logreg_multiclass_ridge(
    lambd,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_classes,
    lr_init=0.01,
    num_epochs=30,
    batch_size=32,
):
    """
    Multiclass logistic regression with ridge regularization via weight_decay.
    λ controls the L2 penalty strength.
    """
    input_dim = 2
    output_dim = num_classes

    model = Logistic_Regression(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()

    # weight_decay is L2 regularization (ridge)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_init, weight_decay=lambd)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.3)

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_accs = []
    test_accs = []

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

        val_loss, val_acc = evaluate_torch_model(model, X_val, y_val, criterion, batch_size)
        test_loss, test_acc = evaluate_torch_model(model, X_test, y_test, criterion, batch_size)

        val_accs.append(val_acc)
        test_accs.append(test_acc)

        print(
            f"[ridge λ={lambd}] Epoch {epoch+1}/{num_epochs} "
            f"train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.3f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.3f}, "
            f"test_acc={test_acc:.3f}, lr={scheduler.get_last_lr()[0]:.5f}"
        )

        scheduler.step()

    # use last epoch validation accuracy as score for this λ
    final_val_acc = val_accs[-1]
    final_test_acc = test_accs[-1]
    return model, final_val_acc, final_test_acc


def main():
    # === LOAD DATA ===
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes = load_split_data_multiclass()

    # ----------------------------------------------------------------------
    # Q3: Decision Tree with max_depth = 2
    # ----------------------------------------------------------------------
    tree_depth2 = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_depth2.fit(X_train, y_train)

    y_train_pred_2 = tree_depth2.predict(X_train)
    y_val_pred_2 = tree_depth2.predict(X_val)
    y_test_pred_2 = tree_depth2.predict(X_test)

    train_acc_2 = accuracy_score(y_train, y_train_pred_2)
    val_acc_2 = accuracy_score(y_val, y_val_pred_2)
    test_acc_2 = accuracy_score(y_test, y_test_pred_2)

    print("\n=== Decision Tree (max_depth=2) ===")
    print(f"Train accuracy: {train_acc_2:.3f}")
    print(f"Validation accuracy: {val_acc_2:.3f}")
    print(f"Test accuracy: {test_acc_2:.3f}")

    # visualize predictions on test set
    plot_decision_boundaries(
        tree_depth2, X_test, y_test, title="Decision Tree (max_depth=2) – Test Predictions"
    )

    # ----------------------------------------------------------------------
    # Q4: Decision Tree with max_depth = 10
    # ----------------------------------------------------------------------
    tree_depth10 = DecisionTreeClassifier(max_depth=10, random_state=42)
    tree_depth10.fit(X_train, y_train)

    y_train_pred_10 = tree_depth10.predict(X_train)
    y_val_pred_10 = tree_depth10.predict(X_val)
    y_test_pred_10 = tree_depth10.predict(X_test)

    train_acc_10 = accuracy_score(y_train, y_train_pred_10)
    val_acc_10 = accuracy_score(y_val, y_val_pred_10)
    test_acc_10 = accuracy_score(y_test, y_test_pred_10)

    print("\n=== Decision Tree (max_depth=10) ===")
    print(f"Train accuracy: {train_acc_10:.3f}")
    print(f"Validation accuracy: {val_acc_10:.3f}")
    print(f"Test accuracy: {test_acc_10:.3f}")

    # visualize predictions on test set
    plot_decision_boundaries(
        tree_depth10, X_test, y_test, title="Decision Tree (max_depth=10) – Test Predictions"
    )

    # ----------------------------------------------------------------------
    # Q5: Multiclass Logistic Regression + Ridge Regularization (bonus)
    # ----------------------------------------------------------------------
    lambda_values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    best_val_acc = -1.0
    best_lambda = None
    best_model = None
    best_test_acc = None

    for lambd in lambda_values:
        print(f"\n=== Training Logistic Regression with ridge λ={lambd} ===")
        model_ridge, val_acc, test_acc = train_logreg_multiclass_ridge(
            lambd,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            num_classes,
            lr_init=0.01,
            num_epochs=30,
            batch_size=32,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lambda = lambd
            best_model = model_ridge
            best_test_acc = test_acc

    print("\n=== Best Ridge Logistic Regression Model ===")
    print(f"Best λ: {best_lambda}, validation accuracy: {best_val_acc:.3f}")
    print(f"Test accuracy of best ridge model: {best_test_acc:.3f}")

    # visualize predictions of the best ridge logistic model on the test set
    plot_decision_boundaries(
        best_model,
        X_test,
        y_test,
        title=f"Ridge Logistic Regression (best λ={best_lambda}) – Test Predictions",
    )


if __name__ == "__main__":
    main()