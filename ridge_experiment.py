# ridge_experiment.py
import numpy as np
import matplotlib.pyplot as plt
import torch

from helpers import read_data_demo, plot_decision_boundaries
from models import Ridge_Regression

np.random.seed(42)
torch.manual_seed(42)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def load_split_data():
    # train
    train_data, _ = read_data_demo('train.csv')
    X_train = train_data[:, :2]          # long, lat
    y_train = train_data[:, 2].astype(int)

    # validation
    val_data, _ = read_data_demo('validation.csv')
    X_val = val_data[:, :2]
    y_val = val_data[:, 2].astype(int)

    # test
    test_data, _ = read_data_demo('test.csv')
    X_test = test_data[:, :2]
    y_test = test_data[:, 2].astype(int)

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()

    lambdas = [0., 2., 4., 6., 8., 10.]

    train_accs = []
    val_accs = []
    test_accs = []

    best_val_acc = -1.0
    best_lambda = None
    best_model = None
    best_test_acc = None

    for lam in lambdas:
        model = Ridge_Regression(lam)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy(y_train, y_train_pred)
        val_acc = accuracy(y_val, y_val_pred)
        test_acc = accuracy(y_test, y_test_pred)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lambda = lam
            best_model = model
            best_test_acc = test_acc

        print(f"λ={lam}: train={train_acc:.3f}, val={val_acc:.3f}, test={test_acc:.3f}")

    print(f"\nBest λ (by validation): {best_lambda}")
    print(f"Validation accuracy: {best_val_acc:.3f}")
    print(f"Test accuracy of best model: {best_test_acc:.3f}")

    # 3.2.1 – plot train/val/test accuracy vs λ
    plt.figure()
    plt.plot(lambdas, train_accs, marker='o', label='train')
    plt.plot(lambdas, val_accs, marker='o', label='validation')
    plt.plot(lambdas, test_accs, marker='o', label='test')
    plt.xlabel('λ')
    plt.ylabel('accuracy')
    plt.title('Ridge Regression: accuracy vs λ')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3.2.2 – decision boundaries: best & worst λ using test points

    # worst λ לפי validation
    worst_idx = int(np.argmin(val_accs))
    worst_lambda = lambdas[worst_idx]
    worst_model = Ridge_Regression(worst_lambda)
    worst_model.fit(X_train, y_train)

    plot_decision_boundaries(best_model, X_test, y_test,
                             title=f"Best λ = {best_lambda}")
    plot_decision_boundaries(worst_model, X_test, y_test,
                             title=f"Worst λ = {worst_lambda}")


if __name__ == "__main__":
    main()