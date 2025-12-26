import numpy as np
import torch
from torch import nn


class Ridge_Regression:

    def __init__(self, lambd):
        self.w = None
        self.lambd = lambd

    def fit(self, X, Y):
        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """
        Y = 2 * (Y - 0.5)  # transform the labels to -1 and 1, instead of 0 and 1.

        n_samples, n_features = X.shape

        # Identity matrix of size d×d
        I = np.eye(n_features)

        # A = (X^T X) / n + λ I
        A = (X.T @ X) / n_samples + self.lambd * I

        # b = (X^T Y) / n
        b = (X.T @ Y) / n_samples

        # Closed-form solution for ridge weights
        self.w = np.linalg.inv(A) @ b

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. 
        :return: The predicted output. 
        """
        preds = None

        # linear scores: s = X w
        scores = X @ self.w  # shape: (n_samples,)

        # convert scores to labels in {-1, 1}
        preds = np.where(scores >= 0, 1, -1)

        preds = (preds + 1) / 2

        return preds

class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        out = self.linear(x)
        return out

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x
