import numpy as np
import matplotlib.pyplot as plt
import torch

np.random.seed(42)
torch.manual_seed(42)
# Our function: f(x, y) = (x - 3)^2 + (y - 5)^2
def f(x):
    """x is a vector [x, y]."""
    return (x[0] - 3) ** 2 + (x[1] - 5) ** 2


def grad_f(x):
    """Gradient of f at x = [x, y]."""
    return np.array([2 * (x[0] - 3), 2 * (x[1] - 5)])


def gradient_descent(lr=0.1, num_iters=1000):
    """
    Run gradient descent starting from (0,0)
    :param lr: learning rate
    :param num_iters: number of iterations
    :return: trajectory (array of shape (num_iters+1, 2)), final point
    """
    x = np.array([0.0, 0.0])  # initial point (x, y) = (0, 0)
    trajectory = [x.copy()]

    for t in range(num_iters):
        g = grad_f(x)
        x = x - lr * g
        trajectory.append(x.copy())

    return np.array(trajectory), x


def main():
    lr = 0.1
    num_iters = 1000

    traj, x_final = gradient_descent(lr=lr, num_iters=num_iters)

    print(f"Final point after {num_iters} iterations: {x_final}")
    print(f"f(x_final) = {f(x_final)}")

    # Plot trajectory: x-axis = x, y-axis = y
    xs = traj[:, 0]
    ys = traj[:, 1]
    iters = np.arange(len(traj))

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(xs, ys, c=iters, cmap='viridis', s=20)
    plt.colorbar(scatter, label='Iteration')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Trajectory on f(x, y) = (x-3)^2 + (y-5)^2')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()