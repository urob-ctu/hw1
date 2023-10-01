import numpy as np
import matplotlib.pyplot as plt
from engine import Tensor


# +++++++++++++++++ Important notice +++++++++++++++++
# You should not change the code in this file.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++


def visualize_manipulator(x, x_goal):
    plt.clf()
    plt.plot(x[0, :], x[1, :], 'o-', color='b', linewidth=5, mew=5)
    plt.text(x[0, 0], x[1, 0] - 0.2, 'x_base', color='k')
    plt.plot([x[0, 0] - 0.1, x[0, 0] + 0.1], [x[1, 0], x[1, 0]], '-', color='k', linewidth=15, mew=2)
    plt.plot(x_goal[0], x_goal[1], 'x', color='r', linewidth=5, mew=5)
    plt.text(x_goal[0], x_goal[1], 'x_goal', color='r')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-5, 5)
    plt.ylim(-1, 9)
    # plt.axis('equal')
    plt.grid()
    plt.pause(0.001)

def generate_task(number_of_joints: int):
    # Generate task without base
    omega_ret = []
    rho_ret = []
    omega = (np.random.randn(number_of_joints) - 0.5) / 1
    for i, num in enumerate(omega):
        omega_ret.append(Tensor(num, label=f'omega{i}', req_grad=True))
    rho = np.ones(number_of_joints)
    for i, num in enumerate(rho):
        rho_ret.append(Tensor(num, label=f'rho{i}', req_grad=True))

    base = np.zeros(2)

    # Generate random goal
    x_goal = np.array((0, 3) + 6 * (np.random.rand(2)) - 0.5)
    return omega_ret, rho_ret, base, x_goal

def to_numpy(X: list) -> list:
    X_ret = []
    if isinstance(X, Tensor):
        return X.data.flatten()
    for x in X:
        if isinstance(x, Tensor):
            X_ret.append(x.data.flatten())
        else:
            X_ret.append(to_numpy(x))
        
    return np.array(X_ret)
