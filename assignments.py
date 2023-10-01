from utils import to_numpy
from engine import Tensor
import numpy as np


# +++++++++++++++++ Assignment +++++++++++++++++
# In this file your task is to complete the functions
# below. You should only change the code in the
# designated areas.
#
# This task is to use the custom tensor class that
# you implemented earlier to implement a simple
# backpropagation algorithm.
# +++++++++++++++++++++++++++++++++++++++++++++++


def dkt(omega: list, rho: list, base: np.ndarray) -> np.ndarray:
    """Direct Kinematic Task non-vectorized

    Args:
        omega: A list of shape (n) containing joint angles, where n is number of joints
        rho: A list of shape (n) containing link lengths, where n is number of joints
        base: A np.ndarray of shape (2) representing base angle and length with respect to origin

    Returns:
        A np.ndarray X of shape (2, n) containing joint positions, where n is number of joints
    """
    n = len(omega)

    omega_base = Tensor(base[0])
    rho_base = Tensor(base[1])

    # Determine base position
    phi = omega_base
    pos_x = rho_base * phi.cos()
    pos_y = rho_base * phi.sin()

    # Add base position to list of joint positions
    x = [np.stack((pos_x, pos_y))]

    for k in range(n):
        # Calculate joint position
        # TODO: Implement this loop
        # Hint: You can use the previous joint position to calculate the current joint position
        # -------------------------------------------------
        # START OF YOUR CODE
        
        
        
        # -------------------------------------------------

        # Add joint position to list of joint positions
        x.append(np.stack((pos_x, pos_y)))

    return np.stack(x, axis=1)

def dkt_vectorized(omega: list, rho: list, base: np.ndarray) -> np.ndarray:
    """Direct Kinematic Task vectorized

    Args:
        omega: A list of shape (n) containing joint angles, where n is number of joints
        rho: A list of shape (n) containing link lengths, where n is number of joints
        base: A np.ndarray of shape (2) representing base angle and length with respect to origin

    Returns:
        A np.ndarray X of shape (2, n) containing joint positions, where n is number of joints
    """

    # Determine number of joints from input
    n = len(omega) + 1

    # Add base to length and angle vectors
    omega = Tensor(np.concatenate((base[0].flatten(), to_numpy(omega).flatten())).flatten())
    rho = Tensor(np.concatenate((base[1].flatten(), to_numpy(rho).flatten())))

    # TODO: Implement vectorized version of DKT
    # -------------------------------------------------
    # START OF YOUR CODE
    
    
    
    
    
    
    # -------------------------------------------------
    return X

def manipulator_loss(X: np.ndarray, x_goal: np.ndarray) -> Tensor:
    """Manipulator loss

    Args:
        X: A np.ndarray of shape (2, n) containing joint positions, where n is number of joints
        x_goal: A np.ndarray of shape (2,) representing goal position

    Returns:
        loss: A Tensor object representing the loss

    Hint: You can add another arguments to this function if you want to, but it is not necessary
    """

    # TODO: Implement loss function
    # -------------------------------------------------
    # START OF YOUR CODE
    
    # -------------------------------------------------
    return loss
