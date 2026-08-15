'''
Utility functions for the effective magnetic field, Langevin function, and 
hysteresis directional tracking needed to calculate the Jiles-Atherton 
hysteresis model.

Functions
    effective_field: Calculates the effective field (Q)
    langevin: Calculates the Langevin function
    delta_langevin: Calculates the derivative of the Langevin function
    delta_x: Creates an array to track the direction of the applied magnetic field
'''

import numpy as np
from typing import Union

def effective_field(H: float, M: float, a: float, alpha: float) -> float:
    '''
    Calculates the effective magnetic field Q experienced by the magnetic domain.
    
    Parameters:
        H: Applied magnetic field (Oe)
        M: Magnetisation (A/m)
        a: Domain shape parameter (A/m)
        alpha: Inter-domain coupling parameter

    Returns:
        A float of the calculated effective magnetic field.
    '''
    return (H + alpha * M) / a


def langevin(Q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Computes the Langevin function (L(Q)) for magnetic domain magnetisation.

    Parameters:
        Q: Effective magnetic field (Oe)
    
    Returns:
        A float or array of the calculated Langevin function. 
    '''
    # Converts all types into an array of floats
    Q = np.asarray(effective_field, dtype = float)

    # If dimension = 0, then calculate and return the float value
    if Q.ndim == 0:
        if abs(Q) > 1e-4:
            return float((1.0 / np.tanh(Q)) - (1.0 / Q))
        return float(Q / 3.0)

    # Creates an array of zeros then calculates the value through a mask
    result = np.zeros_like(Q)
    small_mask = np.abs(Q) <= 1e-4
    large_mask = ~small_mask

    result[small_mask] = Q[small_mask] / 3.0
    result[large_mask] = (1.0 / np.tanh(Q[large_mask])) - (1.0 / Q[large_mask])

    return result


def delta_langevin(Q: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    '''
    Computes the derivative of the Langevin function (dL/dQ).

    Parameters:
        Q: Effective magnetic field (Oe)

    Returns:
        A float or array of the calculated derivative of the Langevin function.
    '''
    # Convert all types into an array of floats
    Q = np.asarray(effective_field, dtype = float)

    # If dimensions = 0 then calculate and return a float
    if Q.ndim == 0:
        if abs(Q) > 1e-4:
            return float((1.0 / Q**2) - (1.0 / np.tanh(Q))**2 + 1)
        return float(1.0 / 3.0)

    # Creates an array of the small mask and applies the large to indicated values
    result = np.full_like(Q, 1.0 / 3.0)
    large_mask = np.abs(Q) > 1e-4
    result[large_mask] = (1.0 / Q[large_mask]**2) - (1.0 / np.tanh(Q[large_mask]))**2 + 1.0

    return result


def delta_x(H: np.ndarray) -> np.ndarray:
    '''
    Tracking whether the applied magnetic field is increasing or decreasing in value.

    Parameters:
        H: Applied magnetic field (Oe)

    Returns:
        An array of +/- 1 to indicate the increase or decrease of the applied magnetic field.
    '''
    # Array of the differences in the array items
    difference = np.diff(H)

    # If statements to change values to +/- 1
    directions = np.where(difference >= 0, 1, -1)

    # Add an extra value to the array to keep length the same
    return np.append(directions, directions[-1])

