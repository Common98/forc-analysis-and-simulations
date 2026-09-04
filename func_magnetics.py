'''
Utility functions for the effective magnetic field, Langevin function, and 
magnetic field generation needed to calculate the Jiles-Atherton 
hysteresis model.

Functions
    effective_field: Calculates the effective field (Q)
    langevin: Calculates the Langevin function
    generate_field_path: Generates the applied field values from the upper and lower limits
'''

import numpy as np

def effective_field(h: float, m: float, a: float, alpha: float) -> float:
    '''
    Calculates the effective magnetic field Q experienced by the magnetic domain.
    
    Parameters
        h: applied magnetic field (A/m)
        m: magnetisation (A/m)
        a: domain shape parameter (A/m)
        alpha: inter-domain coupling parameter

    Returns
        A float of the calculated effective magnetic field.
    '''
    return (h + alpha * m) / a


def langevin(q: float) -> float:
    '''
    Computes the Langevin function (L(q)) for magnetic domain magnetisation.

    Parameters
        q: effective magnetic field (A/m)
    
    Returns
        A float of the calculated Langevin function. 
    '''
    if abs(q) > 1e-4:
        return (1.0 / np.tanh(q)) - (1.0 / q)
    else:
        return q / 3


def generate_field_path(h_max: float, num_data_points: int, h_rev: float = None) -> np.ndarray:
    '''
    Generates the field path, from zero to saturation, to the reversal point and back to saturation.

    Parameters
        h_max: maximum applied magnetic field (A/m)
        h_rev: reversal point (A/m)
        num_data_points: number of data points in the loop
    
    Returns
        An array of the path of the applied magnetic field.
    '''
    # Reversal point trap if none
    if h_rev is None:
        h_rev = -h_max

    # Creating field arrays and stacking
    h_init = np.linspace(0, h_max, int(num_data_points * 0.2))
    h_down = np.linspace(h_max, h_rev, int(num_data_points * 0.4))
    h_up = np.linspace(h_rev, h_max, int(num_data_points * 0.4))

    return np.hstack((h_init, h_down, h_up))


def coercivity(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    '''
    Determine the coercivity or the remanence of a hysteresis loop by finding the
    crossings at zero and then the difference between the two.

    Parameters
        x_data: searched range for the crossings at zero
        y_data: range for the difference; H for the coercivity, M for remanence

    Returns
        Float of the coercivity or the remanence, nan if not avaiable.
    '''
    # Find the crossings at 0
    zero_crossings = np.where(np.diff(np.sign(x_data)))[0]
    
    # Find the difference between the crossings
    if len(zero_crossings) >= 2:
        return y_data[max(zero_crossings)] - y_data[min(zero_crossings)]
    else:
        return np.nan