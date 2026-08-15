'''
Simulation of ferromagnetic hysteresis and the creation of a series of ferromagnetic
reversal curves needed for First-Order Reversal Curves (FORC) analysis.

Functions:
    ferro_hysteresis: Data and plot of a single ferromagnetic hysteresis
    ferro_reversal: Data and plot of a series of ferromagnetic reversals
'''

import numpy as np
import matplotlib.pyplot as plt
from sim_mag_functions import langevin, effective_field

def ferro_hysteresis(
    Ms: float = 1.48e6,
    a: float = 10.0,
    alpha: float = 9.38e-4,
    H_max: float = 4000.0,   
    num_data_points: int = 250,
    plot: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Generates a simulation of a ferromagnetic hysteresis loop.

    Parameters:
        Ms: Saturation magnetisation (A/m)
        a: Shape parameter (A/m)
        alpha: Inter-domain coupling parameter
        H_max: Maximum applied field strength (Oe)
        num_data_points: Number of data points in the hysteresis
        plot: Whether to render the resulting hysteresis loops

    Returns:
        A tuple of the applied magnetic field array and the magnetisation array.
    """

    # Define the magnetic field path
    H_zero = np.linspace(0, H_max, num_data_points // 2 + 1)
    H_down = np.linspace(H_max, -H_max, num_data_points + 1)
    H_up = np.linspace(-H_max, H_max, num_data_points + 1)
    H = np.hstack((H_zero, H_down, H_up))

    # Calculating magnetisation loop
    M_an = np.zeros(len(H))
    for i in range(len(H)):
        M_prev = 0.0 if i == 0 else M_an[i-1]
        Q = effective_field(H[i], M_prev, a, alpha)
        M_an[i] = Ms * langevin(Q)

    # Slice off the initial path from 0 to H_max
    H_loop = H[len(H_zero):]
    M_loop = M_an[len(H_zero):]

    # Plot of the hysteresis loop
    if plot:
        plt.figure(figsize=(9,6))
        plt.plot(H_loop, M_loop, linewidth=1.2)
        plt.title("Simulated Ferromagnetic Hysteresis", fontsize=12)
        plt.xlabel("Applied Field $H$ (A/m)", fontsize=10)
        plt.ylabel("Magnetization $M$ (A/m)", fontsize=10)
        plt.tight_layout()
        plt.show()

    return H_loop, M_loop


def ferro_reversal(
    Ms: float = 1.48e6,
    a: float = 10.0,
    alpha: float = 9.38e-4,
    H_max: float = 4000.0,
    H_upper: float = None,
    H_lower: float = None,
    num_loops: int = 101,
    num_data_points: int = 250,
    end_at_max: bool = False,
    full_loop: bool = True,
    plot: bool = True
) -> list[tuple[np.ndarray, np.ndarray]]:
    
    """
    Generates a series of ferromagnetic reversal loops.
    
    Parameters:
        Ms: Saturation magnetisation (A/m)
        a: Shape parameter (A/m)
        alpha: Inter-domain coupling parameter
        H_max: Maximum applied field strength (Oe)
        H_upper, H_lower: The limits of the reversal fields
        num_loops: Number of reversal field curves to generate
        num_data_points: Number of data points in the magnetic field
        end_at_max: Whether the last reversal point should be at magnetic saturation
        full_loop: Whether the limits should be set to max
        plot: Whether to render the resulting hysteresis loops

    Returns:
        A list of tuples containing (H_field_array, magnetisation_array) representing the 
        reversal loops.

    """
    # Data array and plot setting
    forc_data = []
    
    if plot: 
        plt.figure(figsize=(9,6))

    # Setting the limits of the reversal curves
    if full_loop:
        H_upper, H_lower = H_max, -H_max
    if H_upper is None:
        H_upper = H_max
    if H_lower is None:
        H_lower = -H_max

     # Array of the reversal points
    H_reversal = np.linspace(H_upper, H_lower, num_loops)
    
    # Appending negative max(H) to the reversal array
    if end_at_max: 
        H_reversal = np.append(H_reversal, -H_max)

    for Hr in H_reversal:
        # Define the magnetic field path
        H_zero = np.linspace(0, H_max, num_data_points // 2 + 1)
        H_down = np.linspace(H_max, Hr, num_data_points + 1)
        H_up = np.linspace(Hr, H_max, num_data_points + 1)
        H = np.hstack((H_zero, H_down, H_up))

        # Calculating magnetisation loop
        M_an = np.zeros(len(H))
        for i in range(len(H)):
            M_prev = 0.0 if i == 0 else M_an[i-1]
            Q = effective_field(H[i], M_prev, a, alpha)
            M_an[i] = Ms * langevin(Q)

        # Slice off the initial path from 0 to H_max
        H_loop = H[len(H_zero):]
        M_loop = M_an[len(H_zero):]

        forc_data.append((H_loop, M_loop))

        # Plotting the reversal loops on the same plot        
        if plot:
            plt.plot(H_loop, M_loop, linewidth=0.8, alpha=0.7)

    if plot:
        plt.title("Simulated Ferromagnetic Reversal Curves", fontsize=12)
        plt.xlabel("Applied Field $H$ (A/m)", fontsize=10)
        plt.ylabel("Magnetization $M$ (A/m)", fontsize=10)
        plt.tight_layout()
        plt.show()

    return forc_data