'''
Simulation of irreversible and reversible magnetisation using the Jiles-Atherton model.
A series of reversal curves are then created for First-Order Reversal Curves (FORC) analysis.

Functions
    hysteresis_loop: generate data and plot of a single hysteresis loop
    generate_forc_data: generate data and plot of a series of forc reversal loops
'''

#Imports
import numpy as np
from func_magnetics import effective_field, langevin, generate_field_path

def hysteresis_loop(
        m_sat: float = 4000.0,
        a: float = 980.0,
        alpha: float = 0.72,
        c: float = 0.2,
        k: float = 600.0,          

        h_max: float = 5000.0,
        h_rev: float = None,
        num_data_points: int = 650   
) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Simulates a hysteresis loop using the Jiles-Atherton model.

    Parameters
        m_sat: saturation magnetisation (A/m)
        a: shape parameter (A/m)
        alpha: inter-domain coupling parameter
        c: ratio of reversible magnetisation
        k: coercivity factor (A/m)

        h_max: maximum applied magnetic field strength (A/m)
        h_rev: reversal point (A/M)
        num_data_points: number of data points in the applied field        
        
    Returns
        A tuple of the applied magnetic field array and the magnetisation array.
    """
    # Reversal point control
    if h_rev is None:
        h_rev = - h_max

    # Generate the field path and magnetic arrays
    h_data = generate_field_path(h_max, num_data_points, h_rev)
    m_data = np.zeros(len(h_data))
    m_irr = np.zeros(len(h_data))

    # Calculate dH and no change escape
    for i in range(1, len(h_data)):
        dH = h_data[i] - h_data[i-1]
        if dH == 0:
            m_data[i] = m_data[i-1]
            m_irr[i] = m_irr[i-1]
            continue

        # Tracking the magnetic field direction
        delta_h = 1.0 if dH > 0 else -1.0

        # Calculate the effective field and anhysteric magnetisation
        q = effective_field(h_data[i], m_data[i-1], a, alpha)
        m_an = m_sat * langevin(q)

        # Calculating irreversible derivative with control
        denominator = k * delta_h - alpha * (m_an - m_irr[i-1])
        dMirr_dH = ((m_an - m_irr[i-1])) / (denominator)

        # Using irreversible and reversible to get magnetistation
        m_irr[i] = m_irr[i-1] + (dMirr_dH * dH)
        m_rev = c * (m_an - m_irr[i])
        m_data[i] = m_rev + m_irr[i]

    # Removing h_init from the data
    data_cut = int(num_data_points * 0.2)
    return h_data[data_cut:], m_data[data_cut:]


def gen_reversal_data(
        m_sat: float = 4000.0,
        a: float = 980.0,
        alpha: float = 0.72,
        c: float = 0.2,
        k: float = 600.0,

        h_max: float = 5000.0,
        h_upper: float = None,
        h_lower: float = None,

        num_loops: int = 101,
        num_data_points: int = 650,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    '''
    Generates a collection of reversal curves for FORC analysis.

    Parameters
        m_sat: saturation magnetisation (A/m)
        a: shape parameter (A/m)
        alpha: inter-domain coupling parameter
        c: ratio of reversible magnetisation
        k: coercivity factor (A/m)

        h_max: maximum applied magnetic field strength (A/m)
        h_upper, h_lower: The limits of the reversal fields
        
        num_loops: number of reversal curves generated
        num_data_points: number of data points in the applied field       

    Returns
        Three lists containing 1D arrays for the applied field, magnetisation and 
        reversal fields representing the reversal loops.
    '''
    # Reversal limits control
    if h_upper is None: h_upper = h_max
    if h_lower is None: h_lower = -h_max

    # Create data and reversal field array
    h_data, m_data, hr_data = [], [], []
    h_rev_field = np.linspace(h_upper, h_lower, num_loops)

    # Generate the loops and append them 
    for h_rev in h_rev_field:
        h_temp, m_temp = hysteresis_loop(m_sat, a, alpha, c, k, h_max, h_rev, num_data_points)

        # Append the data into the array
        h_data.append(h_temp)
        m_data.append(m_temp)
        hr_data.append(np.full_like(h_temp, h_rev))

    print('Data generation successful!')
    print('Loops created: {} from limits {} to {}'.format(len(h_data), h_lower, h_upper))

    return h_data, m_data, hr_data
