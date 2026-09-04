'''
Mathematical function to calculate the FORC density from the raw data.

Functions
    compute_forc: calculates the FORC density
'''
import numpy as np
import scipy.ndimage as sn

def compute_forc(
        h_data: np.ndarray,
        m_data: np.ndarray,
        hr_data: np.ndarray,

        sf: int = 5,
        normalise: bool = False
) -> np.ndarray:
    '''
    Calculates the forc density using a smoothing factor to reduce noise 
    and convolving a kernel over the magnetisation to determine the derivative.

    Parameters
        h_data: list of 1D arrays of the applied field data
        m_data: list of 1D arrays containing magnetization data
        hr_data: list of 1D arays of the reversal points
    
        sf: smoothing factor for the kernel size
        normalise: true to normalise the forc density between 0 and 1

    Returns
        Tuple of the forc_data, hc_axis, hu_axis in lists of 1D arrays.
    '''
    # Intervals of h and hr arrays
    step_h = np.mean(np.concatenate([np.diff(h) for h in h_data]))
    step_hr = np.max(np.diff(np.concatenate(hr_data)))

    # Creatung a small grid from the steps
    xx, yy = np.meshgrid(
        np.linspace(sf * step_h, -sf * step_h, 2 * sf + 1),
        np.linspace(sf * step_hr, -sf * step_hr, 2 * sf + 1)
    )

    # Reshaping the grid into one column
    xx = np.reshape(xx, (-1, 1))
    yy = np.reshape(yy, (-1, 1))

    # Creating a grid of weights using the pseudo-inverse
    coefficients = np.linalg.pinv(np.hstack((np.ones_like(xx), xx, xx ** 2, yy, yy ** 2, xx * yy)))
    kernel = np.reshape(coefficients[5], (2 * sf + 1, 2 * sf + 1))

    # Convolution calculation
    forc_data = -0.5 * sn.convolve(m_data, kernel, mode='constant', cval=np.nan)

    max_forc, min_forc = np.nanmax(forc_data), np.nanmin(forc_data)
    print('Raw FORC: minimum = {} and maximum = {}'.format(min_forc, max_forc))

    # Normalisation section
    if normalise:
        forc_temp = []
        for i in range(len(forc_data)):
            array = []
            for rho in forc_data[i]:
                normal = (rho - min_forc) / (max_forc - min_forc)
                array.append(normal)
            forc_temp.append(array)
        forc_data = forc_temp

        max_forc, min_forc = np.nanmax(forc_data), np.nanmin(forc_data)
        print('Normalised FORC: minimum = {} and maximum = {}'.format(min_forc, max_forc))

    # Calculating the axis
    hc_axis, hu_axis = [], []
    for i in range(len(h_data)):
        hc_temp = (h_data[i] - hr_data[i]) / 2
        hu_temp = (h_data[i] + hr_data[i]) / 2
        hc_axis.append(hc_temp) 
        hu_axis.append(hu_temp)
        
    return forc_data, hc_axis, hu_axis