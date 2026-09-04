'''
Utility functions needed to process raw data of the magetisation into
data that can be analysed as a FORC diagram.

Functions
    normalise: normalises the magnetisation and applies an offset to the applied field
    splitter: splits the hysteresis loop into the reversal from -H_max to H_max
    interpolate: interpolates the magnetisation over a h/hr meshgrid and converts the blank space to nans
'''
import numpy as np
import scipy.interpolate as si

def normalise(
        h_data: list[np.ndarray],
        m_data: list[np.ndarray],
        hr_data: list[np.ndarray],

        h_cali: list[np.ndarray] = None,
        m_cali: list[np.ndarray] = None,

        m_flip: int = 1,
        h_offset: float = 0.0
        ):
    '''
    Normalises magnetisation curve and applies field offset corrections.
    Supports simple saturation nomralisation and calibration-loop drift correction.

    Parameters
        h_data: list of 1D arrays of the applied field data
        m_data: list of 1D arrays of the magnetisation data
        hr_data: list of 1D arays of the reversal points
        h_cali: optional dataset for the calibration applied field
        m_cali: optional dataset for the calibration magnetisation
        m_flip: sign multiplier (+1 or -1) to invert magnetisation orientation
        h_offset: applied field axis shift applied to align to zero field 
    
    Returns
        Tuple of (h_norm, m_norm, hr_norm) containing normalised lists of 1D arrays.
    '''
    h_norm, m_norm, hr_norm = [], [], []

    # Normalise without calibration section
    if h_cali is None and m_cali is None:
        for i in range(len(m_data)):
            # Normalise calibraton data
            m_normalise = max(np.abs(m_data[i]))

            h_norm.append(h_data[i] + h_offset)
            m_norm.append((m_data[i] / m_normalise) * m_flip)

    # Normalise with calibration section
    else:
        for i in range(len(m_cali)):
            # Centre calibration factor
            m_centre = (max(m_cali[i]) + min(m_cali[i])) / 2
            # Normalise calibraton data
            m_normalise = (max(m_cali[i]) - min(m_cali[i])) / 2
            # Matching reversal and calibration starting point
            m_start = m_cali[i][-1] - m_data[i][-1]

            h_norm.append(h_data[i] + h_offset)
            m_norm.append(((m_data[i] + m_start - m_centre) / m_normalise) * m_flip)

    hr_norm = [hr + h_offset for hr in hr_data]

    return h_norm, m_norm, hr_norm


def splitter(
        h_data: list[np.ndarray], 
        m_data: list[np.ndarray], 
        hr_data: list[np.ndarray],
        min_diff: float = 1.0e-3
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    '''
    Filter out flat/zero-length curve and extracts only the ascending reversal branch;
    H_min to H_max for each loop.

    Parameters
        h_data: list of 1D arrays of the applied field data
        m_data: list of 1D arrays containing magnetization data
        hr_data: list of 1D arays of the reversal points
        min_field_span: minimum difference needed to keep a loop

    Returns
        Tuple of h_split, m_split, hr_split, sliced lists of 1D arrays.
    '''
    # Creating empty arrays
    h_split, m_split, hr_split = [], [], []

    # Splitting section
    for h, m, hr in zip(h_data, m_data, hr_data):
        # Range guard: skip flat arrays
        if np.ptp(h) < min_diff:
            continue

        # Setting the index and trap
        i_min = np.argmin(h)
        i_max = i_min + np.argmax(h[i_min:])

        h_temp = h[i_min:i_max]
        m_temp = m[i_min:i_max]
        hr_temp = hr[i_min:i_max]

        h_split.append(h_temp)
        m_split.append(m_temp)
        hr_split.append(hr_temp)

    return h_split, m_split, hr_split


def interpolate(
        h_data: list[np.ndarray],
        m_data: list[np.ndarray],
        hr_data: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    '''
    Interpolates the raw data by creating a meshgrid of h and hr and then
    maps the magnetisation to the grid points.

    Parameters
        h_data: list of 1D arrays of the applied field data
        m_data: list of 1D arrays containing magnetization data
        hr_data: list of 1D arays of the reversal points

    Returns
        Tuple of h_int, m_int, hr_int, interpolated lists of 1D arrays.
    '''
    #Interpolation of the data
    # Average step in the applied field measurement
    step = np.mean(np.concatenate([np.diff(h) for h in h_data]))

    # Concatenate the data arrays
    h_vals = np.concatenate(h_data)
    m_vals = np.concatenate(m_data)
    hr_vals = np.concatenate(hr_data)
     
    # Finding the max and min of h and hr
    h_max, h_min = np.max(h_vals), np.min(h_vals)
    hr_max, hr_min = np.max(hr_vals), np.min(hr_vals)

    # Creating a grid from the h and hr values
    h, hr = np.meshgrid(
        np.linspace(h_min, h_max, int((h_max - h_min) // step + 1)),
        np.linspace(hr_min, hr_max, int((hr_max - hr_min) // step + 1))
    )

    # Creating a list of the h and hr values
    hhr_vals = np.concatenate(
        (np.reshape(h_vals, (-1, 1)), np.reshape(hr_vals, (-1, 1))),
        axis=1
    )

    # Overlaying the m values over the grid and making empty points nan
    m = si.griddata(hhr_vals, m_vals, (h,hr), method='cubic')
    m[h<hr] = np.nan

    return h, m, hr


