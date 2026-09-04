'''
Utility functions for plotting either the reversal curves or the data
in field space or a FORC diagram in a contour plot.

Functions
    plot_curves: plots a single loop, or all the reversal curves in a line graph
    plot_contour: plots the data in a contour plot to represent field space or FORC diagram
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_curves(
        x_data: list[np.ndarray],
        y_data: list[np.ndarray],
        z_data: list[np.ndarray] = None,

        colour: str = 'viridis',
        subplot: bool = False,
        save: bool = False,

        path: str = 'reversal_plot.png',
        title: str = 'Simulated Hysteresis Curves',
        x_label: str = 'Applied Field $H$ (A/m)',
        y_label: str = 'Magnetisation $M$ (A/m)',
        z_label: str = 'Reversal Field $Hr$ (A/m)'
) -> None:
    '''
    Plots a  single hysteresis loop or a collection of magnetic reversal curves overlayed 
    in a single plot or in separate subplots.

    Parameters
        x_data: list of 1D arrays of the applied field data
        y_data: list of 1D arrays of the magnetisation data
        z_data: list of 1D arrays of the reversal points

        colour: colour choice for the plot
        subplot: true to separate the curves, false to overlay them
        save: true to save the plot, false to not save the plot

        path: filepath destination 
        title: title for the plot
        x_label: label for the x axis
        y_label: label for the y axis
        z_label: label for the z axis

    Returns
        None    
    '''
    # Error traps for incorrect data format
    if not isinstance(x_data, (list, np.ndarray)) or not isinstance(y_data, (list, np.ndarray)):
        raise TypeError("x_data and y_data must be a 2D NumPy array or a list of 1D NumPy arrays.")
    if len(x_data) == 0 or len(y_data) == 0:
        raise ValueError("Data lists are empty. Provide valid array lists.")
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same number of curves.")

    cmap_choice = plt.colormaps.get_cmap(colour)
    cmap = cmap_choice(np.linspace(0, 1, len(x_data)))

    # Subplots section
    if subplot and len(x_data) > 1:
        # Creating the figure and axes
        fig, ax = plt.subplots(len(x_data), 1, figsize=(8, 3 * len(x_data)), sharex=True)

        # Plotting, labels and grid
        for i, (x, y, z) in enumerate(zip(x_data, y_data, z_data)):
            ax[i].plot(x, y, color=cmap[i], linewidth=1.5)
            ax[i].set_title('$Hr$ = {:.1f} (A/m)'.format(z[0]), fontsize=12)
            ax[i].set_ylabel('$M$ (A/m)', fontsize=10)
            ax[i].grid(True, alpha=0.3)

        ax[-1].set_xlabel(x_label, fontsize=10)

    # Overlay section
    else:
        # Creating figure and axes
        fig, ax = plt.subplots(figsize=(9,6))

        # Colour selection and plotting
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            color = 'b' if len(x_data) == 1 else cmap[i]
            ax.plot(x, y, color=color, linewidth=1.5)

        # Linking colourmap to colourbar
        if len(x_data) > 1:
            norm = plt.Normalize(vmin=min(np.min(hr) for hr in z_data), vmax=max(np.max(hr) for hr in z_data))
            sm = plt.cm.ScalarMappable(norm=norm, cmap=colour)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=z_label)

        # Setting the labels and grid
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Layout, save and show
    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


def plot_contour(
        x_data: list[np.ndarray],
        y_data: list[np.ndarray],
        z_data: list[np.ndarray],

        lim_up: int = None,
        lim_low: int = None,
        lim_quant: int = None,

        three_d: bool = False,
        angle_x: int = 30,
        angle_y: int = 240, 
        angle_z: int = 0,

        xlim: float = None,
        ylim: float = None,

        colour: str = 'inferno',
        extend: str = 'both',
        save: bool = False,

        path: str = 'fieldspace_plot.png',
        title: str = 'Raw Data in Field Space',
        x_label: str = 'Applied Field $H$ (A/m)',
        y_label: str = 'Reversal Field $Hr$ (A/m)',
        z_label: str = 'Magnetisation $M$ (A/m)'
) -> None:
    '''
    Plots a the reversal loops in a contour plot, called field space where the contouring
    represents the magnetisation.

    Parameters
        x_data: list of 1D arrays; applied field (h) or coercive field (hc)
        y_data: list of 1D arrays; reversal points (hr) or interaction field (hu)
        z_data: list of 1D arrays; magnetisation data (m) or forc density

        lim_up, lim low: the limits of the colour bar
        lim_quant: the amount of ticks on the colour bar

        three_d: if statement to print a contour or a 3d plot
        angles: the viewing angles of the 3d plot
        x_lim, y_lim: (x1, x2), limiting the viewing axisi
        
        colour: colour choice for the plot
        extend: both to extend the colour bar, None to not
        save: true to save the plot, false to not save the plot

        path: filepath destination 
        title: title for the plot
        x_label: label for the x axis
        y_label: label for the y axis
        z_label: label for the colour bar

    Returns
        None    
    '''
    # Error traps
    for i in range(len(x_data)):
        diffs = np.diff(x_data[i])
        if np.any(diffs < -1e-5):
            raise ValueError(
            'Input data contains a returning field path or full loop.' \
            'Ensure data is split to include only the reversal trip (H-min to H_max).'
            )

    # Making the levels for the colour bar
    if lim_up is None or lim_low is None or lim_quant is None: 
            levels = None
    else: 
        levels = np.linspace(lim_low, lim_up, lim_quant)

    # 3D plot section
    if three_d:
        # Initialise the figure and axes
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')

        if isinstance(z_data, np.ndarray) and z_data.ndim == 2:
            surface = ax.plot_surface(x_data, y_data, z_data, cmap=colour, edgecolor='none', antialiased=True)
            fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10, label=z_label)
            ax.view_init(angle_x, angle_y, angle_z)
        else:
            ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap=colour)
            ax.set_zlabel(z_label)
            ax.view_init(angle_x, angle_y, angle_z)

    # Contour section
    else:
        # Initialise the figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(9,6))     
        cp = ax.contourf(x_data, y_data, z_data, cmap=colour, levels=levels, extend=extend)

        # Set the colour bars
        cbar = plt.colorbar(cp)
        cbar.ax.tick_params(axis='both', which='both', length=2)
        cbar.ax.minorticks_off()
        cbar.ax.set_ylabel(z_label)

    # Set title and axes labels
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

    # Layout, save and show
    plt.tight_layout()
    if save:
        plt.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
    plt.show()