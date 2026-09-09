# forc-analysis-and-simulations

last updated 08/09/2026

A list of Python functions to simulate magnetic hysteresis loops using the Jiles-Atherton (JA) model. This library is capable of data generation, processing raw magnetisation data, and computing First-Order Reversal Curves (FORC) diagrams in 2D and 3D graphs.

## Features

* **Jiles-Atherton Simulation**: Models magnetic hysteresis using the JA model and can generate a single hysteresis loop or a complete dataset for FORC analysis.
* **Data Processing**: Functions to normalise the magnetisation, correct the Kerr drift from MOKE measurements, isolate the ascending reversal curve from hr to h_max and  interpolate the data onto a regular meshgrid.
* **FORC Density Calculation**: Smooths and computes the mixed second partial derivative $\rho(H, H_r)$ using polynomial kernel convolution.
* **Flexible Visualization**: Built-in 2D and 3D plotting functions for raw reversal loops, field-space grids, and FORC density axes Hc and Hu.
* **Interactive Notebook**: Includes a Jupyter Notebook demonstrating the functions for data generation, processing, calculation and plotting.

## Quick Installation
1. Clone this repository and install the required dependencies:
```bash
git clone https://github.com/Common98/forc-analysis-and-simulations.git 
cd forc-analysis-and-simulations
pip install -r requirements.txt
```

2. The easiest way to get started is by opening the included demonstration notebook:
```bash
jupyter notebook example_forc_analysis.ipynb
```

##  Repository Structure

```text
├── example_forc_analysis.ipynb # Demonstration notebook
├── func_data_process.py        # Normalise, loop splitting and interpolation
├── func_forc_analysis.py       # FORC density calculations
├── func_magnetics.py           # Core functions (Langevin, effective field, coercivity)
├── func_plot.py                # Plotting functions for 2D and 3D graphs and contour plots
├── sim_ja_model.py             # JA model execution and reversal curve generation
├── requirements.txt            # Dependency requirements
└── README.md                   # Project documentation
```

## Scientific Background
First-Order Reversal Curve (FORC) diagrams are used to analyse complex magnetic properties in multiphase materials, geological samples, and nanoscale magnetic systems. 

To collect FORC data, a sample is magnetically saturated at an applied field $H_{\text{max}}$, brought down to a specified reversal field $H_r$, and then swept back up to saturation while recording the magnetisation $M(H, H_r)$. Repeating this for a range of reversal fields yields a collection of ascending reversal curves which are then used to determine the FORC density.

### Mathematical Definition
 The local FORC density distribution $\rho(H, H_r)$ is calculated as the mixed second derivative of magnetization with respect to applied field $H$ and reversal field 
$H_r$:

$$\rho(H, H_r) = -\frac{1}{2} \frac{\partial^2 M(H, H_r)}{\partial H \, \partial H_r}$$

However, the axes of the diagram are transformed from $H$ and $H_r$ into the coercive field ($H_c$) and interaction field ($H_u$):


$$H_c = \frac{H - H_r}{2}, \quad H_u = \frac{H + H_r}{2}$$

Broad distributions along $H_c$ indicate a wide range of coercivities or multiple magnetic phases within the sample. Whereas, distributions along $H_u$ reflect magnetostatic or exchange interaction strengths. Symmetric, narrow distributions along $H_u$ imply largely reversible magnetization and coherent switching, whereas asymmetrical or horizontal/diagonal branching distributions indicate strong interactions or complex multi-domain switching.

### The JA Model
To simulate hysteresis loops and test the FORC code, we used the JA model. A collection of equations to determine irreversible magnetisation.

The effective field $H_e$ is the magnetic field experienced by the magnetic domains of the sample and incorporates domain coupling effects via parameter $\alpha$:

$$H_e = H + \alpha M$$

The anhysteretic or reversible magnetization $M_{\text{an}}$ is calculated using the Langevin function $L(x):$

$$L(x) = coth(x) - \frac{1}{x}$$
$$M_{an} = M_s coth\left(\frac{H_e}{a}\right) - \frac{a}{H_e}$$

An approximation for the Langevin function was used, if $\frac{H_e}{a} \le 1\times10^{-4}$ the Langevin function was approximated to; $L(x) \approx \frac{x}{3}$. This was used as a method to avoid any division by zero.

The total magnetization $M$ was then determined as the sum of the irreversible magnetisation ($M_{\text{irr}}$) and reversible magnetisation ($M_{\text{rev}}$):

$$\frac{d M_{\text{irr}}}{d H} = \frac{M_{\text{an}} - M_{\text{irr}}}{k \delta - \alpha (M_{\text{an}} - M_{\text{irr}})}$$

$$M_{\text{rev}} = c (M_{\text{an}} - M_{\text{irr}})$$

$$M = M_{\text{rev}} + M_{\text{irr}}$$

Model Parameters Summary:
* $M_s$: Saturation magnetization
* $a$: Domain shape/density parameter
* $\alpha$: Inter-domain coupling parameter
* $k$: Average energy needed to overcome domain pinning sites
* $c$: Reversibility coefficient
* $\delta$: Direction parameter ($\delta = +1$ for $dH/dt > 0$, $\delta = -1$ for $dH/dt < 0$)

### Smoothing Factor 
Because computing numerical second derivatives amplifies noise in experimental data (e.g. MOKE measurements), a local polynomial smoothing is applied using a designated smoothing factor (SF). A low SF, approximately 2, preserves sharp coercivity peaks and fine domain features, but is sensitive to noise within the data. A high SF, approximately 10, suppresses the experimental noise, but may over smooth the data, broadening distributions and potentially obscuring true physical features.

A more in-depth explanation on the scientific background of FORC analysis and the Python code's features and limitations is available at [9]: 
https://livrepository.liverpool.ac.uk/3194366/rities/Hysteresis.html.

## References
    [1] C. R. Pike et al. Characterizing interactions in fine magnetic particle systems using first order reversal curves. Journal of Applied Physics, 85(9): 6660–6667, 1999. doi:10.1063/1.370176.
    [2] A. P. Roberts, C. R. Pike, and K. L. Verosub. First-order reversal curve diagrams: A new tool for characterizing the magnetic properties of natural samples. Journal of Geophysical Research: Solid Earth, 105(B12):28461–28475, 2000. doi:10.1029/2000JB900326.
    [3] A. R. Muxworthy and A. P. Roberts. First-order-reversal-curve (forc) diagrams. In Encyclopedia of Geomagnetism and Paleomagnetism, pages 266–272. Springer, 2007. doi:10.1007/978-1-4020-4423-6 99.
    [4] B. C. Dodrill. First-order-reversal-curve analysis of permanent magnet materials, 2014. Available at: https://www.lakeshore.com/docs/default-source/about-us-document-library/publications/first-order-reversal-curve-analysis-of-permanent-magnet-materials.pdf?sfvrsn=ce4aa118_1.
    [5] J. Grafe et al. Application of magneto-optical kerr effect to first-order reversal curve measurements. Review of Scientific Instruments, 85:023901, 2014. doi:10.1063/1.4865135.
    [6] P. D. Murray et al. Forc analysis in python. GitHub repository, 2022. Available at: https://github.com/peytondmurray/PyFORC.
    [7] Paul et al. Equation describing magnetic hysteresis, 2013. Available at: https://physics.stackexchange.com/questions/60194/equation-describing-magnetic-hysteresis.
    [8] J. Chowdhury. Complex nonlinearity – hysteresis, 2021. Available at: https://ccrma.stanford.edu/~jatin/ComplexNonlinearities/Hysteresis.html.       
    [9] N. Cook. Synthesis of Thin Film Mn3Sn and the Magnetic Behaviour of Thin Film LiFe5O8. Master of Philosophy thesis, University of Liverpool, 2025. Available at: https://livrepository.liverpool.ac.uk/3194366/rities/Hysteresis.html.
