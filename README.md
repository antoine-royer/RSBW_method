# RSBW Method

By Louis PETIT, Amr HUSSEIN and Loris DELAFOSSE
Laboratoire de Chimie Quantique, Institut de Chimie, CNRS/Université de Strasbourg

## Files

* `RSBW_1_perturber.ipynb`
* `RSBW_2_perturbers.ipynb`

## Description

`RSBW_1_perturber.ipynb` (resp. `RSBW_2_perturbers.ipynb`) allows for calculations of the exact and approximated energies of a simple system, made up of two model states and one (resp. two) perturber(s) state(s).

The approximation schemes supported by both programs are: 2nd-order Rayleigh-Schrödinger perturbation theory, 2nd/3rd-order Brillouin-Wigner perturbation theory applied to the model space restricted Hamiltonian, 2nd-order combined RSBW perturbation theory.

Both programs produce raw data and curves showing either the energy or the relative difference to the exact energy for each approximation method. Please be careful when plotting relative differences: when the exact energy becomes low (near zero), the relative difference can become arbitrarily large while the approximation remains good. In such cases, an error is displayed above the graphs. To solve this problem, you can shift the energy reference or reduce the energy window that is displayed.


## How to use

Run all cells on a jupyter notebook, then choose whatever you want to display with the two upper cursors, and finally set the parameters of your system using the other cursors.
To recover raw data, use "np.save()".
