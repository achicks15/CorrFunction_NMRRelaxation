# CorrFunction_NMRRelaxation
General Overview of Code:
Python code to calculate the protein backbone NMR relaxation 1H-15N T1, T2 and heteronuclear NOE parameters via fitting the N-H bond correlation functions to a sum of exponentials

Originally written as a standalone Jupyter notebook, but is in the process of being adapted to stand alone script.

Outline of Code:
1. Calculate the backbone N-H bond vectors
    a. Load trajectory using mdtraj
2. Calculate the angular correlation functions, C(t) = <P2(NH(t+t0)*NH(t0))>, where P2 is the 2nd order Legendre polynomials
3. Fit correlation functions to a sum of exponentials
    a. Use a model fitting procedure if you don't know how many exponentials you need to fit
    b. Or fix the fit to a set sum of exponentials
4. Direct fourier transform fitted amplitudes and correlation times to get the spectral densities
5. Calculate the NMR relaxation parameters for a specified magnetic field

Using This Code:
