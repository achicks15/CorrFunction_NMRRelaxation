# CorrFunction_NMRRelaxation
General Overview of Code:
Python code to calculate the protein backbone NMR relaxation 1H-15N T1, T2 and heteronuclear NOE parameters via fitting the N-H bond correlation functions to a sum of exponentials


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
    Required Python version and modules:
    Python 3.5:
        1. Numpy v1.17.3
        2. Pandas v0.25.2
        3. Scipy v1.3.1
        4. mdtraj v1.9.2
            a. See https://github.com/mdtraj/mdtraj or http://mdtraj.org/1.9.2/ for installation/documentation
    
        The latest anaconda environment for this version is listed in "NMRRelax_CondaEnv.txt".
            "conda create --name NMRRelaxEnv --file NMRRelax_CondaEnv.txt"

    The code can be split into three different parts: calculation of correlation functions, fitting of the correlation functions, calculation of NMR relaxation parameters. The benefit of the ipynb format of the code is that the user can choose which sections they want to use. For instance, if the user has precalculated the correlation functions, i.e. with the vector and timecorr functions in cpptraj, 
    1. Calculation of Correlation Functions:
        The correlation functions

Acknowledgements:
    Several functions were adapted and updated from https://github.com/zharmad/SpinRelax according to the MIT License.  These include "split_NHVecs", calc_Ct, _bound_check, calc_chi, _return_parameter_names, do_Expstyle_fit2, findbest_Expstyle_fits2. 