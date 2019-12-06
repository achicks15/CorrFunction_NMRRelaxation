#!/usr/bin/env python
# coding: utf-8
## Author: Alan Hicks
# In[1]:


import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import mdtraj as md



# In[2]:


def split_NHVecs(nhvecs, dt, tau):
    """
    This function will split the trajectory in chunks defined by tau. 
    nhvecs = array of N-H bond vectors,
    dt = timestep of the simulation
    tau = length of chunks  
    """
    nFiles = len(nhvecs) ## number of trajectories
    nFramesPerChunk = int(tau/dt) ###tau/timestep 
    used_frames = np.zeros(nFiles,dtype=int)
    remainingFrames = np.zeros(nFiles,dtype=int)
    for i in range(nFiles):
        nFrames = nhvecs[i].shape[0]
        used_frames[i] = int(nFrames/nFramesPerChunk)*nFramesPerChunk
        remainingFrames[i] = nFrames % nFramesPerChunk
    
    nFramesTot=int(used_frames.sum())
    out = np.zeros((nFramesTot,NHVecs[0].shape[1],NHVecs[0].shape[2]), dtype=NHVecs[0].dtype)
    start = 0
    for i in range(nFiles):
        end = int(start+used_frames[i])
        endv = int(used_frames[i])
        out[start:end,...] = nhvecs[i][0:endv,...]
        start = end
        
    sh = out.shape
    vecs = out.reshape((int(nFramesTot/nFramesPerChunk), nFramesPerChunk, sh[-2], sh[-1]))
    
    return vecs


# In[3]:


def calc_Ct(nhvecs):
    """
    Calculates the correlation function of the N-H bond vectors found in nhvecs. 
    """
    sh = nhvecs.shape
    nReplicates=sh[0] ; nDeltas=int(sh[1]/2) ; nResidues=sh[2]
    Ct  = np.zeros( (nDeltas, nResidues), dtype=nhvecs.dtype )
    dCt = np.zeros( (nDeltas, nResidues), dtype=nhvecs.dtype )
    
    for delta in range(1,1+nDeltas):
        nVals=sh[1]-delta
        # = = Create < vi.v'i > with dimensions (nRep, nFr, nRes, 3) -> (nRep, nFr, nRes) -> ( nRep, nRes ), then average across replicates with SEM.
        tmp = -0.5 + 1.5 * np.square( np.einsum( 'ijkl,ijkl->ijk', nhvecs[:,:-delta,...] , nhvecs[:,delta:,...] ) )
        tmp  = np.einsum( 'ijk->ik', tmp ) / nVals
        Ct[delta-1]  = np.mean( tmp, axis=0 )
        dCt[delta-1] = np.std( tmp, axis=0 ) / ( np.sqrt(nReplicates) - 1.0 )
    
    return Ct, dCt


# In[5]:


def _bound_check(func, params):
    """
    Checks if the fit returns a sum of the amplitudes greater than 1.
    """
    if len(params) == 1:
        return False
    elif len(params) %2 == 0 :
        s = sum(params[0::2])
        return (s>1)
    else:
        s = params[0]+sum(params[1::2])
        return (s>1)


# In[6]:


def calc_chi(y1, y2, dy=[]):
    """

    calculates the chi^2 difference between the predicted model and the actual data
    """
    if dy != []:
        return np.sum( (y1-y2)**2.0/dy )/len(y1)
    else:
        return np.sum( (y1-y2)**2.0 )/len(y1)


# In[8]:


## Functions 1,3,5,7,9 are the functions that the sum of coefficients are equal to 1. They have one less parameter.
## Functions 2,4,6,8,10 are the functions where the sum of coefficients are not restricted.

def func_exp_decay1(t, tau_a):
    return np.exp(-t/tau_a)
def func_exp_decay2(t, A, tau_a):
    return A*np.exp(-t/tau_a)
def func_exp_decay3(t, A, tau_a, tau_b):
    return A*np.exp(-t/tau_a) + (1-A)*np.exp(-t/tau_b)
def func_exp_decay4(t, A, tau_a, B, tau_b ):
    return A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b)
def func_exp_decay5(t, A, tau_a, B, tau_b, tau_g ):
    return A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + (1-A-B)*np.exp(-t/tau_g)
def func_exp_decay6(t, A, tau_a, B, tau_b, G, tau_g ):
    return A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g)
def func_exp_decay7(t, A, tau_a, B, tau_b, G, tau_g, tau_d):
    return A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + (1-A-B-G)*np.exp(-t/tau_d)
def func_exp_decay8(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d):
    return A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d)
def func_exp_decay9(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d, tau_e):
    return A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d) + (1-A-B-G-D)*np.exp(-t/tau_e)
def func_exp_decay10(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d, E, tau_e):
    return A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d) + E*np.exp(-t/tau_e)


# In[7]:


def _return_parameter_names(num_pars):
    """
    Function that returns the names of the parameters for writing to the dataframe after the fit.
    num_pars is the number of parameters in the fit. 1,3,5,7,9 are the num_params that constrain the fit.
    while the even numbers are the parameters for the functions that don't constrain the fits.
    """
    if num_pars==1:
        return ['C_a', 'tau_a']
    elif num_pars==2:
         return ['C_a', 'tau_a']
    elif num_pars==3:
         return ['C_a', 'tau_a', 'tau_b']
    elif num_pars==4:
         return ['C_a', 'tau_a', 'C_b', 'tau_b']
    elif num_pars==5:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'tau_g']
    elif num_pars==6:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g']
    elif num_pars==7:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'tau_d']
    elif num_pars==8:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d']
    elif num_pars==9:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d', 'tau_e']
    elif num_pars==10:
         return [ 'C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d', 'C_e', 'tau_e']

    return []


# In[9]:


def do_Expstyle_fit2(num_pars, x, y, dy=np.empty([]), tau_mem=50.):
    """
    Performs the exponential fit on the function defined by num_pars using scipy optimize curve fit.
    Provides initial guesses for the amplitudes and the correlation times.
    Takes the number of parameters, x values, y values, error in the y (dy), and tau_mem.
    Tau_mem is the maximum tau that the correlation times can take, which bound the fit.
    Can also be set to np.inf if you want no bounds. 
    
    Returns, the Chi-squared value of the fit to the model along with the parameter values (popt),
    the parameter error (popv) and the model itself.
    """
    
    b1_guess = y[0]/num_pars/2 
    t1_guess = [tau_mem/1280.0, tau_mem/640.0, tau_mem/64.0, tau_mem/8.0]
    
    if num_pars==1:
        func=func_exp_decay1
        guess=(t1_guess[2])
        bound=(0.,np.inf)
    elif num_pars==2:
        func=func_exp_decay2
        guess=(b1_guess, t1_guess[2])
        bound=([0.0, x[0]],[1., tau_mem])
    elif num_pars==3:
        func=func_exp_decay3
        guess=(b1_guess, t1_guess[3], t1_guess[2])
        bound=([0.0,x[0],x[0]],[1., tau_mem, tau_mem])
    elif num_pars==4:
        func=func_exp_decay4
        guess=(b1_guess, t1_guess[3], b1_guess, t1_guess[2])
        bound=([0.0, x[0], 0.0, x[0]],[1., tau_mem, 1., tau_mem])
    elif num_pars==5:
        func=func_exp_decay5
        guess=(b1_guess, t1_guess[3], b1_guess, t1_guess[2], t1_guess[1])
        bound=([0.0, x[0], 0.0, x[0],x[0]],[1., tau_mem, 1., tau_mem, tau_mem])
    elif num_pars==6:
        func=func_exp_decay6
        guess=(b1_guess, t1_guess[3], b1_guess, t1_guess[2], b1_guess, t1_guess[1])
        bound=([0.0, x[0], 0.0, x[0], 0.0, x[0]],[1., tau_mem, 1., tau_mem, 1., tau_mem])
    elif num_pars==7:
        func=func_exp_decay7
        guess=(b1_guess, t1_guess[2], b1_guess, t1_guess[1], b1_guess, t1_guess[0],
               t1_guess[3])
        bound=([0.0, x[0], 0.0, x[0], 0.0, x[0], x[0]],[1., tau_mem, 1., tau_mem, 1., tau_mem, tau_mem])
    elif num_pars==8:
        func=func_exp_decay8
        guess=(b1_guess, t1_guess[3], b1_guess, t1_guess[2], b1_guess, t1_guess[1],
               b1_guess, t1_guess[0])
        bound=([0.0, x[0], 0.0, x[0], 0.0, x[0], 0.0, x[0]],[1., tau_mem, 1., tau_mem, 1., tau_mem, 1., tau_mem])

    if dy != []:
        popt, popv = curve_fit(func, x, y, p0=guess, sigma=dy, bounds=bound, method='trf', loss='soft_l1')
    else:
        popt, popv = curve_fit(func, x, y, p0=guess, bounds=bound, loss='soft_l1')

    ymodel=[ func(x[i], *popt) for i in range(len(x)) ]
    #print ymodel

    bExceed=_bound_check(func, popt)
    if bExceed:
        print >> sys.stderr, "= = = WARNING, curve fitting in do_LSstyle_fit returns a sum>1.//"
        return 9999.99, popt, np.sqrt(np.diag(popv)), ymodel
    else:
        return calc_chi(y, ymodel, dy), popt, popv, ymodel


# In[11]:


def findbest_Expstyle_fits2(x, y, taum=150.0, dy=[], bPrint=True, par_list=[2,3,5,7,9], threshold=1.0):
    """
        Function tries to find the best set of parameters to describe the correlation fucntion for each residues
        Takes the x,y values for the fit and the errors, dy. par_list is the number of parameters to check,
        threshold is the cutoff for the chi2. This is the old way of checking, but can be re-implemented.
        Runs the fit for a given parameter by calling do_Expstyle_fit3. The initial fit is chosen, but 
        subsequent fits are chosen with a strict criteria based on the ratio of the number of parameters from 
        the current best fit and the latest fit.
        
        Returns the chi^2, names of the parameters, parameters, errors, model, and covariance matrix of the best fit.
        
    """
    
    chi_min=np.inf
    # Search forwards
    print('Starting New Fit')
    for npars in par_list:
        print(npars)
        names = _return_parameter_names(npars)
        try:
            chi, params, covarMat, ymodel = do_Expstyle_fit2(npars, x, y, dy, taum)
        except:
            print(" ...fit returns an error! Continuing.")
            break
        bBadFit=False
        
        errors = np.sqrt(np.diag(covarMat))
        step_check = 0
        
        while step_check < npars:
            
            ## 1st Check to see if any of the taus are equal to the maximum tau_mem
            tau_chk = np.log(abs(params[step_check]-taum))
            if (tau_chk < 0.0):
                print( " --- fit shows one of the taus is equal to Tau_Mem with %d parameters." % npars )
                print(  "  --- Occurred with parameter %s: %g == %g " % (names[step_check], params[step_check], taum ))
                print(  " -- Setting parameter %s = %s and %s = %s to 0.0" % (names[step_check], params[step_check],
                                                                              names[step_check-1], params[step_check-1]))
                print(  " -- Rerunning Fit with the previous exponentials using new params as initial guesses ")
                
                bBadFit=True
                
            ## 2nd Check the error to make sure there is no overfitting
            chkerr = errors[step_check]/params[step_check]
            if (chkerr>0.10):
                print( " --- fit shows overfitting with %d parameters." % npars)
                print(  "  --- Occurred with parameter %s: %g +- %g " % (names[step_check], params[step_check],
                                                                         errors[step_check]))
                bBadFit=True
                break
            
            step_check += 1
            
                
        chi_check = chi/chi_min
        if npars == par_list[0]:
            threshold = 1.0
        else:
            threshold = (1-npar_min/npars)*0.5
                
        print("--- The chi_check for {} parameters is {}".format(npars, chi_check))
        print("--- The threshold for this check is {}".format(threshold))
        if (not bBadFit) and (chi/chi_min < threshold):
            chi_min=chi ; par_min=params ; err_min=errors ; npar_min=npars ; ymod_min=ymodel; covar_min = covarMat;
        else:
            break; 
            
    tau_min = par_min[1::2]
    sort_tau = np.argsort(tau_min)[::-1]
    nsort_params = np.array([[2*tau_ind, 2*tau_ind+1] for tau_ind in sort_tau]).flatten()
    
    err_min = err_min[nsort_params] 
    par_min = par_min[nsort_params]
    sort_covarMat = covar_min[:,nsort_params][nsort_params]
    names = _return_parameter_names(npar_min)    
    
    if bPrint:       
        print( "= = Found %d parameters to be the minimum necessary to describe curve: chi(%d) = %g vs. chi(%d) = %g)" % (npar_min, npar_min, chi_min,  npars, chi))
        print( "Parameter %d %s: %g +- %g " % (npar_min, len(names), len(par_min), len(err_min)))
        for i in range(npar_min):
            print( "Parameter %d %s: %g +- %g " % (i, names[i], par_min[i], err_min[i]))
        print('\n')   
    return chi_min, names, par_min, err_min, ymod_min, sort_covarMat


# In[12]:

def fitstoDF(resnames, chi_list, pars_list, errs_list, names_list):
    ## Set Up columns indices and names for the data frame
    """
    Function that takes the residue names, chi^2, parameters, errors and names of the fits and returns a data frame
    of the parameters.
    """
    mparnames = _return_parameter_names(8) ## Always return the longest possible number of 
    mtau_names = np.array(mparnames)[1::2]
    mc_names = np.array(mparnames)[::2]
    colnames = np.array(['Resname','NumExp'])
    tau_errnames = np.array([[c,"{}_err".format(c)] for c in mtau_names]).flatten()
    mc_errnames = np.array([[c, "{}_err".format(c)] for c in mc_names]).flatten()
    colnames = np.hstack([colnames,mc_errnames])
    colnames = np.hstack([colnames,tau_errnames])
    colnames = np.hstack([colnames,np.array(['Chi_Fit'])])
    FitDF = pd.DataFrame(index=np.arange(len(pars_list)), columns=colnames).fillna(0.0)
    FitDF['Resname'] = resnames
    FitDF['Chi_Fit'] = chi_list
    
    for i in range(len(pars_list)):
        npar = len(pars_list[i])
        if (npar%2)==1:
            ccut = npar-2
            tau_f, terr = pars_list[i][1:ccut+1:2], errs_list[i][1:ccut+1:2]
            tau_f = np.hstack([tau_f, pars_list[i][-1]])
            terr = np.hstack([terr, errs_list[i][-1]])
            sort_tau = np.argsort(tau_f)
            coeff, cerr= pars_list[i][0:ccut:2], errs_list[i][0:ccut:2]
            Clast = 1; Clasterr = 0.0;
            for n,m in zip(coeff, cerr):
                Clast -= n
                Clasterr += m
            
            coeff =np.hstack([coeff, np.array(Clast)])
            cerr =np.hstack([cerr, np.array(Clasterr)])
    
            tne = np.array([[c,"{}_err".format(c)] for c in mparnames[1:npar+1:2]]).flatten()
            cne = np.array([[c, "{}_err".format(c)] for c in mparnames[0:npar:2]]).flatten()
                
        else:
            tau_f, terr = pars_list[i][1::2], errs_list[i][1::2] 
            coeff, cerr= pars_list[i][0::2], errs_list[i][0::2]
            sort_tau = np.argsort(tau_f)[::-1]
            tne = np.array([[c,"{}_err".format(c)] for c in names_list[i][1::2]]).flatten()
            cne = np.array([[c, "{}_err".format(c)] for c in names_list[i][0::2]]).flatten()
    
        NumExp=np.array(len(tau_f))
        tau_err = np.array([[t,e] for t,e in zip(tau_f[sort_tau],terr[sort_tau])]).flatten()
        c_err = np.array([[c,e] for c,e in zip(coeff[sort_tau], cerr[sort_tau])]).flatten()
        namesarr = np.hstack([np.array('NumExp'),cne,tne])
        valarr = np.hstack([NumExp,c_err,tau_err])
    
        FitDF.loc[i,namesarr] = valarr
        
    FitDF['AUC_a'] = FitDF.C_a*FitDF.tau_a; FitDF['AUC_b'] = FitDF.C_b*FitDF.tau_b; 
    FitDF['AUC_g'] = FitDF.C_g*FitDF.tau_g; FitDF['AUC_d'] = FitDF.C_d*FitDF.tau_d;
    FitDF['AUC_Total'] = FitDF[['AUC_a','AUC_b','AUC_g','AUC_d']].sum(axis=1)
    
    return FitDF


# In[13]:


def fitCorrF(CorrDF, dCorrDF, tau_mem, pars_l, fixfit=False):
    """
        Input Variables:
            CorrDF: Dataframe containing the correlation functions. Columns are the NH-bond vectors, rows are timesteps. 
            dCorrDF: Error in the correlation function at time t
            tau_mem: Cut-Off time to remove noise at the tail of the correlation function 
            pars_l : parameters list. 
            fixfit : Boolean to decide if you want to use a specific exponential function 
        Main function to fit the correlation function. 
        Loops over all residues with N-H vectors and calculates the fit, appends the best fit from findbest_Expstyle_fits2.
        Passes the set of lists to fitstoDF to return a data frame of the best fits for each residue. 
        
        Takes the correlation function CorrDF and errors in the correlation function, maximum tau mem to cut correlation
        function off from, the list of parameters you want to fit too. If you don't want to test the fit and use 
        a fixed parameter set, set fixfit to True and pass a list of length 1 into pars_l.
    """
    NH_Res = CorrDF.columns
    chi_list=[] ; names_list=[] ; pars_list=[] ; errs_list=[] ; ymodel_list=[]; covarMat_list = [];
    for i in CorrDF.columns:
        
        tstop = np.where(CorrDF.index.values==tau_mem)[0][0]
            
        x = CorrDF.index.values[:tstop]
        y = CorrDF[i].values[:tstop]
        dy = dCorrDF[i].values[:tstop]
        
        ## if not fixfit then find find the best expstyle fit. Otherwise force the fit to nparams 
        if (not fixfit)&(len(pars_l)>1):
            print("Finding the best fit for residue {}".format(i))
            
            chi, names, pars, errs, ymodel, covarMat = findbest_Expstyle_fits2(x, y, tau_mem, dy,  
                                                            par_list=pars_l, threshold=thresh)
        
        elif (fixfit)&(len(pars_l)==1):
            print("Performing a fixed fit for {} exponentials".format(int(pars_l[0]/2)))
            
            chi, pars, covarMat, ymodel = do_Expstyle_fit2(pars_l[0], x, y, dy, tau_mem)
            names = _return_parameter_names(len(pars))
            errs = np.sqrt(np.diag(covarMat))
            
        else:
            print("The list of parameters is empty. Breaking out.")
            break;
            
        chi_list.append(chi)
        names_list.append(names)
        pars_list.append(pars)
        errs_list.append(errs)
        ymodel_list.append(ymodel)
        covarMat_list.append(covarMat)
        
    FitDF = fitstoDF(NH_Res, chi_list, pars_list, errs_list, names_list)
    
    return FitDF, covarMat_list


# In[14]:


def J_direct_transform(om, consts, taus):
    
    """
        Calculation of the spectral density from the parameters of the fit by direct fourier transform
    """
    ## Calculation for the direct spectral density 
    ndecay=len(consts) ; noms=1;###lnden(om)
    Jmat = np.zeros( (ndecay, noms ) )
    for i in range(ndecay):
        Jmat[i] = consts[i]*(taus[i]*1e-9)/(
            1 + np.power((taus[i]*1e-9)*(om),2.))
    return Jmat.sum(axis=0)


# In[15]:


def calc_NMR_Relax(J, fdd, fcsa, gammaH, gammaN):
    """
        Function to calculate the R1, R2 and NOE from the spectral densities and the physical parameters for the 
        dipole-dipole and csa contributions, fdd and fcsa. 
    """
    R1 = fdd * (J['Diff'] + 3*J['15N'] + 6*J['Sum']) + fcsa * J['15N']
    
    R2 = (0.5 * fdd * (4*J['0'] + J['Diff'] + 3*J['15N'] + 6*J['1H'] + 6*J['Sum']) 
          + (1./6.) * fcsa*(4*J['0'] + 3*J['15N']) )
    
    NOE = 1 + ((fdd*gammaH)/(gammaN*R1))*(6*J['Sum'] - J['Diff'])
    
    return R1, R2, NOE

## Ending function definitions and beginning main implementation of the code:
# In[18]:


## Global Variables for the calculation of the NH Vecs and the correlation functions
ChiZLoc = "/disks/zhou/d8/users/ah14k/ChiZ/Amber/"
FF=[['AMBER14SB','Tip4pD'],['AMBER03WS','Tip4p2005']]
FLOC = "/disks/zhou/d8/users/ah14k/ChiZ/Amber/{}/{}".format(FF[1][0],FF[1][1])
#FTOPN = "05_Prod.noH20.ChiZN_0.025M-NaCl_capped.prmtop"
FTOPN = "05_Prod.noH20.ChiZN_amber03ws.parm7"
RUN = ["Run{}".format(i) for i in range(1,5)]
JOBS = ['PROD','UIC/Job1','UIC/Job2']
TRAJLIST_LOC = ["{}/Analysis/{}".format(J,R) for J in JOBS for R in RUN]
FMDN = "05_Prod.noH20.nc"
NHVecs = []
dtL = 20.0  ## 20 ps
tauL = 2500000  ## 2500 ns == 2.5 us
dts = 0.1   ## 0.1 ps
tauS = 100  ## ps


# In[19]:

## Parameters and Physical Constants for calculation of Relaxation Rates

H_gyro = 2*np.pi*42.57748*1e6     ## Gyromagnetic Ratio: Hydrogen ([rad]/[s][T]) 
N_gyro = -2*np.pi*4.317267*1e6     ## Gyromagnetic Ratio: Nitrogen ([rad]/[s][T])
B0 = 18.8                        ## Field Strength = 18.8 Teslas

## Need 5 Frequencies: ## J[0], J[wH], J[wN], J[wH-wN], J[wH+wN]
Larmor1H = H_gyro*B0              ## Larmor Frequency: Hydrogen ([rad]/[s])
Larmor15N = N_gyro*B0             ## Larmor Frequency: Hydrogen ([rad]/[s])
omDiff = Larmor1H - Larmor15N    ## Diff in Larmor Frequencies of Spin IS
omSum  = Larmor1H + Larmor15N    ## Sum of Larmor Frequencies of Spin IS
vB = 800                         ## 800 MHz B-field

#mu_0 = 8.85418782e-12 ; # m^-3 kg^-1 s^4 A^2
mu_0 = 4*np.pi*1e-7    ; ## H/m
hbar = 1.0545718e-34  ; # [J] * [s] = [kg] * [m^2] * [s^-1] 
####omegaB = 2.0*np.pi*vB / 267.513e6  ##(800 MHz) ?????
R_NH = 1.02e-10                     ## distance between N-H atoms in Angstroms
dSigmaN = -170e-6 
#mu_0=1
###f_DD = 7.958699205571828e-67 * R_NH**-6.0 * N_gyro**2
FDD = (1./10.)*np.power((mu_0*hbar*H_gyro*N_gyro)/(4*np.pi*np.power(R_NH,3)),2)
#FCSA = 498637299.69233465
FCSA = (2.0/15.0)*(Larmor15N**2)*(dSigmaN**2)        ## CSA factor 


# In[25]:
## Load trajectories and calculate the NH-Vecs in the laboratory frame; Skip this if you have calculated it before
# In[27]:

""" 
    Uses mdtraj to load the trajectories and get the atomic indices and coordinates to calculate the correlation functions.
    For each, trajectory load the trajectory using mdtraj, get the atomic index for the the N-H atoms and calculate the vector between the two.
    Append the vector to the NHVecs list for all the trajectories. 
"""
for T in TRAJLIST_LOC:
    print(T)
    traj = md.load_netcdf("{}/{}/{}".format(FLOC,T,FMDN), top="{}/{}/{}".format(FLOC,T,FTOPN))
    top = traj.topology
    
    ##AtomSelection Indices
    Nit = top.select('name N and not resname PRO')
    Hyd = top.select('name H and not resname PRO')
    NH_Pair = [[i,j] for i,j in zip(Nit,Hyd)]
    NH_Pair_Name = [[top.atom(i),top.atom(j)] for i,j in NH_Pair]
    NH_Res = ["{}-{}{}".format(str(i).split('-')[0],str(i).split('-')[1], str(j).split('-')[1]) for i,j in NH_Pair_Name]
    
    ##Generate the N-H vectors in Laboratory Frame
    NHVecs_tmp = np.take(traj.xyz, Hyd, axis=1) - np.take(traj.xyz, Nit, axis=1)
    sh = list(NHVecs_tmp.shape)
    sh[2] = 1
    NHVecs_tmp = NHVecs_tmp / np.linalg.norm(NHVecs_tmp, axis=2).reshape(sh)
    if "UIC" in T:
        NHVecs.append(NHVecs_tmp[5000:145000])
    else:
        #skip = NHVecs_tmp.shape[0]-125000-1
        NHVecs.append(NHVecs_tmp[5000:145000])
        
    del traj

## Split the vecs based off the tau_m you want and the time step. 
# Here we calculate the correlation function over the full trajectory 
vecs_LS = split_NHVecs(NHVecs, 20, 2800000) ## Use all the points
# In[31]:

## Calculate the correlation functions and the standard deviation in the correlation function.
## Save the correlation functions in a dataframe and then to a csv file for later use.

Ct, dCt = calc_Ct(vecs_LS)
CtDF = pd.DataFrame(Ct, index = np.arange(1,Ct.shape[0]+1)*20/1000, columns=NH_Res)
dCtDF = pd.DataFrame(dCt, index = np.arange(1,dCt.shape[0]+1)*20/1000, columns=NH_Res)
CtDF.to_csv('/scratch/users/ah14k/ChiZ/Analysis/AMBER03WS/Ct_{}_comb_36us.csv'.format(tauL))
dCtDF.to_csv('/scratch/users/ah14k/ChiZ/Analysis/AMBER03WS/dCt_{}_comb_36us.csv'.format(tauL))

# Begin Curve Fitting; If you don't need to calculate the vectors then skip to here.

# In[20]:

## Load Experimental NOE data: This will depend on the shape of your NMR data
top14 = "{}AMBER14SB/Tip4pD/PROD/Analysis/07_Prod.noH20.ChiZN_0.025M-NaCl_capped.prmtop".format(ChiZLoc)
parm714 = md.load_topology(top14)
CAsel = parm714.select('name N and not resname PRO')
RESCaINFO = np.array(["{}".format(parm714.atom(x)) for x in CAsel])
RESINFO = np.array([x.replace('-N',"") for x in RESCaINFO])
EXPNOEF = "/scratch/users/ah14k/ChiZ/ChiZN164NOEpH7_New.csv"
EXPNOEdf = pd.read_table(EXPNOEF,delimiter=',',skiprows=1,header=None, names=['Residue Number','T1','T1_Err','T2','T2_Err','NOE','NOE_Err'])
EXPNOEdf = EXPNOEdf.drop(0)
EXPNOEdf = EXPNOEdf.replace('-',np.nan)
EXPNOEdf['RES'] = RESINFO
EXPNOEdf.iloc[:,1:5] = EXPNOEdf.iloc[:,1:5].astype('float')/1000
EXPNOEdf.iloc[:,5:7] = EXPNOEdf.iloc[:,5:7].astype('float')


# In[21]:
## Calculate mean array for the experimental data 
T1MEANArrExpNT = np.array([0.5732]*EXPNOEdf.loc[3:24].shape[0])
T1MEANArrExpCT = np.array([0.5824]*EXPNOEdf.loc[25:50].shape[0])
T2MEANArrExpNT = np.array([0.2112]*EXPNOEdf.loc[3:24].shape[0])
T2MEANArrExpCT = np.array([0.2405]*EXPNOEdf.loc[25:50].shape[0])
NOEMEANArrExpNT = np.array([0.34]*EXPNOEdf.loc[3:24].shape[0])
NOEMEANArrExpCT = np.array([0.25]*EXPNOEdf.loc[25:50].shape[0])


# In[151]:

## Load the correlation functions from the saved 
CtDF14 = pd.read_csv('/scratch/users/ah14k/ChiZ/Analysis/AMBER14SB/Ct_2500000_comb_36us.csv', index_col=0)
dCtDF14 = pd.read_csv('/scratch/users/ah14k/ChiZ/Analysis/AMBER14SB/dCt_2500000_comb_36us.csv', index_col=0)

# In[53]:


tau_mem=25.0; thresh=0.5; FF='AMBER14SB'; fthresh=False; ShortSim=False; fixfit=True;

if (FF == 'AMBER14SB')&(not ShortSim):
    print('Running Fits for {} without the Short Simulations'.format(FF))
    FitDF, covarMat_lists = fitCorrF(CtDF14, dCtDF14, tau_mem, [6], fixfit)
elif (FF == 'AMBER03WS')&(not ShortSim):
    print('Running Fits for {} without the Short Simulations'.format(FF))
    FitDF, covarMat_lists = fitCorrF(CtDF03, dCtDF03, tau_mem, [6], fixfit)
elif (FF == 'AMBER14SB')&(ShortSim):
    print('Running Fits for {} with the Short Simulations'.format(FF))
    FitDF, covarMat_lists = fitCorrF(CtDFComb14, dCtDFComb14, tau_mem, [6], fixfit)
elif (FF == 'AMBER03WS')&(ShortSim):
    print('Running Fits for {} with the Short Simulations'.format(FF))
    FitDF, covarMat_lists = fitCorrF(CtDFComb03, dCtDFComb03, tau_mem, [6], fixfit)
else:
    print("Not a valid force field")


# In[54]:
## Calculate spectral density from the FitDF by calling the J_direct_transform function for each of the 5 frequencies.
## Loop over the rows of the FitDF dataframe from fitCorrF function and calcuate the spectral densities.
## Save the spectral densities to a dictionary and append to a list.
Jarr = []

for i,fit in FitDF.iterrows():
    c = fit[['C_a','C_b','C_g','C_d']].values
    t = fit[['tau_a','tau_b','tau_g','tau_d']].values
    Jdict = {'0':0, '1H':0,'15N':0,'Sum':0,'Diff':0} 
    J0 = J_direct_transform(0, c, t)
    JH = J_direct_transform(Larmor1H, c, t)
    JN = J_direct_transform(Larmor15N, c, t)
    JSum = J_direct_transform(omSum,  c, t)
    JDiff = J_direct_transform(omDiff,  c, t)
    Jdict['1H'] = JH ; Jdict['15N'] = JN; Jdict['0'] = J0; 
    Jdict['Sum'] = JSum; Jdict['Diff'] = JDiff;
    Jarr.append(Jdict)


# In[55]:

## Calculate the NMR relaxation and save to a DF
NMRRelaxDF = pd.DataFrame(np.zeros((len(Jarr),3)),index=range(1,len(Jarr)+1), columns=['T1','T2','NOE'])
for index in range(1,len(Jarr)+1):
    r1, r2, noe = calc_NMR_Relax(Jarr[index-1], FDD, FCSA, H_gyro, N_gyro)
    NMRRelaxDF.loc[index,'T1'] = 1/r1; 
    NMRRelaxDF.loc[index,'T2'] = 1/r2; 
    NMRRelaxDF.loc[index,'NOE'] = noe; 

NMRRelaxDF['Resname'] = FitDF['Resname'].values
NMRRelaxDF['RESNUM'] = NMRRelaxDF['Resname'].str.extract('([0-9]+)',expand=False).astype('int')+1
## Calculation of RMSE between values:
RMSE={"{}_SE".format(nmr):(NMRRelaxDF[nmr]-EXPNOEdf[nmr])**2 for nmr in ['T1','T2','NOE']}


# In[56]:

## Merge the FitDF and NMRRelaxDF together into one DF. 
FitRelaxDF = FitDF.merge(NMRRelaxDF, how='left', left_on='Resname',right_on='Resname').set_index(NMRRelaxDF.index)
## Add the RMSE values to the dataframe merged
FitRelaxDF = FitRelaxDF.join(pd.DataFrame(RMSE),how='left')

## Save the final DataFrame to a csv
if (not ShortSim):
    FitRelaxName= "NMRFitRelax_Full_noRestrict_tauM{}ns_errchk10_thresh{}_36us_FixExp3_NBInf".format(int(tau_mem),'None')
else:
    FitRelaxName= "NMRFitRelax_Full_noRestrict_tauM{}ns_errchk10_thresh{}_30us_ShortSimComb".format(int(tau_mem), 'NPar3')
FitRelaxDF.to_csv('/scratch/users/ah14k/ChiZ/Analysis/{}/NMRRelax/{}.csv'.format(FF,FitRelaxName))

# In[26]:
#### Plot the whatever data you want here or in a different script. 
# In[53]:
