from physical_constants import *
from colossus.lss import mass_function
from scipy.special import gamma, expm1
import numpy as np
from Config import SHMF_model
import matplotlib.pyplot as plt

#output dn/dM in the unit of [(Mpc/h)^(-3) (Msun/h)^(-1)]
#input M in the unit of Msun/h
def HMF_Colossus(M, z, model = 'press74'):
    mfunc = mass_function.massFunction(M, z, model = model, q_out = 'M2dndM')
    return mfunc/M**2*rho_m0*(1+z)**3/h_Hubble**2


def plot_hmf(halos, index_selected, current_redshift, dark_matter_resolution, simulation_volume, hmf_filename):
    '''
    Parameters:
    halos: the full halo catalog in TNG
    current_redshift: the redshift of the snapshot
    dark_matter_resolution: the resolution of the dark matter particles
    simulation_volume: the simulation_volume of the simulation box in unit of (Mpc/h)^3
    hmf_filename: the output filename of the plot

    '''
    #plot HMF (halo mass function)
    M_all = halos['GroupMass']*1e10   #unit: 1e10 Msun/h
    selected_M_all = M_all[index_selected]
    max_M = np.max(M_all)

    # Create a histogram (logarithmic bins and logarithmic mass)
    bins = np.logspace(np.log10(min(M_all)), np.log10(max(M_all)), num=50)
    hist, bin_edges = np.histogram(M_all, bins=bins)
    hist_selected, _ = np.histogram(selected_M_all, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Convert counts to number density
    log_bin_widths = np.diff(np.log10(bins))
    number_density = hist / simulation_volume / log_bin_widths
    number_density_selected = hist_selected / simulation_volume / log_bin_widths

    # Plot the mass function
    fig = plt.figure(facecolor='white')
 
    logM_limits = [6, np.log10(1.1*max_M)]  # Limits for log10(M [Msun/h])
    HMF_lgM_press74 = []
    HMF_lgM_sheth99 = []
    logM_list = np.linspace(logM_limits[0], logM_limits[1],57)
    #plot analytical HMF
    for logM in logM_list:
        M = 10**(logM)
        HMF_lgM_press74.append(HMF_Colossus(10**logM, current_redshift, 'press74')* np.log(10)*M)  
        HMF_lgM_sheth99.append(HMF_Colossus(10**logM, current_redshift, 'sheth99')* np.log(10)*M)
    #plot the dark matter resolution and TNG HMF
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(100*dark_matter_resolution, color='black', linestyle='--')
    plt.scatter(bin_centers, number_density, c='none', edgecolor='blue', marker='o', label='All TNG halos')
    plt.scatter(bin_centers, number_density_selected, c='none', edgecolor='green', marker='^',label='Selected TNG halos')

    plt.plot(10**(logM_list),HMF_lgM_press74,color='k',linestyle='-',label='Press74')
    plt.plot(10**(logM_list),HMF_lgM_sheth99,color='red',linestyle='-',label='Sheth99')
    plt.legend(fontsize=13)
    
    plt.xlabel(r'Mass [$\mathrm{M}_{\odot}/\mathrm{h}$]', fontsize=14)
    plt.ylabel(r'$\frac{\text{dN}}{\text{ d\lg M}}$ [$(\text{Mpc/h})^{-3}$]',fontsize=14)
    plt.tight_layout()
    plt.savefig(hmf_filename,dpi=300)
    





#return the bestfit parameters (alpha beta_ln10 omega lgA)
def read_SHMF_fit_parameters(filename):
    with open(filename, 'r') as f:
        f.readline()
        lines = f.readlines()
    #only use the last line
    #snapNum bin_index logM_min logM_max alpha beta_ln10 omega lgA
    last_line = lines[-1]
    params = last_line.split()
    print("BestFit parameters: ",params)
    return np.array([float(p) for p in params[4:]])


def get_M_Jeans(z):
    return 220*(1+z)**1.425*h_Hubble


#x  = m/M
#dN/dlgx = A* x**(-alpha) exp(-beta x**omega)
#lg[dN/dlgx] = lgA - alpha lgx - beta x**omega / ln(10)
def fitFunc_lg_dNdlgx(lgx,alpha,beta_ln10, omega, lgA):
    x = 10**lgx
    return lgA - alpha*lgx - beta_ln10*x**omega 

#fix omega = 4 and beta_ln10 = 50/ln(10)
def fitFunc_lg_dNdlgx_fixomega(lgx,alpha,lgA):
    x = 10**lgx
    beta_ln10 = 50/np.log(10)
    omega = 4
    return lgA - alpha*lgx - beta_ln10*x**omega


def write_SHMF_fit_parameters(filename,all_fitting_params):
    with open(filename, 'w') as f:
        f.write('snapNum bin_index logM_min logM_max alpha beta_ln10 omega lgA\n')
        for params in all_fitting_params:
            f.write(' '.join(map(str,params)) + '\n')

#return the bestfit parameters (alpha beta_ln10 omega lgA)
def read_SHMF_fit_parameters(filename):
    with open(filename, 'r') as f:
        f.readline()
        lines = f.readlines()
    #only use the last line
    #snapNum bin_index logM_min logM_max alpha beta_ln10 omega lgA
    last_line = lines[-1]
    params = last_line.split()
    print("BestFit parameters: ",params)
    return np.array([float(p) for p in params[4:]])







#dN/ d lnx, x = m/M, input ln(x)
def Subhalo_Mass_Function_ln(ln_m_over_M,bestfitparams=None):
    if SHMF_model == 'Giocoli2010':
        m_over_M = np.exp(ln_m_over_M)
        f0 = 0.1
        beta = 0.3
        gamma_value = 0.9
        x = m_over_M/beta
        return f0/(beta*gamma(1 - gamma_value)) * x**(-gamma_value) * np.exp(-x)
    elif SHMF_model == 'Bosch2016':
        #x  = m/M
        #dN/dlgx = A* x**(-alpha) exp(-beta x**omega)
        #dN/d ln(x) = dN/dlgx /ln10 = A* x**(-alpha) exp(-beta x**omega) / ln10
        #fitFunc_lg_dNdlgx(lgx,alpha,beta_ln10, omega, lgA)
        #p_guess = [0.86, 50/np.log(10),4 ,np.log10(0.065)]
        m_over_M = np.exp(ln_m_over_M)
        alpha = 0.86
        beta = 50
        omega = 4
        A = 0.065
        return A* m_over_M**(-alpha) * np.exp(-beta * m_over_M**omega) / np.log(10)
    elif SHMF_model == 'BestFit':
        #(alpha,beta_ln10, omega, lgA) provided by bestfitparams
        if bestfitparams is None:
            print("Error: bestfitparams is None")
            return -999
        m_over_M = np.exp(ln_m_over_M)
        alpha = bestfitparams[0]
        beta = bestfitparams[1]
        omega = bestfitparams[2]
        A = 10**bestfitparams[3]
        return A* m_over_M**(-alpha) * np.exp(-beta * m_over_M**omega) / np.log(10)
        
def Subhalo_Mass_Function_dN_dlgm(m, M, bestfitparams=None):
    #dN/dlgm = dN/dlnx * dlnx/dlgx * dlgx/dlgm, x = m/M
    ln_m_over_M = np.log(m/M)
    dN_dlnx = Subhalo_Mass_Function_ln(ln_m_over_M,bestfitparams)
    dlnx_dlgx = np.log(10)
    dlgx_dlgm = 1  #dlg(m/M)/dlg(m/[Msun])
    return dN_dlnx * dlnx_dlgx * dlgx_dlgm

