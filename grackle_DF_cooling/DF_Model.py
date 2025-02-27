
import sys
import os
from physical_constants import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import LogFormatter, LogLocator
import illustris_python as il
import h5py
from scipy.special import gamma, expm1
from scipy.integrate import solve_ivp, quad
from scipy.integrate import nquad
from scipy.optimize import curve_fit
from colossus.lss import mass_function
from DF_cooling_rate import run_cool_rate, get_Grackle_TDF, get_Grackle_TDF_nonEquilibrium
from dfnumerical_subsonic import *


def Vel_Virial_analytic(M_vir_in_Msun, z):  #van den Bosch Lecture 11
    #M_vir in solar mass, return virial velocity in m/s    
    global h_Hubble, Omega_m
    #Delta_vir = Overdensity_Virial(z)
    Delta_vir = 200
    V_vir = 163*1e3 * (M_vir_in_Msun/1e12*h_Hubble)**(1/3) * (Delta_vir/200)**(1/6) * Omega_m**(1/6) *(1+z)**(1/2)
    return V_vir #m/s

def Vel_Virial_numerical(M_vir_in_Msun, R_vir_Mpc):
    #M_vir in solar mass, R_vir in Mpc
    V_vir = np.sqrt(G_grav*M_vir_in_Msun*Msun/(R_vir_Mpc*Mpc))
    return V_vir #m/s    

def Temperature_Virial_analytic(M_vir_in_Msun,z):  #van den Bosch Lecture 15
    halo_profile_factor = 3.0/2.0
    V_vir = Vel_Virial_analytic(M_vir_in_Msun, z)
    T_vir = halo_profile_factor* mu*mp*V_vir**2 /kB/3
    return T_vir
    
#only valid for ionized gas
# def Temperature_Virial(M,z): #van den Bosch Lecture 15
#     #M in solar mass, return virial temperature in K
#     return 3.6e5 * (Vel_Virial(M,z)/1e3/100)**2

def Temperature_Virial_numerical(M_vir_in_Msun, R_vir_Mpc):
    #M_vir in solar mass, R_vir in Mpc
    halo_profile_factor = 3.0/2.0
    V_vir = Vel_Virial_numerical(M_vir_in_Msun, R_vir_Mpc)
    T_vir = halo_profile_factor* mu*mp*V_vir**2 /kB/3
    return T_vir

def get_A_number(mPert,cS,rSoft):
    A = G_grav*mPert/(cS*cS*rSoft)
    return A

def crossing_time(cS,rSoft):
    tCross = rSoft/cS
    return tCross

def I_Ostriker99_subsonic(Mach):
    if not (0 <= Mach <= 1):
        print("Warning: Mach > 1 in I_Ostriker99_subsonic.")
        raise ValueError("Mach > 1")
    #avoid singularity at Mach = 1
    delta = 0.05
    if Mach > 1 - delta and Mach <= 1:
        Mach = 1 - delta
    return 0.5*np.log((1+Mach)/(1-Mach)) - Mach

def I_Ostriker99_supersonic(Mach, rmin, Cs, t): 
    if not (Mach >= 1):
        print("Warning: Mach < 1 in I_Ostriker99_supersonic.")
        raise ValueError("Mach < 1")
    if (Mach -1 < rmin/t/Cs):
        print("Warning: Mach -1 < rmin/t/Cs in I_Ostriker99_supersonic.")
        raise ValueError("Mach -1 < rmin/t/Cs")
    #avoid singularity at Mach = 1
    delta = 0.05
    if Mach >= 1 and Mach < 1 + delta:
        Mach = 1 + delta
    #same as 0.5*np.log(1 - 1/Mach**2) + np.log(V*t/rmin)    
    return 0.5*np.log((1+Mach)/(Mach-1)) + np.log((Mach-1)/(rmin/t/Cs))


#output dn/dM in the unit of [(Mpc/h)^(-3) (Msun/h)^(-1)]
#input M in the unit of Msun/h
def HMF_Colossus(M, z):
    mfunc = mass_function.massFunction(M, z, model = 'press74', q_out = 'M2dndM')
    return mfunc/M**2*rho_m0*(1+z)**3/h_Hubble**2


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

#dN/ d lnx, x = m/M, input ln(x)
def Subhalo_Mass_Function_ln(ln_m_over_M,bestfitparams=None):
    global SHMF_model
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
    

    
#use Bosch2016 model to calculate the DF heating
def integrand_DFheating(ln_m_over_M, logM, z, *bestfitparams):
    if not bestfitparams:
        bestfitparams = None
    global SHMF_model
    
    I_DF = 1.0  #debug: I_DF not considered here
    
    M = 10**logM
    m_over_M = np.exp(ln_m_over_M)
    m = m_over_M * M  
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    DF_heating =  4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Vel_Virial_analytic(M/h_Hubble, z) *rho_g *I_DF
    if SHMF_model == 'BestFit':
        DF_heating *= Subhalo_Mass_Function_ln(ln_m_over_M,bestfitparams)
    else:
        DF_heating *= Subhalo_Mass_Function_ln(ln_m_over_M) 

    DF_heating *= HMF_Colossus(M,z) * np.log(10)*M   #convert from M to log10(M)
    
    return DF_heating



def plot_2D_histogram_analytic(output_filrname, z, ratio=1e-2, n_bins=50):
    """
    Create a 2D histogram of host and subhalo masses.
    
    Parameters:
    -----------
    ratio : float
        Maximum allowed ratio of subhalo to host halo mass (m/M)
    n_bins : int
        Number of bins for both axes
    """
    # Set up host halo mass bins (log scale)
    M_Jeans = get_M_Jeans(z)
    lgM_limits = [np.log10(10*M_Jeans), 10]
    lgM_bins = np.linspace(lgM_limits[0], lgM_limits[1], n_bins)
    M_values = 10**lgM_bins

    # Set up subhalo mass bins (log scale)
    lgm_min = np.log10(M_Jeans)  # Minimum subhalo mass (in Msun/h)
    lgm_max = 10  # Maximum subhalo mass (in Msun/h)
    lgm_bins = np.linspace(lgm_min, lgm_max, n_bins)
    
    # Create 2D histogram array
    hist = np.zeros((n_bins-1, n_bins-1))
    
    # Fill histogram
    for i in range(len(M_values)-1):
        M = M_values[i]
        
        # Calculate valid subhalo masses for this host mass
        m_min = M_Jeans
        m_max = M * ratio
        
        # Convert to log scale
        lgm_valid_min = np.log10(m_min)
        lgm_valid_max = np.log10(m_max)
        
        # Find valid bin indices
        valid_bins = np.where((lgm_bins[:-1] >= lgm_valid_min) & 
                            (lgm_bins[:-1] <= lgm_valid_max))[0]
        
        #host halo mass function [(Mpc/h)^(-3)/dlog10(M/(Msun/h))]
        dN_dlgM = HMF_Colossus(M,z) * np.log(10)*M
        
        # Fill histogram
        for j in valid_bins:
            m = 10**lgm_bins[j]
            dn_dm = Subhalo_Mass_Function_dN_dlgm(m, M)
            hist[i, j] = dn_dm * dN_dlgM
            
            
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Replace zeros with nan to avoid log(0) issues
    hist[hist == 0] = np.nan
    
    im = ax.imshow(hist.T, origin='lower', aspect='auto',
                   extent=[lgM_limits[0], lgM_limits[1], lgm_min, lgm_max],
                   cmap='viridis',
                   norm=colors.LogNorm(vmin=hist[~np.isnan(hist)].min(), 
                                     vmax=hist[~np.isnan(hist)].max()))
    
    ax.set_xlabel('log$_{10}$(M [M$_\odot$/h])', fontsize=14)
    ax.set_ylabel('log$_{10}$(m [M$_\odot$/h])', fontsize=14)
    ax.set_title(f'Valid Subhalo Masses (m/M ≤ {ratio:.0e})', fontsize=16)
    
    # Add colorbar with log scale and format it
    cbar = plt.colorbar(im, ax=ax, label=r'$\frac{d^2 N}{d \log_{10}M \, d \log_{10}m}$ [(Mpc/h)$^{-3}$]')
    
    # Custom formatting function for the colorbar ticks
    def formatter(x, p):
        return f"{x:.2e}"
    
    # Set locator for more tick marks and custom formatter
    cbar.ax.yaxis.set_major_locator(LogLocator(numticks=10))
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
    
    # Add reference lines
    ax.plot(lgM_bins, lgM_bins, 'r--', alpha=0.5, label='m = M')
    ax.plot(lgM_bins, lgM_bins + np.log10(ratio), 'g--', alpha=0.5, 
            label=f'm = M x {ratio:.2e}')
    ax.axhline(y=np.log10(M_Jeans), color='k', linestyle=':', alpha=0.5,
               label=f'M$_{{Jeans}}$ = {M_Jeans:.2e} M$_\odot$/h')
    
    ax.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(output_filrname, dpi=300)



def plot_Anumber_analytic(output_filename, z, ratio=1e-2, n_bins=50):
    """
    plot A number as a function of host halo mass and subhalo mass (analytic model).
    
    Parameters:
    -----------
    ratio : float
        Maximum allowed ratio of subhalo to host halo mass (m/M)
    n_bins : int
        Number of bins for both axes
    """
    # Set up host halo mass bins (log scale)
    M_Jeans = get_M_Jeans(z)
    lgM_limits = [np.log10(10*M_Jeans), 10]
    lgM_bins = np.linspace(lgM_limits[0], lgM_limits[1], n_bins)
    M_values = 10**lgM_bins

    # Set up subhalo mass bins (log scale)
    lgm_min = np.log10(M_Jeans)  # Minimum subhalo mass (in Msun/h)
    lgm_max = 10  # Maximum subhalo mass (in Msun/h)
    lgm_bins = np.linspace(lgm_min, lgm_max, n_bins)
    
    #create A number array
    A_number_array = np.zeros((n_bins-1, n_bins-1))
    
    # Fill histogram
    for i in range(len(M_values)-1):
        M = M_values[i]
        
        rho_halo = 200 * rho_m0*(1+z)**3 *Msun/Mpc**3
        R_host = (3*M*Msun/h_Hubble/(4*np.pi*rho_halo))**(1/3) #in m
        rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
        Tvir_host_analytic = Temperature_Virial_analytic(M/h_Hubble, z)
        Cs_host = np.sqrt(5.0/3.0 *kB *Tvir_host_analytic/(mu*mp)) #sound speed in m/s
        vel = Vel_Virial_analytic(M/h_Hubble, z)
        freefall_factor = np.sqrt(3 * np.pi / 32)
        t_dyn = freefall_factor/np.sqrt(G_grav*rho_halo)
        
        
        # Calculate valid subhalo masses for this host mass
        m_min = M_Jeans
        m_max = M * ratio
        
        # Convert to log scale
        lgm_valid_min = np.log10(m_min)
        lgm_valid_max = np.log10(m_max)
        
        # # Find valid bin indices
        # valid_bins = np.where((lgm_bins[:-1] >= lgm_valid_min) & 
        #                     (lgm_bins[:-1] <= lgm_valid_max))[0]
        
        for j in range(len(lgm_bins)-1):
            m = 10**lgm_bins[j]
            m_over_M = m/M
            
            r_sub = R_host * m_over_M**(1/3)
            
            tcross = r_sub/Cs_host
            Anumber = get_A_number(m*Msun/h_Hubble, Cs_host, r_sub)
            
            # print(f"m: {m:.5e}, M: {M:.5e}, r_sub: {r_sub:.5e}, R_host: {R_host:.5e}, Cs: {Cs_host:.5e}, tcross: {tcross:.5e}, A: {Anumber:.5e}")
            
            A_number_array[i, j] = Anumber
        
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Replace zeros with nan to avoid log(0) issues
    A_number_array[A_number_array == 0] = np.nan
    
    im = ax.imshow(A_number_array.T, origin='lower', aspect='auto',
                     extent=[lgM_limits[0], lgM_limits[1], lgm_min, lgm_max],
                        cmap='seismic',
                        norm=colors.LogNorm(vmin=A_number_array[~np.isnan(A_number_array)].min(),
                                            vmax=A_number_array[~np.isnan(A_number_array)].max()))
    
    ax.set_xlabel('log$_{10}$(M [M$_\odot$/h])', fontsize=14)
    ax.set_ylabel('log$_{10}$(m [M$_\odot$/h])', fontsize=14)
    ax.set_title(f'A Number (m/M ≤ {ratio:.0e})', fontsize=16)
    
    # Add colorbar with log scale and format it
    cbar = plt.colorbar(im, ax=ax, label='A Number')
    
    # Custom formatting function for the colorbar ticks
    def formatter(x, p):
        return f"{x:.2e}"
    
    # Set locator for more tick marks and custom formatter
    cbar.ax.yaxis.set_major_locator(LogLocator(numticks=10))
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(formatter))
    
    # Add reference lines
    ax.plot(lgM_bins, lgM_bins, 'r--', alpha=0.5, label='m = M')
    ax.plot(lgM_bins, lgM_bins + np.log10(ratio), 'g--', alpha=0.5,
            label=f'm = M x {ratio:.2e}')
    ax.axhline(y=np.log10(M_Jeans), color='k', linestyle=':', alpha=0.5,
                label=f'M$_{{Jeans}}$ = {M_Jeans:.2e} M$_\odot$/h')
    
    ax.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    
    #also plot Tvir as a function of M
    fig, ax = plt.subplots(figsize=(10, 8))
    Tvir_array = np.zeros(len(M_values)-1)
    for i in range(len(M_values)-1):
        M = M_values[i]
        Tvir_array[i] = Temperature_Virial_analytic(M/h_Hubble, z)
        
    ax.plot(lgM_bins[:-1], Tvir_array, 'b-', label='Tvirial')
    ax.set_xlabel('log$_{10}$(M [M$_\odot$/h])', fontsize=14)
    ax.set_ylabel('Tvirial [K]', fontsize=14)
    ax.set_yscale('log')
    ax.set_title(f'Virial Temperature', fontsize=16)
    ax.legend(fontsize=13)
    plt.tight_layout()
    
    plt.savefig(output_filename.replace('A_number', 'Tvir'), dpi=300)
    
            
            
            
def analytic_T_DF_singlevolume(ln_m_over_M, logM, z):
    M = 10**logM
    m_over_M = np.exp(ln_m_over_M)
    m = m_over_M * M
    rho_halo = 200 * rho_m0*(1+z)**3 *Msun/Mpc**3
    R_host = (3*M*Msun/h_Hubble/(4*np.pi*rho_halo))**(1/3) #in m
    r_sub = R_host * m_over_M**(1/3)
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    
    Tvir_host_analytic = Temperature_Virial_analytic(M/h_Hubble, z)
    Cs_host = np.sqrt(5.0/3.0 *kB *Tvir_host_analytic/(mu*mp)) #sound speed in m/s
    vel = Vel_Virial_analytic(M/h_Hubble, z)
    
    freefall_factor = np.sqrt(3 * np.pi / 32)
    t_dyn = freefall_factor/np.sqrt(G_grav*rho_halo)
    tcross = r_sub/Cs_host
    
    gas_metallicity = 0.0

    #compare the length scales: (r_sub, R_host, v*t, Cs*t)
    print(f"r_sub: {r_sub:.5e}")
    print(f"R_host: {R_host:.5e}")
    print(f"v*t_dyn: {vel*t_dyn:.5e}")
    print(f"Cs*t_dyn: {Cs_host*t_dyn:.5e}")
    print(f"v*t_cross: {vel*tcross:.5e}")
    print(f"Cs*t_cross: {Cs_host*tcross:.5e}")
    
    rho_g_wake = rho_g #debug: set wake density = background density
    nH = rho_g_wake/mp
    nH_cm3 = nH/1e6
    lognH = np.log10(nH_cm3)
    
    t_evaluate = t_dyn #debug: temporary set t = t_dyn
    wake_volume = 4/3*np.pi*(Cs_host*t_evaluate)**3
    wake_volume_cm3 = wake_volume*1e6
    
    I_DF = 1.0 #debug: I_DF not considered here
    DF_heating =  4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / vel *rho_g *I_DF
    normalized_heating = DF_heating/wake_volume_cm3/nH_cm3**2 #erg cm^3 s^-1
    evolve_cooling = True
    tfinal, T_DF_NonEq, cooling_rate_TDF_NonEq, cooling_rate_Tvir = get_Grackle_TDF_nonEquilibrium(Tvir_host_analytic, lognH, gas_metallicity, normalized_heating, z)
    return tfinal, T_DF_NonEq, cooling_rate_TDF_NonEq, cooling_rate_Tvir

def analytic_temperature_profile_subsonic(ln_m_over_M, logM, z, mach):
    #based on dfnumerical_subsonic.py profile and use Grackle to calculate the temperature profile
    M = 10**logM
    m_over_M = np.exp(ln_m_over_M)
    m = m_over_M * M
    rho_halo = 200 * rho_m0*(1+z)**3 *Msun/Mpc**3
    R_host = (3*M*Msun/h_Hubble/(4*np.pi*rho_halo))**(1/3) #in m
    r_sub = R_host * m_over_M**(1/3)
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    
    Tvir_host_analytic = Temperature_Virial_analytic(M/h_Hubble, z)
    Cs_host = np.sqrt(5.0/3.0 *kB *Tvir_host_analytic/(mu*mp)) #sound speed in m/s
    vel = Vel_Virial_analytic(M/h_Hubble, z)
    
    freefall_factor = np.sqrt(3 * np.pi / 32)
    t_dyn = freefall_factor/np.sqrt(G_grav*rho_halo)
    tcross = r_sub/Cs_host
    
    gas_metallicity = 0.0
    
    #debug: change to density profile later
    rho_g_wake = rho_g #debug: set wake density = background density
    nH = rho_g_wake/mp
    nH_cm3 = nH/1e6
    lognH = np.log10(nH_cm3)
    
    #debug: temporary set r_orbit = 0.5* R_host, and t_evaluate = 2 * r_orbit / vel
    r_orbit = 0.5*R_host
    t_evaluate = 2 * r_orbit / vel
    
    r_list = np.linspace(1.0*r_sub, 10*r_sub, 20)
    x_list = r_list/(Cs_host*t_evaluate)
    
    print("calculate I_df, volume, and heating rate ...")
    I_fdf_x = np.array([fdf_subsonic_from_xinn_to_xout(xout=xval, xinn=0.0, mach=mach) for xval in x_list])
    volume_x = np.array([volume_integral_subsonic(x, mach) for x in x_list])
    
    DF_heating =  4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / vel *rho_g
    
    #calculate heating rate in each shell
    dI = np.diff(I_fdf_x)
    dV = np.diff(volume_x)
    
    print("dI: ", dI)
    print("dV: [(Cs t)^3]", dV)
    
    dV_physical = dV * (Cs_host*t_evaluate)**3
    
    r_list_center = (r_list[:-1] + r_list[1:])/2
    DF_heating_profile = DF_heating * dI
    
    #now test density profile
    nH_cm3_profile = np.linspace(10*nH_cm3, 0.1*nH_cm3, len(r_list_center))
    #nH_cm3_profile = nH_cm3 * np.ones(len(r_list_center))
    
    normalized_heating_profile = DF_heating_profile/dV_physical/nH_cm3_profile**2 #erg cm^3 s^-1
    #use Grackle to calculate the temperature profile
    evolve_cooling = True
    
    print("calculate temperature profile ...")
    t_final_list = []
    T_DF_NonEq_profile = []
    for normalized_heating in normalized_heating_profile:
        
        
        tfinal, T_DF_NonEq, cooling_rate_TDF_NonEq, cooling_rate_Tvir = get_Grackle_TDF_nonEquilibrium(Tvir_host_analytic, lognH, gas_metallicity, normalized_heating, z)
        print(f"t: {tfinal}, T_DF_NonEq: {T_DF_NonEq}")
        t_final_list.append(tfinal)
        T_DF_NonEq_profile.append(T_DF_NonEq)
    
    #plot the temperature profile
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(r_list_center, T_DF_NonEq_profile, 'b-', label='T_DF_NonEq')
    ax.set_xlabel('r [m]', fontsize=14)
    ax.set_ylabel('T [K]', fontsize=14)
    ax.set_title(f'Temperature Profile (Mach = {mach}, lgM = {logM}), m/M = {m_over_M:.2e}', fontsize=16)
    ax.legend(fontsize=13)
    plt.tight_layout()
    plt.savefig(f"subsonic_T_profile_Mach{mach}_lgM{logM}_moverM{m_over_M:.2e}.png", dpi=300)
    
    


'''
def process_subhalo_NonEq(i, Sub_data, current_redshift, shm_name, loop_num, dtype_subhalowake,start_idx=None):
    
    try:
        logging.info(f"Subhalo {i} ...")
        Tvir = Sub_data['Tvir_host'][i]
        
        #use host halo gas metallicity because subhalo gas metallicity is often not available
        gas_metallicity_sub = Sub_data['gas_metallicity_sub'][i]
        gas_metallicity_sub /= 0.01295
        gas_metallicity_host = Sub_data['gas_metallicity_host'][i]
        gas_metallicity_host /= 0.01295
        #logging.info("Zsub: ", gas_metallicity_sub, "Zsun")
        #logging.info("Zhost: ", gas_metallicity_host, "Zsun")
        Mach_rel = Sub_data['Mach_rel'][i]
        vel_rel = Sub_data['vel_rel'][i] #m/s
        
        
        heating = Sub_data['DF_heating'][i]
        heating *= 1e7 #convert to erg/s
        
        rho_g = Sub_data['rho_g'][i]
        overdensity = 1.0
        rho_g_wake = rho_g*(1+overdensity)
        nH = rho_g_wake/mp
        nH_cm3 = nH/1e6
        lognH = np.log10(nH_cm3)
        
        #test which volume to use
        volume_wake_tdyn = Sub_data['Volume_wake'][i]
        volume_wake_tdyn_cm3 = volume_wake_tdyn*1e6
        volumetric_heating = heating/volume_wake_tdyn_cm3 #erg/s/cm^3
        normalized_heating = heating/volume_wake_tdyn_cm3/nH_cm3**2 #erg cm^3 s^-1
        Mgas_wake = volume_wake_tdyn*rho_g_wake
        specific_heating = heating/(Mgas_wake*1e3) #erg/s/g
        evolve_cooling = True
        
        #logging.info(f"\nTvir;: {Tvir} K, lognH: {lognH}")
        #logging.info("Normalized heating rate: ", normalized_heating, "erg cm^3/s")
        
        tfinal, T_DF_NonEq, cooling_rate_TDF_NonEq, cooling_rate_Tvir = get_Grackle_TDF_nonEquilibrium(Tvir, lognH, gas_metallicity_host, normalized_heating, current_redshift)
        
        #logging.info("Cooling rate (zero heating): ", cooling_rate_Tvir, "erg cm^3/s")
        
        result = ResultNonEq(i, Tvir, lognH, rho_g_wake, gas_metallicity_host, volume_wake_tdyn_cm3, heating, specific_heating, volumetric_heating, normalized_heating, tfinal, T_DF_NonEq, cooling_rate_Tvir, cooling_rate_TDF_NonEq, Mach_rel, vel_rel)
        
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shared_array = np.ndarray(loop_num, dtype=dtype_subhalowake, buffer=existing_shm.buf)
        
        i_memory = i
        if start_idx is not None:
            i_memory -= start_idx
        
        for field in result._fields:
            shared_array[i_memory][field] = getattr(result, field)
        
        #return subhalowake_info
        #shared_array[i] = result
        logging.info(f"Subhalo {i} Done")
        existing_shm.close()
        
'''

def plot_DF_heating_per_logM_comparison(volume,current_redshift,dark_matter_resolution,logM_bins,subhalo_DF_heating_hostmassbin,hosthalo_DF_heating_hostmassbin,filename,bestfitparams=None):
    global simulation_set, SHMF_model
    #convert to per volume per logM bin size
    logM_bin_width = logM_bins[1] - logM_bins[0]
    logM_bin_centers = (logM_bins[:-1] + logM_bins[1:]) / 2
    subhalo_DF_heating_hostmassbin_perV_perBinsize = subhalo_DF_heating_hostmassbin/logM_bin_width/volume
    hosthalo_DF_heating_hostmassbin_perV_perBinsize = hosthalo_DF_heating_hostmassbin/logM_bin_width/volume

    print("subhalo DF_heating max: ",max(subhalo_DF_heating_hostmassbin_perV_perBinsize))
    print("hosthalo DF_heating max: ",max(hosthalo_DF_heating_hostmassbin_perV_perBinsize))
 
    
       
    # Plot DF heating as a function of host halo mass bin
    fig = plt.figure(facecolor='white')
    plt.plot(logM_bin_centers, 1e7*subhalo_DF_heating_hostmassbin_perV_perBinsize,'r-',label=simulation_set+' subhalo')
    plt.plot(logM_bin_centers, 1e7*hosthalo_DF_heating_hostmassbin_perV_perBinsize,'b-',label=simulation_set+' host halo')

    #check contribution to heating (analytical result)
    z_value = current_redshift
    M_Jeans = get_M_Jeans(z_value)
    print("Jeans mass: ",M_Jeans)
    lgM_limits = [np.log10(M_Jeans), 15]  # Limits for log10(M [Msun/h])
    ln_m_over_M_limits = [np.log(1e-3), 0]  # Limits for m/M

    lgM_list = np.linspace(lgM_limits[0], lgM_limits[1],57)


    DF_heating_perlogM = []
    for logM in lgM_list:
        if SHMF_model == 'BestFit':
            result, error = quad(integrand_DFheating, ln_m_over_M_limits[0], ln_m_over_M_limits[1], args=(logM, z_value,*bestfitparams))
        else:
            result, error = quad(integrand_DFheating, ln_m_over_M_limits[0], ln_m_over_M_limits[1], args=(logM, z_value))



        if (abs(error) > 0.01 * abs(result)):
            print("Possible large integral error at z = %f, relative error = %f\n", z_value, error/result)

        DF_heating_perlogM.append(result)

    DF_heating_perlogM = np.array(DF_heating_perlogM)
    plt.plot(lgM_list,1e7*DF_heating_perlogM,'g-',label='subhalo analytic')
    plt.axvline(np.log10(100*dark_matter_resolution), color='black', linestyle='--')
    plt.legend()
    #plt.xlim([4,12])
    xlim_min = min(np.min(lgM_list),4)
    xlim_max = max(np.max(lgM_list),12)
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([1e37,1e43])
    plt.yscale('log')
    plt.ylabel(r'DF heating per logM [erg/s (Mpc/h)$^{-3}$]',fontsize=12)
    plt.xlabel('logM [Msun/h]',fontsize=12)
    plt.savefig(filename,dpi=300)


    #compare total heating rate
    subhalo_DF_heating_total_TNG = subhalo_DF_heating_hostmassbin.sum()/volume
    hosthalo_DF_heating_total_TNG = hosthalo_DF_heating_hostmassbin.sum()/volume

    print("subhalo_DF_heating_total_TNG: ",subhalo_DF_heating_total_TNG)
    print("hosthalo_DF_heating_total_TNG: ",hosthalo_DF_heating_total_TNG)

    if SHMF_model == 'BestFit':
        DF_heating_total_analytic, error = nquad(integrand_DFheating, [[ln_m_over_M_limits[0], ln_m_over_M_limits[1]], [lgM_limits[0], lgM_limits[1]]], args=(z_value,*bestfitparams))
    else:
        DF_heating_total_analytic, error = nquad(integrand_DFheating, [[ln_m_over_M_limits[0], ln_m_over_M_limits[1]], [lgM_limits[0], lgM_limits[1]]], args=(z_value,))


    if (abs(error) > 0.01 * abs(DF_heating_total_analytic)):
        print("possible large integral error at z = %f, relative error = %f\n",z_value,error/DF_heating_total_analytic)
    print("subhalo DF_heating_total_analytic: ",DF_heating_total_analytic)
    print("ratio: ",subhalo_DF_heating_total_TNG/DF_heating_total_analytic)
    

def plot_2D_histogram(Subhalo_Properties,output_dir):
    print(Subhalo_Properties)
    print(Subhalo_Properties.shape)
    
    subhalo_masses = Subhalo_Properties['subhalo_mass']
    host_Mtot = Subhalo_Properties['host_Mtot']
    host_M200 = Subhalo_Properties['host_M200']
    halfmass_radius = Subhalo_Properties['halfmass_radius']
    rsoft = Subhalo_Properties['rsoft']
    host_R200 = Subhalo_Properties['host_R200']
    Mach_rel = Subhalo_Properties['mach_number']
    t_dyn = Subhalo_Properties['dynamical_time']
    tcross = Subhalo_Properties['crossing_time']
    Anumber = Subhalo_Properties['A_number']
    
    #2D histogram    
    
    #Mtot vs Msub
    fig = plt.figure(figsize=(8,6),facecolor='w')
    plt.hist2d( np.log10(host_Mtot),np.log10(subhalo_masses),bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(tot M$_{host}$ [Msun/h])',fontsize=14)
    plt.ylabel(r'log$_{10}$(m$_{sub}$ [Msun/h])',fontsize=14)
    plt.savefig(f'{output_dir}/Mtot_Msub.png',dpi=300)
    plt.close()
    
    #M200 vs Msub
    fig = plt.figure(figsize=(8,6),facecolor='w')
    plt.hist2d( np.log10(host_M200),np.log10(subhalo_masses),bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(M$_{200}$ [Msun/h])',fontsize=14)
    plt.ylabel(r'log$_{10}$(m$_{sub}$ [Msun/h])',fontsize=14)
    plt.savefig(f'{output_dir}/M200_Msub.png',dpi=300)
    plt.close()
    
    #R200 vs sub halfmass radius
    fig = plt.figure(figsize=(8,6),facecolor='w')
    plt.hist2d( np.log10(host_R200),np.log10(halfmass_radius),bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(R$_{200}$ [m])',fontsize=14)
    plt.ylabel(r'log$_{10}$(r$_{sub, halfmass}$ [m])',fontsize=14)
    plt.savefig(f'{output_dir}/R200_rsubhalfmass.png',dpi=300)
    plt.close()
    
    #R200 vs rsoft
    fig = plt.figure(figsize=(8,6),facecolor='w')
    plt.hist2d( np.log10(host_R200),np.log10(rsoft),bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(R$_{200}$ [m])',fontsize=14)
    plt.ylabel(r'log$_{10}$($r_{soft} = r_{sub,VmaxRad}$ [m])',fontsize=14)
    plt.savefig(f'{output_dir}/R200_rsoft.png',dpi=300)
    plt.close()
    
    #tdyn vs tcross
    fig = plt.figure(figsize=(8,6),facecolor='w')
    plt.hist2d( np.log10(t_dyn),np.log10(tcross),bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(t$_{dyn, host}$ [s])',fontsize=14)
    plt.ylabel(r'log$_{10}$(t$_{cross}$ [s])',fontsize=14)
    plt.savefig(f'{output_dir}/tdyn_tcross.png',dpi=300)
    plt.close()
    
    #M200 vs Mach number
    fig = plt.figure(figsize=(8,6),facecolor='w')
    plt.hist2d( np.log10(host_M200),Mach_rel,bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(M$_{200}$ [Msun/h])',fontsize=14)
    plt.ylabel(r'Mach number',fontsize=14)
    plt.savefig(f'{output_dir}/M200_Mach.png',dpi=300)
    plt.close()
    
    #M200 vs A number
    fig = plt.figure(figsize=(8,6),facecolor='w')
    plt.hist2d( np.log10(host_M200),np.log10(Anumber),bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(M$_{200}$ [Msun/h])',fontsize=14)
    plt.ylabel(r'log$_{10}$(A number)',fontsize=14)
    plt.savefig(f'{output_dir}/M200_Anumber.png',dpi=300)
    plt.close()
    


def TNG_model():
    
    simulation_set = 'TNG50-1'
    

    if simulation_set == 'TNG50-1':
        gas_resolution = 4.5e5 * h_Hubble #convert from Msun to Msun/h
        dark_matter_resolution = 8.5e4 * h_Hubble
    elif simulation_set == 'TNG100-1':
        gas_resolution = 1.4e6 * h_Hubble  
        dark_matter_resolution = 7.5e6 * h_Hubble  
    elif simulation_set == 'TNG300-1':
        gas_resolution = 1.1e7 * h_Hubble
        dark_matter_resolution = 5.9e7 * h_Hubble
    
    basePath = '/home/zwu/21cm_project/TNG_data/'+simulation_set+'/output'
    snapNum = 2
    output_dir = '/home/zwu/21cm_project/WakeVolume/results/'+simulation_set+'/'
    output_dir += f'snap_{snapNum}/'
    #mkdir for this snapshot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("output_dir: ",output_dir)
    
    print("loading header ...")
    header = il.groupcat.loadHeader(basePath, snapNum)
    print("loading halos ...")
    halos = il.groupcat.loadHalos(basePath, snapNum, fields=['GroupFirstSub', 'GroupNsubs', 'GroupPos', 'GroupMass', 'GroupMassType','Group_M_Crit200','Group_R_Crit200','Group_M_Crit500','Group_R_Crit500','GroupVel','GroupGasMetallicity'])
    print("loading subhalos ...")
    subhalos = il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloMass', 'SubhaloPos', 'SubhaloVel', 'SubhaloHalfmassRad','SubhaloVmaxRad','SubhaloGrNr', 'SubhaloMassType','SubhaloGasMetallicity'])
    
    print("\n")
    print("redshift: ", header['Redshift'])
    current_redshift = header['Redshift']
    scale_factor = header['Time']
    print(scale_factor)
    print("box size: ", header['BoxSize']," ckpc/h")  
    volume = (header['BoxSize']/1e3 * scale_factor) **3  # (Mpc/h)^3

    print(header.keys())
    print(halos.keys())
    print(subhalos.keys())

    print("number of halos: ", halos['count'])
    print("number of subhalos: ", subhalos['count'])
    
    print(halos['Group_M_Crit200'])
    print(halos['GroupMass'])
    

        
    #---------------------------------------------------
    #Main Loop over each host halo and its subhalos
    #---------------------------------------------------
    Subhalo_Properties = []
    Output_Subhalo_Info = []
    Output_Hosthalo_Info = []
    vel_host_list = []
    vel_host_gas_list = []
    T_DFheating_list = []
    DF_thermal_energy_list = []
    Mach_rel_list = []
    H_R0_ratio_list = []
    R0cube_Coeff_list = []
    Wake_Volume_filling_factor_list = []
    I_DF_list = [] 
    
    
    #select halos with mass > 100*dark_matter_resolution
    mask_groupmass = halos['GroupMass']*1e10 > 100*dark_matter_resolution #Msun/h
    mask_M200 = halos['Group_M_Crit200']*1e10 > 0
    mask_R200 = halos['Group_R_Crit200'] > 0
    mask_subhalo = halos['GroupNsubs'] > 1 #at least 1 subhalo besides the central subhalo
    num_unresolved = np.sum(~mask_groupmass)
    num_M200_zero = np.sum(~mask_M200)
    num_R200_zero = np.sum(~mask_R200)
    num_nosubhalo = np.sum(~mask_subhalo)
    
    print(f"number of unresolved halos: {num_unresolved}")
    print(f"number of halos with M_crit200 = 0: {num_M200_zero}")
    print(f"number of halos with R_crit200 = 0: {num_R200_zero}")
    print(f"number of halos with no subhalo: {num_nosubhalo}")
    #print(halos['GroupMass'][~mask_M200])

    
    mask = mask_groupmass & mask_M200 & mask_R200 & mask_subhalo
    N_selected = np.sum(mask)
    print("number of selected halos: ", N_selected)
    index_selected = np.where(mask)[0]
    #print("index_selected: ", index_selected)
    #print(len(index_selected))
    
    
    for host in index_selected:
        M = halos['GroupMass'][host]*1e10  #Msun/h
     
        M_crit200 = halos['Group_M_Crit200'][host]*1e10 #Msun/h
        M_gas = halos['GroupMassType'][host][0]*1e10  #Msun/h
        R_crit200 = halos['Group_R_Crit200'][host]  #ckpc/h
        R_crit200 = R_crit200/1e3 * scale_factor / h_Hubble  #Mpc
        R_crit200_m = R_crit200 * Mpc
        
        rho_g_analytic = rho_b0*(1+current_redshift)**3 *Msun/Mpc**3
        rho_m_analytic = rho_m0*(1+current_redshift)**3 *Msun/Mpc**3
        rho_g_analytic_200 = 200 *rho_g_analytic
        rho_m_analytic_200 = 200 *rho_m_analytic
        group_vel = halos['GroupVel'][host]*1e3/scale_factor  #km/s/a to m/s
        #vel_analytic = Vel_Virial_analytic(M_crit200/h_Hubble, current_redshift)
        vel_numerical = Vel_Virial_numerical(M_crit200/h_Hubble, R_crit200)
        #print(f"vel_analytic: {vel_analytic}, vel_numerical: {vel_numerical}")
        
        gas_metallicity_host = halos['GroupGasMetallicity'][host]

        
        vel_host = np.sqrt(np.sum(group_vel**2))
        vel_host_list.append(vel_host)
        rho_halo = (M_crit200*Msun/h_Hubble)/(4/3*np.pi*R_crit200_m**3)
        
        #Tvir_host_analytic = Temperature_Virial_analytic(M_crit200/h_Hubble, current_redshift)
        Tvir_host = Temperature_Virial_numerical(M_crit200/h_Hubble, R_crit200)
        #print(f"Tvir_host: {Tvir_host_analitic}, Tvir_host_numerical: {Tvir_host}")
        
        Cs_host = np.sqrt(5.0/3.0 *kB *Tvir_host/(mu*mp)) #sound speed in m/s
        freefall_factor = np.sqrt(3 * np.pi / 32)
        t_dyn = freefall_factor/np.sqrt(G_grav*rho_halo)
        
        # Get the subhalos of this host
        first_sub = int(halos['GroupFirstSub'][host])
        num_subs = int(halos['GroupNsubs'][host])
        # subhalos_of_host = [(j, subhalos['SubhaloMass'][j]*1e10,
        #                         subhalos['SubhaloHalfmassRad'][j]/1e3 *scale_factor/h_Hubble*Mpc)
        #                     for j in range(first_sub, first_sub + num_subs)]  #mass to Msun/h; radius ckpc/h to m

        #check if the first subhalo is the most massive one
        # subhalo_masses = [subhalos['SubhaloMass'][j]*1e10 for j in range(first_sub, first_sub + num_subs)]
        # max_subhalo_index = first_sub + subhalo_masses.index(max(subhalo_masses))
        
        #previous method: sort and preclude the most massive subhalo
        # subhalos_of_host.sort(key=lambda x: x[1])
        # maxsub_index = subhalos_of_host[-1][0]
        # subhalos_of_host = subhalos_of_host[:-1]
        
        #new method: preclude subhalo 0 (the central subhalo, which is usually the most massive one)

        # Loop over each subhalo
        for j in range(first_sub+1, first_sub + num_subs):
            subhalo_mass = subhalos['SubhaloMass'][j]*1e10
            subhalo_halfrad= subhalos['SubhaloHalfmassRad'][j]/1e3 *scale_factor/h_Hubble*Mpc
            m = subhalo_mass   
            #mass to Msun/h; radius ckpc/h to m

            #exclude small subhalos not resolved
            if(m < 100*dark_matter_resolution):
                continue
            #exclude possible incorrect subhalos
            if (m/M >= 1):
                print("Warning: m/M > 1")
                continue
            
            subhalo_rsoft = subhalos['SubhaloVmaxRad'][j]/1e3 *scale_factor/h_Hubble*Mpc #ckpc/h to m
            tcross = crossing_time(Cs_host,subhalo_rsoft)
            Anumber = get_A_number((m*Msun/h_Hubble),Cs_host,subhalo_rsoft)
            
            subhalo_vel = subhalos['SubhaloVel'][j]*1e3  #km/s to m/s (default unit for subhalo vel is km/s, see TNG data specification)
            
            vel = np.sqrt(np.sum((group_vel - subhalo_vel)**2))
            Mach_rel = vel/Cs_host
            gas_metallicity_sub = subhalos['SubhaloGasMetallicity'][j]
            
            I_DF = 0.0
            if Mach_rel <= 1:
                I_DF = I_Ostriker99_subsonic(Mach_rel)
                t_evaluate = t_dyn
            else:
                rmin = 2.25*subhalo_halfrad  #Sánchez-Salcedo & Brandenburg 1999
                t_evaluate = t_dyn
                #check if rmin and t_evaluate satisfy the condition in Ostriker99, i.e. V*t - Cs*t > rmin
                if (vel*t_evaluate - Cs_host*t_evaluate <= rmin):
                    print("Warning: vel*t - Cs*t <= rmin")
                    #reset t_evaluate
                    t_evaluate = rmin/(vel-Cs_host)*1.1
                
                I_DF = I_Ostriker99_supersonic(Mach_rel, rmin, Cs_host, t_evaluate)
                
            #compare Cs*t with host halo radius
            if (Cs_host*t_evaluate > R_crit200_m):
                print("Warning: Cs*t > R_crit200")
                print(f"Cs_host*t_evaluate: {Cs_host*t_evaluate}, R_crit200: {R_crit200_m}")
                print(f"Cs_host: {Cs_host}, t_evaluate: {t_evaluate/Myr} Myr")
                #debug: to be checked
            
            
            DF_heating = I_DF* 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / vel *rho_g_analytic_200 
            
            dtype_subhalo = np.dtype([
                ('subhalo_mass', np.float64),
                ('host_Mtot', np.float64),
                ('host_M200', np.float64),
                ('halfmass_radius', np.float64),
                ('rsoft', np.float64),
                ('host_R200', np.float64),
                ('mach_number', np.float64),
                ('dynamical_time', np.float64),
                ('crossing_time', np.float64),
                ('A_number', np.float64)
            ])
                            
            properties = np.array([(m, M, M_crit200, subhalo_halfrad, subhalo_rsoft, R_crit200_m, Mach_rel, t_dyn, tcross, Anumber)], dtype=dtype_subhalo)
            
            Subhalo_Properties.append(properties)
    
    #---------------------------------------------------
    Subhalo_Properties = np.concatenate(Subhalo_Properties)
    #plot_2D_histogram(Subhalo_Properties,output_dir)

def integrate_SHMF_for_heating(lgM_list,ln_m_over_M_min,ln_m_over_M_max,z_value,bestfitparams=None):
    global SHMF_model
    DF_heating_perlogM = []
    for i, logM in enumerate(lgM_list):
        if SHMF_model == 'BestFit':
            result, error = quad(integrand_DFheating, ln_m_over_M_min[i],ln_m_over_M_max[i],args=(logM, z_value,*bestfitparams))
        else:
            result, error = quad(integrand_DFheating, ln_m_over_M_min[i],ln_m_over_M_max[i],args=(logM, z_value))

        if (abs(error) > 0.01 * abs(result)):
            print("Possible large integral error at z = %f, relative error = %f\n", z_value, error/result)

        DF_heating_perlogM.append(result)
    return np.array(DF_heating_perlogM)
    
    

        
    
def Analytic_model():
    global SHMF_model
    simulation_set = 'TNG50-1'
    TNG50_redshift_list = [20.05,14.99,11.98,10.98,10.00,9.39,9.00,8.45,8.01]
    snapNum = 4
    current_redshift = TNG50_redshift_list[snapNum]
    output_dir = '/home/zwu/21cm_project/WakeVolume/results/'+simulation_set+'/'
    #output_dir = '/Users/wuzhenyu/Desktop/PhD/21cm_project/WakeVolume/results/'+simulation_set+'/'
    output_dir += f'snap_{snapNum}/'
    #mkdir for this snapshot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("output_dir: ",output_dir)
    output_filename = output_dir+f"DF_heating_perlogM_analytic_snap{snapNum}_z{current_redshift:.2f}.png"
    if SHMF_model == 'BestFit':

        #read bestfit parameters
        params_dir = f"/home/zwu/21cm_project/compare_TNG/results/{simulation_set}/snap_{snapNum}/"
        SHMF_fit_params_file = params_dir+f"Average_Mratio_dN_dlogMratio_snap{snapNum}_z{current_redshift:.2f}_fit_parameters.txt"
        bestfitparams = read_SHMF_fit_parameters(SHMF_fit_params_file)
        print(bestfitparams)


       
    # Plot DF heating as a function of host halo mass bin

    #check contribution to heating (analytical result)
    z_value = current_redshift
    M_Jeans = get_M_Jeans(z_value)
    print("Jeans mass: ",M_Jeans)
    lgM_limits = [np.log10(10*M_Jeans), 12]  # Limits for log10(M [Msun/h])

    lgM_list = np.linspace(lgM_limits[0], lgM_limits[1],50)
    #set Jeans mass as min subhalo mass
    ln_m_over_M_min = np.array([np.log(M_Jeans/10**lgM_list[j]) for j in range(len(lgM_list))])
    #ln_m_over_M_max = 0.0*np.ones_like(ln_m_over_M_min)
    ln_m_over_M_max_0 = np.log(1)*np.ones_like(ln_m_over_M_min)
    ln_m_over_M_max_2 = np.log(1e-2)*np.ones_like(ln_m_over_M_min)
    ln_m_over_M_max_3 = np.log(1e-3)*np.ones_like(ln_m_over_M_min)
    ln_m_over_M_max_4 = np.log(1e-4)*np.ones_like(ln_m_over_M_min)
    
    DF_heating_perlogM_0 = integrate_SHMF_for_heating(lgM_list,ln_m_over_M_min,ln_m_over_M_max_0,z_value,bestfitparams=None)
    DF_heating_perlogM_2 = integrate_SHMF_for_heating(lgM_list,ln_m_over_M_min,ln_m_over_M_max_2,z_value,bestfitparams=None)
    DF_heating_perlogM_3 = integrate_SHMF_for_heating(lgM_list,ln_m_over_M_min,ln_m_over_M_max_3,z_value,bestfitparams=None)
    DF_heating_perlogM_4 = integrate_SHMF_for_heating(lgM_list,ln_m_over_M_min,ln_m_over_M_max_4,z_value,bestfitparams=None)
    
    fig = plt.figure(facecolor='white')
    plt.plot(lgM_list,1e7*DF_heating_perlogM_0,'g-',label=r'$m/M \in [m_J/M,1]$')
    plt.plot(lgM_list,1e7*DF_heating_perlogM_2,'r-',label=r'$m/M \in [m_J/M,10^{-2}]$')
    plt.plot(lgM_list,1e7*DF_heating_perlogM_3,'b-',label=r'$m/M \in [m_J/M,10^{-3}]$')
    plt.plot(lgM_list,1e7*DF_heating_perlogM_4,'y-',label=r'$m/M \in [m_J/M,10^{-4}]$')
    plt.axvline(np.log10(M_Jeans), color='black', linestyle='--',label='Jeans mass')
    plt.legend()
    xlim_min = min(np.min(lgM_list),3)
    xlim_max = max(np.max(lgM_list),12)
    plt.xlim([xlim_min,xlim_max])
    plt.ylim([1e33,1e42])
    plt.yscale('log')
    plt.ylabel(r'DF heating per logM [erg/s (Mpc/h)$^{-3}$]',fontsize=12)
    plt.xlabel('logM [Msun/h]',fontsize=12)
    plt.savefig(output_filename,dpi=300)
    print("output_filename: ",output_filename)
    
    #calculate number density of subhalos in each host halo mass bin
    # histogram2D_filename = output_dir+f"2D_histogram_snap{snapNum}_z{current_redshift:.2f}.png"
    # plot_2D_histogram_analytic(histogram2D_filename,z_value,1e-3,50)
    
    # Anumber_filename = output_dir+f"A_number_snap{snapNum}_z{current_redshift:.2f}.png"
    # plot_Anumber_analytic(Anumber_filename,z_value,1e-3,50)
    
    
    #now test temperature profile
    mach_test = 0.5
    
    ln_m_over_M = np.log(1e-3)
    logM = 8.0
    analytic_temperature_profile_subsonic(ln_m_over_M, logM, z_value, mach_test)
    
    
    
    
    #test temperature T_DF_NonEq
    # logM_test = np.linspace(lgM_limits[0], lgM_limits[1],10)
    # ln_m_over_M_min_test = np.array([np.log(M_Jeans/10**logM_test[j]) for j in range(len(logM_test))])
    # ln_m_over_M_max_test = np.log(1e-3)*np.ones_like(ln_m_over_M_min_test)
    # for i, logM in enumerate(logM_test):
    #     ln_m_over_M_list = np.linspace(ln_m_over_M_min_test[i],ln_m_over_M_max_test[i],10)
    #     for j, ln_m_over_M in enumerate(ln_m_over_M_list):
    #         tfinal, T_DF_NonEq, cooling_rate_TDF_NonEq, cooling_rate_Tvir = analytic_T_DF_singlevolume(ln_m_over_M, logM, z_value)
    #         m_over_M = np.exp(ln_m_over_M)
    #         print(f"\nlogM: {logM}, m_over_M: {m_over_M}, T_DF_NonEq: {T_DF_NonEq}")
    #         print(f"tfinal: {tfinal/Myr} Myr")
        
        

    
SHMF_model = 'Bosch2016'
if __name__ == "__main__":
    #TNG_model()
    Analytic_model()