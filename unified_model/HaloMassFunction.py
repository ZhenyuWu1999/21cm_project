from colossus.lss import mass_function
from hmf import MassFunction
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from TNGDataHandler import get_simulation_resolution
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib
from scipy.special import gamma
import matplotlib.ticker as ticker


from physical_constants import *
from Config import simulation_set, hmf_ratio_params, \
p_evolved, p_unevolved, alpha_z_params, lgA_z_params, omega_z_params, lnbeta_z_params
from HaloProperties import Vel_Virial_analytic_oldversion

#output dn/dM in the unit of [(Mpc/h)^(-3) (Msun/h)^(-1)]
#input M in the unit of Msun/h
def HMF_Colossus(M, z, model, mdef = None):
    if model == 'press74' or model == 'sheth99':
        mfunc = mass_function.massFunction(M, z, model = model, q_out = 'M2dndM', mdef = 'fof')
    elif model == 'reed07':
        ps_path = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/analysis/input_spectrum_PLANCK15.txt'
        #mfunc = mass_function.massFunction(M, z, model = model, ps_args={'model': 'test', 'path': ps_path}, q_out = 'M2dndM')
        mfunc = mass_function.massFunction(M, z, model = model, ps_args={'model': 'eisenstein98'}, q_out = 'M2dndM')
    elif model == 'tinker08':
        mfunc = mass_function.massFunction(M, z, model = model, mdef=mdef, q_out = 'M2dndM')
    else:
        print('Error: model not supported')
        return -999
    return mfunc/M**2*rho_m0*(1+z)**3/h_Hubble**2

def HMF_py_dndlog10m(lgMmin, lgMmax, dlog10m, z, hmf_model, mdef_model, mdef_params):
    '''
    Parameters:
    lgMmin: log10(Mmin [Msun/h])
    lgMmax: log10(Mmax [Msun/h])
    dlog10m: log10(M) bin width
    z: redshift
    hmf_model: the model of HMF, e.g. 'Tinker08'
    mdef_model: the model of mass definition, e.g. "SOCritical"
    mdef_params: the parameters of mass definition, e.g. {'overdensity': 200}
    Returns:
    dndlog10m: the HMF in the unit of [(cMpc/h)^(-3) dex^{-1}]
    '''
    my_mf = MassFunction(hmf_model = hmf_model, cosmo_model = 'Planck15')
    my_mf.update(
        z = z,
        Mmin = lgMmin,
        Mmax = lgMmax,
        dlog10m = dlog10m,
        mdef_model = mdef_model,
        mdef_params = mdef_params
    )
    return my_mf.m, my_mf.dndlog10m

    

def plot_hmf(halos, index_selected, current_redshift, dark_matter_resolution, simulation_volume, hmf_filename):
    '''
    Parameters:
    halos: the full halo catalog in TNG
    current_redshift: the redshift of the snapshot
    dark_matter_resolution: the resolution of the dark matter particles
    simulation_volume: the simulation_volume of the simulation box in unit of (Mpc/h)^3
    hmf_filename: the output filename of the plot

    '''
    scale_factor = 1.0/(1.+current_redshift)
    comoving_factor = scale_factor**3
    #plot HMF (halo mass function)
    M_all = halos['GroupMass']*1e10   #unit: 1e10 Msun/h
    # M_all = halos['Group_M_Crit200']*1e10  #unit: Msun/h
    selected_M_all = M_all[index_selected]
    max_M = np.max(M_all)

    # Create a histogram (logarithmic bins and logarithmic mass)
    bins = np.logspace(np.log10(min(M_all[M_all > 0])), np.log10(max(M_all)), num=50)
    hist, bin_edges = np.histogram(M_all, bins=bins)
    hist_selected, _ = np.histogram(selected_M_all, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Convert counts to number density
    log_bin_widths = np.diff(np.log10(bins))
    number_density = hist / simulation_volume / log_bin_widths
    number_density_selected = hist_selected / simulation_volume / log_bin_widths

    # Plot the mass function
    fig = plt.figure(facecolor='white')
    ax = fig.gca()
 
    logM_limits = [6, np.log10(1.1*max_M)]  # Limits for log10(M [Msun/h])
    HMF_lgM_press74 = []
    HMF_lgM_sheth99 = []
    # HMF_lgM_tinker08 = []

    dlog10m = (logM_limits[1] - logM_limits[0]) / 60
    logM_list = np.arange(logM_limits[0], logM_limits[1], dlog10m)
    #plot analytical HMF
    for logM in logM_list:
        M = 10**(logM)
        HMF_lgM_press74.append(HMF_Colossus(10**logM, current_redshift, 'press74')* np.log(10)*M)  
        HMF_lgM_sheth99.append(HMF_Colossus(10**logM, current_redshift, 'sheth99')* np.log(10)*M)
        # HMF_lgM_tinker08.append(HMF_Colossus(10**logM, current_redshift, 'tinker08', mdef = '200c')* np.log(10)*M)
    HMF_lgM_press74 = np.array(HMF_lgM_press74)
    HMF_lgM_sheth99 = np.array(HMF_lgM_sheth99)
    # HMF_lgM_tinker08 = np.array(HMF_lgM_tinker08)
    # _, HMF_lgM_tinker08_test = HMF_py_dndlog10m(logM_limits[0], logM_limits[1], dlog10m, current_redshift, 'Tinker08', mdef_model = 'SOCritical', mdef_params = {'overdensity': 200})

    #plot the dark matter resolution and TNG HMF
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(100*dark_matter_resolution, color='black', linestyle='--')
    plt.scatter(bin_centers, number_density*comoving_factor, c='none', edgecolor='blue', marker='o', label='All TNG halos')
    plt.scatter(bin_centers, number_density_selected*comoving_factor, c='none', edgecolor='green', marker='^',label='Selected TNG halos')

    plt.plot(10**(logM_list),HMF_lgM_press74*comoving_factor, color='k',linestyle='-',label='Press74')
    plt.plot(10**(logM_list),HMF_lgM_sheth99*comoving_factor, color='red',linestyle='-',label='Sheth99')
    # plt.plot(10**(logM_list),HMF_lgM_tinker08*comoving_factor, color='blue',linestyle='-',label='Tinker08')
    # plt.plot(10**(logM_list),HMF_lgM_tinker08_test,color='orange',linestyle='-',label='Tinker08')
    plt.legend(fontsize=13)
    
    plt.xlabel(r'Mass [$\mathrm{M}_{\odot}/\mathrm{h}$]', fontsize=14)
    plt.ylabel(r'$\frac{\text{dN}}{\text{ d\lg M}}$ [$(\text{cMpc/h})^{-3}$]',fontsize=14)
    ax.tick_params(direction='in', which='both', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(hmf_filename,dpi=300)
    
    #also save the data
    hmf_data_filename = hmf_filename.replace('.png','.txt')
    with open(hmf_data_filename, 'w') as f:
        f.write('bin_edge_left, bin_edge_right, bin_center, number_density_all, number_density_selected (dN/dlgM [(Mpc/h)^(-3)])\n')
        for i in range(len(bin_centers)):
            f.write(str(bin_edges[i]) + ' ' + str(bin_edges[i+1]) + ' ' + str(bin_centers[i]) + ' ' + str(number_density[i]) + ' ' + str(number_density_selected[i]) + '\n')

def fitFunc_hmf_ratio_2D(input_data, a, b, c, d, e, f):
    # Unpack the data
    lg_mass, redshift = input_data
    # q and p as functions of scale factor
    q = a + b/(1.0+redshift) + c*(1.0+redshift)
    p = d + e/(1.0+redshift) + f*(1.0+redshift)
    # Calculate the sigmoid
    return 1.0 / (1.0 + p**(-lg_mass + q))

def plot_hmf_redshift_evolution(snapNums, redshifts, dark_matter_resolution):
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/'
    output_dir = os.path.join(base_dir, 'analysis')
    output_filename = os.path.join(output_dir, 'HMF_redshift_evolution.png')
    hmf_filename_list = [os.path.join(base_dir, f'snap_{snapNum}', 'analysis', f'HMF_snap_{snapNum}.txt') for snapNum in snapNums]
    #bin_edge_left, bin_edge_right, bin_center, number_density_all, number_density_selected (dN/dlgM [(Mpc/h)^(-3)])
    hmf_data_list = []; scale_factor_list = []
    for redshift, hmf_filename in zip(redshifts, hmf_filename_list):
        with open(hmf_filename, 'r') as f:
            hmf_data = np.loadtxt(f, delimiter=' ', skiprows=1)
            hmf_data_list.append(hmf_data)
            scale_factor = 1.0/(1.+redshift)
            scale_factor_list.append(scale_factor)
    redshifts = np.array(redshifts); scale_factor_list = np.array(scale_factor_list)
    #plot HMF redshift evolution
    fig = plt.figure(figsize=(8,6), facecolor='white')
    ax = fig.gca()
    plt.yscale('log')
    plt.xscale('log')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(hmf_data_list)))
    labels = [f'z={redshift}' for redshift in redshifts]

    for i in range(len(hmf_data_list)):
        hmf_data = hmf_data_list[i]
        redshift = redshifts[i]
        scale_factor = scale_factor_list[i]
        comoving_factor = scale_factor**3

        bin_centers = hmf_data[:,2]
        number_density_all = hmf_data[:,3]
        number_density_selected = hmf_data[:,4]
        # lgMmin = np.log10(min(bin_centers))
        # lgMmax = np.log10(max(bin_centers))
        # dlog10m = (lgMmax - lgMmin) / (len(bin_centers) - 1)
        # M_list, HMF_Tinker08 = HMF_py_dndlog10m(lgMmin, lgMmax, dlog10m, redshift, 'Tinker08', mdef_model = 'SOCritical', mdef_params = {'overdensity': 200})   
        #sheth99
        lgM = np.log10(bin_centers)
        HMF_lgM_sheth99 = np.array([HMF_Colossus(10**lgM[j], redshift, 'sheth99')* np.log(10)*10**lgM[j] for j in range(len(lgM))])
        
        # plt.plot(M_list, HMF_Tinker08, color=colors[i],linestyle='-',label=labels[i])
        plt.plot(10**lgM,HMF_lgM_sheth99*comoving_factor, color=colors[i],linestyle='-',label=labels[i])
        plt.scatter(bin_centers, number_density_all*comoving_factor, c='none', edgecolor=colors[i], marker='o')
        plt.scatter(bin_centers, number_density_selected*comoving_factor, c='none', edgecolor=colors[i], marker='^')
    
    plt.axvline(100*dark_matter_resolution, color='black', linestyle='--')
    plt.legend(fontsize=11)
    plt.xlabel(r'Mass [$\mathrm{M}_{\odot}/\mathrm{h}$]', fontsize=14)
    plt.grid()
    plt.ylabel(r'$\frac{\text{dN}}{\text{ d\lg M}}$ [$(\text{cMpc/h})^{-3} \text{dex}^{-1}$]',fontsize=14)
    ax.tick_params(direction='in', which='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_filename,dpi=300)


    #then plot the ratio between selected and all halos
    output_ratio_fit_filename = os.path.join(output_dir, 'HMF_redshift_evolution_ratio_fit.png')
    all_lg_masses = []
    all_redshifts = []
    all_ratios = []
        
    for i in range(len(hmf_data_list)):
        hmf_data = hmf_data_list[i]
        redshift = redshifts[i]
        bin_centers = hmf_data[:,2]
        number_density_all = hmf_data[:,3]
        number_density_selected = hmf_data[:,4]
        # Avoid zero division and avoid resolution effect
        mask = (number_density_all > 0) & (number_density_selected > 0)
        ratio_masked = number_density_selected[mask]/number_density_all[mask]
        bin_centers_masked = bin_centers[mask]
        # Add to our collections
        all_lg_masses.extend(np.log10(bin_centers_masked))
        all_redshifts.extend([redshift] * len(bin_centers_masked))
        all_ratios.extend(ratio_masked)

    all_lg_masses = np.array(all_lg_masses)
    all_redshifts = np.array(all_redshifts)
    all_ratios = np.array(all_ratios)
    # Initial guess for parameters [a, b, c, d, e, f]
    initial_guess = [10.0, -0.2, 0.0, 2.0, 0.0, 0.0]
    popt, pcov = curve_fit(fitFunc_hmf_ratio_2D, (all_lg_masses, all_redshifts), all_ratios, p0=initial_guess)
  
    # Evaluate fit quality and plot the results, saving fit parameters and errors
    fitfilename = os.path.join(output_dir, 'HMF_redshift_evolution_ratio_fit_params.txt')
    print("Fit parameters:")
    print(f"a = {popt[0]:.4f}, b = {popt[1]:.4f}, c = {popt[2]:.4f}")
    print(f"d = {popt[3]:.4f}, e = {popt[4]:.4f}, f = {popt[5]:.4f}")

    with open(fitfilename, 'w') as f:
        f.write(f"fit parameters: a, b, c, d, e, f\n")
        f.write(f"{popt[0]:.4f} {popt[1]:.4f} {popt[2]:.4f} {popt[3]:.4f} {popt[4]:.4f} {popt[5]:.4f}\n")
    
    fig = plt.figure(figsize=(8,6), facecolor='white')
    ax = fig.gca()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(hmf_data_list)))
    labels = [f'z={redshift}' for redshift in redshifts]

    for i in range(len(hmf_data_list)):
        hmf_data = hmf_data_list[i]
        redshift = redshifts[i]
        
        bin_centers = hmf_data[:,2]
        number_density_all = hmf_data[:,3]
        number_density_selected = hmf_data[:,4]
        
        mask = number_density_all > 0
        ratio_masked = number_density_selected[mask]/number_density_all[mask]
        bin_centers_masked = bin_centers[mask]
        lg_masses = np.log10(bin_centers_masked)
        
        # Calculate fitted values using new 2D model
        z_array = np.full_like(lg_masses, redshift)
        predicted_ratios = fitFunc_hmf_ratio_2D((lg_masses, z_array), *popt)
        
        # Calculate error
        max_error = np.max(np.abs(ratio_masked - predicted_ratios))
        print(f"z={redshift}, max error: {max_error:.4f}")
        with open(fitfilename, 'a') as f:
            f.write(f"z={redshift}, max error: {max_error:.4f}\n")
       
        plt.scatter(bin_centers_masked, ratio_masked, c='none', edgecolor=colors[i], marker='o')
        
        # Add smooth fitted line
        lgM_list = np.linspace(np.log10(min(bin_centers_masked)), np.log10(max(bin_centers_masked)), 100)
        z_list = np.full_like(lgM_list, redshift)
        ratio_fit = fitFunc_hmf_ratio_2D((lgM_list, z_list), *popt)
        plt.plot(10**lgM_list, ratio_fit, color=colors[i],label=labels[i],alpha=0.5)
        plt.legend(fontsize=11)
    
    plt.xscale('log')
    plt.xlabel(r'Mass [$\mathrm{M}_{\odot}/\mathrm{h}$]', fontsize=14)
    plt.ylabel(r'$\text{HMF}_{\text{selected}}/\text{HMF}_\text{all}$',fontsize=14)
    #write text of the fitting formula
    fitting_eq_txt1 = r'$y = \frac{1}{1+\mathrm{p}(z)^{-(\lg M-\mathrm{q}(z))}}$' 
    fitting_eq_txt2 = r'$\mathrm{q}(z) = a + \frac{b}{1+z} + c(1+z)$'+'\n' + \
                    r'$\mathrm{p}(z) = d + \frac{e}{1+z} + f(1+z)$'
    plt.text(1e10, 0.5, fitting_eq_txt1, fontsize=17)
    plt.text(1e10, 0.3, fitting_eq_txt2, fontsize=14)
    ax.tick_params(direction='in', which='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(output_ratio_fit_filename,dpi=300)

def run_hmf_redshift_evolution():
    TNG_snaplist_file = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/TNG_fullsnapshot_redshifts.txt'
    #snapNum, scale factor, redshift
    snaplist = np.loadtxt(TNG_snaplist_file, skiprows=1)
    snapNums = snaplist[:,0].astype(int)
    redshifts = snaplist[:,2]

    #add snapNum 1 before the list
    snapNums = np.insert(snapNums, 0, 1)
    redshifts = np.insert(redshifts, 0, 15.0)

    gas_resolution, dark_matter_resolution = get_simulation_resolution(simulation_set)
    plot_hmf_redshift_evolution(snapNums, redshifts, dark_matter_resolution)    



def plot_hmfhist(redshift):
        
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/'
    output_dir = os.path.join(base_dir, 'analysis')
    # filename = os.path.join(output_dir, f'HMFhistogram_z_{redshift}.png')
    filename = os.path.join(output_dir, f'HMFhistogram_z_{redshift}_smallbox.png')
    scale_factor = 1.0/(1.+redshift)
    comoving_factor = scale_factor**3
    delta_lgM = 0.2
    # lgM_bin_edges = np.arange(6, 15+delta_lgM, delta_lgM) #log10(M [Msun/h])
    lgM_bin_edges = np.arange(4, 10+delta_lgM, delta_lgM) #log10(M [Msun/h])
    lgM_bin_centers = (lgM_bin_edges[:-1] + lgM_bin_edges[1:])/2
    
    HMF_lgM_sheth99 = np.array([HMF_Colossus(10**lgM, redshift, 'sheth99')* np.log(10)*10**lgM for lgM in lgM_bin_centers])
    HMF_lgM_sheth99_comoving = HMF_lgM_sheth99*comoving_factor
    nhalo_in_bin_comoving = HMF_lgM_sheth99_comoving*delta_lgM
    # boxsize1_cMpc = 200
    # boxsize2_cMpc = 150
    # boxsize3_cMpc = 100
    boxsize1_cMpc = 50
    boxsize2_cMpc = 10
    boxsize3_cMpc = 5
    boxsize4_cMpc = 1
    Nhalo_in_bin_box1 = nhalo_in_bin_comoving*(boxsize1_cMpc*h_Hubble)**3
    Nhalo_in_bin_box2 = nhalo_in_bin_comoving*(boxsize2_cMpc*h_Hubble)**3
    Nhalo_in_bin_box3 = nhalo_in_bin_comoving*(boxsize3_cMpc*h_Hubble)**3
    Nhalo_in_bin_box4 = nhalo_in_bin_comoving*(boxsize4_cMpc*h_Hubble)**3

    # Create the figure and plot the histograms
    fig = plt.figure(figsize=(10, 7), facecolor='white')
    ax = fig.gca()

    # Plot histograms for the three boxsizes
    width = delta_lgM * 1.0  # Slightly narrower than the bin width for better visualization

    # First boxsize case
    ax.bar(lgM_bin_centers, Nhalo_in_bin_box1, 
        width=width, 
        alpha=0.7, 
        color='red', 
        label=f'Box {boxsize1_cMpc} cMpc')

    # Second boxsize case
    ax.bar(lgM_bin_centers, Nhalo_in_bin_box2, 
        width=width*0.9, 
        alpha=0.7, 
        color='yellow', 
        label=f'Box {boxsize2_cMpc} cMpc')

    # Third boxsize case
    ax.bar(lgM_bin_centers, Nhalo_in_bin_box3, 
        width=width*0.8, 
        alpha=0.7, 
        color='green', 
        label=f'Box {boxsize3_cMpc} cMpc')
    
    # Fourth boxsize case
    ax.bar(lgM_bin_centers, Nhalo_in_bin_box4,
        width=width*0.7,
        alpha=0.7,
        color='royalblue',
        label=f'Box {boxsize4_cMpc} cMpc')

    # Set axis labels and title
    ax.set_xlabel(r'$\log_{10}(\mathrm{M} \, [\mathrm{M}_{\odot}/\mathrm{h}])$', fontsize=15)
    ax.set_ylabel(r'Number of halos in bin', fontsize=15)
    ax.set_title(f'Halo Mass Function Histogram at z = {redshift}', fontsize=16)
    # Set logarithmic y-scale for better visualization
    ax.set_yscale('log')
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1.5, alpha=0.7)


    # Add ticks for each dex (order of magnitude)
    x_major_ticks = np.arange(4, 15, 1)  # Major ticks for each dex on x-axis
    ax.set_xticks(x_major_ticks)
    # ax.set_xlim(6.0, 15.0)  # Adjust x limits for better visualization
    ax.set_xlim(4.0, 10.0)  # Adjust x limits for better visualization

    # Set y-axis to have major ticks at each power of 10
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.set_ylim(1e-2, 1e10)  # Adjust y limits to match your plot
    ax.tick_params(axis='both', direction='in', which='both')

    # Add grid for better readability
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=1.0)


    # Add legend
    ax.legend(loc='center right', frameon=True, fontsize=15)

    # Add text with information about the bin size
    ax.text(0.5, 0.9, f'Bin width = {delta_lgM} dex', 
            transform=ax.transAxes, fontsize=15,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')


def run_hmfhist():
    z_list = [20, 15, 10, 8, 6, 0]
    for z in z_list:
        plot_hmfhist(z)



def HMF_ratio_2Dbestfit(lg_mass, redshift):
    '''
    Parameters:
    lg_mass: log10(Mass [Msun/h])
    redshift
    Returns:
    HMF_selected/HMF_all
    '''
    # filepath = os.path.join('/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/analysis/',
    #                     'HMF_redshift_evolution_ratio_fit_params.txt')
    # with open(filepath, 'r') as f:
    #     f.readline()
    #     params_line = f.readline()
    #     params = [np.float64(param) for param in params_line.split()]
    # print("a, b, c, d, e, f: ", params)
    return fitFunc_hmf_ratio_2D((lg_mass, redshift), *hmf_ratio_params)

def HMF_2Dbestfit(lgM, redshift):
    '''
    Parameters:
    lgM: log10(Mass [Msun/h])
    redshift
    Returns:
    HMF [dN/dlgM [(Mpc/h)^(-3)]]  (in physical units, not comoving)
    '''

    return HMF_Colossus(10**lgM, redshift, 'sheth99')* np.log(10)*10**lgM * HMF_ratio_2Dbestfit(lgM, redshift)


#----------------------------------- Subhalo Mass Function -----------------------------------


#x  = m/M
#dN/dlgx = A* x**(-alpha) exp(-beta x**omega)
#lg[dN/dlgx] = lgA - alpha lgx - beta x**omega / ln(10)
def fitFunc_lg_dNdlgx(lgx,alpha,beta_ln10, omega, lgA):
    x = 10**lgx
    return lgA - alpha*lgx - beta_ln10*x**omega 

def piecewise_func_for_omega_lnbeta(z, c, m):
    b = c - m*6  # Derive b to ensure continuity at z=6
    result = np.where(z >= 6, c, m*z + b)
    return result

def plot_shmf_redshift_evolution(snapNums, redshifts, dark_matter_resolution):
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/'
    alpha_list = []; beta_list = []; omega_list = []; lgA_list = []
    for snapNum in snapNums:
        input_dir = os.path.join(base_dir, f'snap_{snapNum}', 'analysis')
        SHMF_fit_filename = os.path.join(input_dir, f'SHMF_BestFit_params_snap_{snapNum}.txt')
        # alpha, beta/ln10, omega, lgA, beta, A   
        with open(SHMF_fit_filename, 'r') as f:
            f.readline(); f.readline()
            SHMF_fit_data = f.readline().split()
            SHMF_fit_data = [np.float64(param) for param in SHMF_fit_data]
            alpha = SHMF_fit_data[0]
            beta_ln10 = SHMF_fit_data[1]
            omega = SHMF_fit_data[2]
            lgA = SHMF_fit_data[3]
            beta = SHMF_fit_data[4]
            A = SHMF_fit_data[5]
            alpha_list.append(alpha); beta_list.append(beta); omega_list.append(omega); lgA_list.append(lgA)
    ln_beta_list = [np.log(beta) for beta in beta_list]
    #plot SHMF redshift evolution
    output_filename = os.path.join(base_dir, 'analysis', 'SHMF_redshift_evolution_alpha_lgA.png')
    fig = plt.figure(figsize=(8,6), facecolor='white')
    ax = fig.gca()
    labels = [f'z={redshift}' for redshift in redshifts]
    ax.scatter(redshifts, alpha_list, c='r', marker='o', label=r'$\alpha$')
    #then use linear regression to fit the data
    popt, pcov = curve_fit(lambda x, a, b: a*x + b, redshifts, alpha_list)
    x_fit = np.linspace(np.min(redshifts), np.max(redshifts), 100)
    y_fit = popt[0]*x_fit + popt[1]
    ax.plot(x_fit, y_fit, color='r', linestyle='--',
            label=r'$\alpha$'+f' fit: {popt[0]:.2f}z {"-" if popt[1] < 0 else "+"} {abs(popt[1]):.2f}')
    #also plot van den Bosch 2016
    ax.scatter(0, p_unevolved[0], facecolors='none', edgecolors='grey', marker='o')
    ax.scatter(0, p_evolved[0], c='grey', marker='o')
    
    ax2 = ax.twinx()
    ax2.scatter(redshifts, lgA_list, c='b', marker='^', label=r'$\lg\mathrm{A}$')
    popt, pcov = curve_fit(lambda x, a, b: a*x + b, redshifts, lgA_list)
    y_fit = popt[0]*x_fit + popt[1]
    ax2.plot(x_fit, y_fit, color='b', linestyle='--', 
         label=r'$\lg\mathrm{A}$'+f' fit: {popt[0]:.2f}z {"-" if popt[1] < 0 else "+"} {abs(popt[1]):.2f}')
    ax2.scatter(0, p_unevolved[3], facecolors='none', edgecolors='grey', marker='^')
    ax2.scatter(0, p_evolved[3], c='grey', marker='^')

    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'$\alpha$', fontsize=14, color='r')
    ax2.set_ylabel(r'$\lg\mathrm{A}$', fontsize=14, color='b')
    ax.tick_params(direction='in', which='both', labelsize=12)
    ax2.tick_params(direction='in', which='both', labelsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    evolved_circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='grey', markersize=6)
    evolved_triangle = Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', markeredgecolor='grey', markersize=6)
    unevolved_circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='grey', markersize=6)
    unevolved_triangle = Line2D([0], [0], marker='^', color='w', markerfacecolor='none', markeredgecolor='grey', markersize=6)
    # Create the second legend (with both markers on each line)
    second_legend = ax.legend(
        [(evolved_circle, evolved_triangle), (unevolved_circle, unevolved_triangle)],
        ['Jiang&Bosch16 evolved', 'Jiang&Bosch16 unevolved'],
        handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)},
        loc='lower center', 
        fontsize=14
    )
    # Add the second legend manually
    first_legend = ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=14)
    ax.add_artist(second_legend)

    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_filename,dpi=300)


    #then plot omega and ln(beta)
    output_filename = os.path.join(base_dir, 'analysis', 'SHMF_redshift_evolution_omega_beta.png')
    fig = plt.figure(figsize=(8,6), facecolor='white')
    ax = fig.gca()
    labels = [f'z={redshift}' for redshift in redshifts]
    ax.scatter(redshifts, omega_list, c='r', marker='o', label=r'$\omega$')
    #then use a piecewise linear function to fit
    # Initial parameter guesses [c, m]
    omega_high_z = [omega for i, omega in enumerate(omega_list) if redshifts[i] >= 6]
    initial_c_omega = np.mean(omega_high_z)
    initial_m_omega = 0.5  # Start with a positive slope for z<6
    params_omega, _ = curve_fit(piecewise_func_for_omega_lnbeta,redshifts,omega_list,p0=[initial_c_omega, initial_m_omega])
    c_omega, m_omega = params_omega

    x_fit = np.linspace(0, np.max(redshifts), 100)
    y_fit_omega = piecewise_func_for_omega_lnbeta(x_fit, c_omega, m_omega)
    b_omega = c_omega - m_omega*6
    print('omega fit: c=%.2f, m=%.2f, b=%.2f' % (c_omega, m_omega, b_omega))
    ax.plot(x_fit, y_fit_omega, color='r', linestyle='--', 
        label=r'$\omega = %.2f$ for $z \geq 6$, $%.2f z + %.2f$ for $z < 6$' % 
        (c_omega, m_omega, b_omega))

    #also plot van den Bosch 2016
    ax.scatter(0, p_unevolved[2], facecolors='none', edgecolors='grey', marker='o')
    ax.scatter(0, p_evolved[2], c='grey', marker='o')


    ax2 = ax.twinx()
    ax2.scatter(redshifts, ln_beta_list, c='b', marker='^', label=r'$\ln\beta$')
    #then use a piecewise linear function to fit
    # Initial parameter guesses [c, m]
    ln_beta_high_z = [ln_beta for i, ln_beta in enumerate(ln_beta_list) if redshifts[i] >= 6]
    initial_c_ln_beta = np.mean(ln_beta_high_z)
    initial_m_ln_beta = 0.5  # Start with a positive slope for z<6
    params_ln_beta, _ = curve_fit(piecewise_func_for_omega_lnbeta,redshifts,ln_beta_list,p0=[initial_c_ln_beta, initial_m_ln_beta])
    c_ln_beta, m_ln_beta = params_ln_beta
    b_ln_beta = c_ln_beta - m_ln_beta*6
    print('ln beta fit: c=%.2f, m=%.2f, b=%.2f' % (c_ln_beta, m_ln_beta, b_ln_beta))
    y_fit_ln_beta = piecewise_func_for_omega_lnbeta(x_fit, c_ln_beta, m_ln_beta)
    ax2.plot(x_fit, y_fit_ln_beta, color='b', linestyle='--', 
         label=r'$\ln\beta = %.2f$ for $z \geq 6$, $%.2f z + %.2f$ for $z < 6$' % 
         (c_ln_beta, m_ln_beta, b_ln_beta))
    #also plot van den Bosch 2016
    ax2.scatter(0, np.log(np.log(10)*p_unevolved[1]), facecolors='none', edgecolors='grey', marker='^')
    ax2.scatter(0, np.log(np.log(10)*p_evolved[1]), c='grey', marker='^')

    y_min = min(0.3*np.min(omega_list), 0.3*np.min(ln_beta_list))
    y_max = max(1.1*np.max(omega_list), 1.1*np.max(ln_beta_list))
    ax.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'$\omega$', fontsize=14, color='r')
    ax2.set_ylabel(r'$\ln\beta$', fontsize=14, color='b')
    ax.tick_params(direction='in', which='both', labelsize=12)
    ax2.tick_params(direction='in', which='both', labelsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    evolved_circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markeredgecolor='grey', markersize=6)
    evolved_triangle = Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', markeredgecolor='grey', markersize=6)
    unevolved_circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='grey', markersize=6)
    unevolved_triangle = Line2D([0], [0], marker='^', color='w', markerfacecolor='none', markeredgecolor='grey', markersize=6)
    # Create the second legend (with both markers on each line)
    second_legend = ax.legend(
        [(evolved_circle, evolved_triangle), (unevolved_circle, unevolved_triangle)],
        ['Jiang&Bosch16 evolved', 'Jiang&Bosch16 unevolved'],
        handler_map={tuple: matplotlib.legend_handler.HandlerTuple(ndivide=None)},
        loc='lower center', 
        fontsize=14
    )
    # Add the second legend manually
    first_legend = ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=14)
    ax.add_artist(second_legend)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_filename,dpi=300)


    #show how the exponential tail of SHMF changes with omega and beta
    output_filename = os.path.join(base_dir, 'analysis', 'SHMF_redshift_evolution_exponential_tail.png')
    fig = plt.figure(figsize=(8,6), facecolor='white')
    ax = fig.gca()
    labels = [f'z={redshift}' for redshift in redshifts]
    lgx_all = [-2, -1, -0.8, -0.5, -0.2]

    for lgx in lgx_all:
        x = 10**lgx
        exponential_tail = [np.exp(-beta*x**omega) for beta, omega in zip(beta_list, omega_list)]
        ax.plot(redshifts, exponential_tail, label=f'lg(m/M)={lgx}')
    
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel(r'$\exp(-\beta (m/M)^{\omega})$', fontsize=14)
    ax.tick_params(direction='in', which='both', labelsize=12)
    ax.legend(fontsize=11)
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(output_filename,dpi=300)





def run_shmf_redshift_evolution():
    TNG_snaplist_file = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/TNG_fullsnapshot_redshifts.txt'
    #snapNum, scale factor, redshift
    snaplist = np.loadtxt(TNG_snaplist_file, skiprows=1)
    snapNums = snaplist[:,0].astype(int)
    redshifts = snaplist[:,2]

    #add snapNum 1 before the list
    snapNums = np.insert(snapNums, 0, 1)
    redshifts = np.insert(redshifts, 0, 15.0)

    gas_resolution, dark_matter_resolution = get_simulation_resolution(simulation_set)
    plot_shmf_redshift_evolution(snapNums, redshifts, dark_matter_resolution)



def SHMF_BestFit_dN_dlgx(lgx, redshift, SHMF_model):
    '''
    Parameters:
    lgx: np.log10(m/M)
    redshift
    SHMF_model
    Returns:
    dN/dlgx(x, z)
    '''
    if SHMF_model == 'BestFit_z':
        alpha_z = alpha_z_params[0]*redshift + alpha_z_params[1]
        lgA_z = lgA_z_params[0]*redshift + lgA_z_params[1]
        omega_z = piecewise_func_for_omega_lnbeta(redshift, omega_z_params[0], omega_z_params[1])
        lnbeta_z = piecewise_func_for_omega_lnbeta(redshift, lnbeta_z_params[0], lnbeta_z_params[1])
        beta_z = np.exp(lnbeta_z); beta_ln10_z = beta_z/np.log(10)
        lg_dNdlgx = fitFunc_lg_dNdlgx(lgx, alpha_z, beta_ln10_z, omega_z, lgA_z)
        return 10**lg_dNdlgx
    
    elif SHMF_model == 'Bosch16evolved':
        alpha, beta_ln10, omega, lgA = p_evolved
        lg_dNdlgx = fitFunc_lg_dNdlgx(lgx, alpha, beta_ln10, omega, lgA)
        return 10**lg_dNdlgx
    elif SHMF_model == 'Bosch16unevolved':
        alpha, beta_ln10, omega, lgA = p_unevolved
        lg_dNdlgx = fitFunc_lg_dNdlgx(lgx, alpha, beta_ln10, omega, lgA)
        return 10**lg_dNdlgx
    
    else:
        raise ValueError("SHMF Error: SHMF_model not supported")
    

'''
#dN/ d lnx, x = m/M, input ln(x)
def Subhalo_Mass_Function_ln(ln_m_over_M,bestfitparams=None):
    if SHMF_model == 'Giocoli2010':
        m_over_M = np.exp(ln_m_over_M)
        f0 = 0.1
        beta = 0.3
        gamma_value = 0.9
        x = m_over_M/beta
        return f0/(beta*gamma(1 - gamma_value)) * x**(-gamma_value) * np.exp(-x)
    elif SHMF_model == 'Bosch16evolved':
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
'''



#old version for testing
#dN/ d ln(x)
#input: ln_m_over_M, logM, z, SHMF_model, bestfitparams
def Subhalo_Mass_Function_ln_oldversion(ln_m_over_M, SHMF_model, bestfitparams=None):
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
        

#old version for testing
#use Bosch2016 model to calculate the DF heating
def integrand_oldversion(ln_m_over_M, logM, z, SHMF_model, *bestfitparams):
    if not bestfitparams:
        bestfitparams = None
    global G_grav,rho_b0,h_Hubble, Mpc, Msun
    
    eta = 1.0
    I_DF = 1.0
    
    M = 10**logM
    m_over_M = np.exp(ln_m_over_M)
    m = m_over_M * M  
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    DF_heating =  eta * 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Vel_Virial_analytic_oldversion(M/h_Hubble, z) *rho_g *I_DF
    if SHMF_model == 'BestFit':
        DF_heating *= Subhalo_Mass_Function_ln_oldversion(ln_m_over_M, SHMF_model, bestfitparams)
    else:
        DF_heating *= Subhalo_Mass_Function_ln_oldversion(ln_m_over_M, SHMF_model) 


    DF_heating *= HMF_Colossus(M,z) * np.log(10)*M   #convert from M to log10(M)
    
    return DF_heating



def plot_shmfhist(redshift):
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/'
    output_dir = os.path.join(base_dir, 'analysis')
    filename = os.path.join(output_dir, f'SHMFhistogram_z_{redshift}.png')
    #plot the BestFit SHMF histogram, x= m/M
    lgx_bin_edges = np.arange(-4, 0.0+0.1, 0.1)  #log10(m/M)
    lgx_bin_centers = (lgx_bin_edges[:-1] + lgx_bin_edges[1:])/2
    delta_lgx = lgx_bin_edges[1] - lgx_bin_edges[0]
    #calculate the SHMF for each bin center
    SHMF_lgx_bestfit = np.array([SHMF_BestFit_dN_dlgx(lgx, redshift, 'BestFit_z') for lgx in lgx_bin_centers])
    Nsubhalo_in_bin = SHMF_lgx_bestfit * delta_lgx  #number of subhalos in each bin

    #also compare with mass resolution
    mass_resolution = 1000 #Msun/h
    sub_mass_resolution = 50 * mass_resolution  #minimum subhalo mass
    lg_host_mass_list = [6.0, 6.5, 7.0, 7.5] #Msun/h
    color_list = ['blue','green','orange','red']
    critical_ratio_list = []
    for lg_host_mass in lg_host_mass_list:
        host_mass = 10**lg_host_mass
        critical_ratio = sub_mass_resolution / host_mass
        critical_ratio_list.append(critical_ratio)
    lg_critical_ratio_list = np.log10(critical_ratio_list)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    # Plot the histogram
    width = delta_lgx * 0.9  # Slightly narrower than the bin
    ax.bar(lgx_bin_centers, Nsubhalo_in_bin,
           width=width, 
           alpha=0.7,  
           label=f'SHMF at z={redshift}')
    # Set axis labels and title
    ax.set_xlabel(r'$\log_{10}(m/M)$', fontsize=15)
    ax.set_ylabel(r'Number of subhalos in bin', fontsize=15)
    ax.set_title(f'Subhalo Mass Function Histogram at z = {redshift}', fontsize=16)
    # Set logarithmic y-scale for better visualization
    ax.set_yscale('log')
    ax.axhline(y=1, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    # Add ticks for each dex (order of magnitude)
    x_major_ticks = np.arange(-4, 0, 1)  # Major ticks for each dex on x-axis
    
    ax.set_xticks(x_major_ticks)
    ax.set_xlim(-4.0, 0.0)  # Adjust x limits
    ax.set_ylim(1e-3, 1e2)    
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    #use vertical lines to indicate the critical ratio
    for i, lg_critical_ratio in enumerate(lg_critical_ratio_list):
        ax.axvline(x=lg_critical_ratio, color=color_list[i], linestyle='--', linewidth=2.0, alpha=1.0, label=f"host: lg(M/[Msun/h]) = {lg_host_mass_list[i]:.1f}, sub: 50x 1000 Msun/h")
    plt.legend(loc='upper right', frameon=True, fontsize=12)
    ax.text(0.2, 0.9, f'Bin width = {delta_lgx:.2f} dex',
            transform=ax.transAxes, fontsize=15,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def run_shmfhist():
    z_list = [15, 10, 8, 6, 0]
    # z_list = [15]
    for z in z_list:
        plot_shmfhist(z)


#----------------------------------- Variation of SHMF and its correction -----------------------------------

#correction factor for the variance of the cumulative SHMF, Equation 16 in Jiang & van den Bosch paperIII
def get_cumulativeSHMF_sigma_correction(N_avg, correction_model):
    if correction_model == 'superPoisson': #BK10
        epsilon = 0.18
        return np.sqrt(1+epsilon**2*N_avg)
    elif correction_model == 'supersubPoisson': #Jiang & van den Bosch paperIII, including both super and sub Poissonian correction
        N0 = 0.12
        x = np.sqrt(N_avg/N0)
        eta = 0.09
        epsilon = 0.18
        return (1 - eta*x**2 * np.exp(-x)) * np.sqrt(1+epsilon**2*N_avg)
    elif correction_model == 'None':
        return np.ones_like(N_avg)  # No correction, return 1 for all N_avg




#------------------------------ Random Sampling of SHMF -----------------------------------
#use inverse transform sampling from the cumulative SHMF, see section 5.1 of Hiroshima+ 2022
#SHMF_model = "BestFit_z": BestFit of TNG + redshift evolution fit

def get_normalized_SHMF_Cumulative(lgx_min, lgx_max, redshift, SHMF_model):
    """
    return: P(<psi) = N(< m/M)/Ntot_mean, not to be confused with N(> m/M)
    """
    lg_x_bin_edges = np.linspace(lgx_min, lgx_max, 50)
    lg_x_bin_centers = 0.5*(lg_x_bin_edges[1:] + lg_x_bin_edges[:-1])
    lg_x_bin_width = lg_x_bin_edges[1] - lg_x_bin_edges[0]
    dN_dlgx_mean = SHMF_BestFit_dN_dlgx(lg_x_bin_centers, redshift, SHMF_model)
    N_subs_per_bin_mean = dN_dlgx_mean * lg_x_bin_width

    # Calculate N(<psi): cumulative from left to right
    N_cumulative_less_than = np.cumsum(N_subs_per_bin_mean)
    # print("N_cumulative_less_than:", N_cumulative_less_than)
    # Normalize to probability distribution [0, 1]
    N_tot_mean = N_cumulative_less_than[-1]
    F_cumulative_bins = N_cumulative_less_than / N_tot_mean
    # Add the leftmost point: F = 0 at psi = psi_min
    F_cumulative = np.concatenate([[0.0], F_cumulative_bins])
    return lg_x_bin_edges, F_cumulative, N_tot_mean

def onetime_sample_SHMF_for_Ntot(lg_x_values, F_cumulative, Ntot):
    """
    Sample subhalo mass function for given total number using inverse transform
    
    Parameters:
    -----------
    lg_x_values, F_cumulative: array, the shape of SHMF probability P(psi < x)
        taken from get_normalized_SHMF_Cumulative
    Ntot : int
        total number of subhalos to generate
    Returns:
    --------
    sampled_lg_psi : array
        array of sampled log10(psi) values (length = Ntot)
    """
    if Ntot == 0:
        return np.array([])
    
    # Generate uniform random numbers [0, 1]
    u_values = np.random.uniform(0, 1, Ntot)
    
    # Inverse transform: F^(-1)(u) using linear interpolation
    sampled_lg_psi = np.interp(u_values, F_cumulative, lg_x_values)
    
    return sampled_lg_psi



def test_SHMF_sampling():
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/'
    output_dir = os.path.join(base_dir, 'analysis')

    redshift = 15
    SHMF_model = "BestFit_z"
    lgx_min = -4
    lgx_max =  0
    lg_x_vals, F_vals, N_mean = get_normalized_SHMF_Cumulative(-4, 0, redshift, SHMF_model)
    print(f"Mean total number of subhalos: {N_mean:.2f}")
    print(f"F_vals range: [{F_vals[0]:.3f}, {F_vals[-1]:.3f}]")


    #plot histogram of the generated lg_psi and compare with BestFit_z model
    lg_x_bin_edges = np.linspace(lgx_min, lgx_max, 50)
    lg_x_bin_centers = 0.5*(lg_x_bin_edges[1:] + lg_x_bin_edges[:-1])
    lg_x_bin_width = lg_x_bin_edges[1] - lg_x_bin_edges[0]
    n_bins = len(lg_x_bin_centers)

    dN_dlgx_mean = SHMF_BestFit_dN_dlgx(lg_x_bin_centers, redshift, SHMF_model)
    theoretical_counts = dN_dlgx_mean * lg_x_bin_width

    n_samples = 500
    sample_results = []
    for i in range(n_samples):
        #Ntot_sample = round(N_mean)  # Use mean as test case, without Poisson fluctuations
        Ntot_sample = np.random.poisson(N_mean) #with Poisson fluctuations

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_samples} samples")
        
        # Bin the sample
        if Ntot_sample == 0:
            sample_counts = np.zeros(n_bins)
        else:
            # Sample individual subhalo masses (psi values)
            sampled_lg_psi = onetime_sample_SHMF_for_Ntot(lg_x_vals, F_vals, Ntot_sample)
            sample_counts, _ = np.histogram(sampled_lg_psi, bins=lg_x_bin_edges)
        sample_results.append(sample_counts)
    
    # Convert to numpy array for easier manipulation

    sample_results = np.array(sample_results)  # Shape: [n_samples, n_bins]
    # Convert differential counts to cumulative N(>psi)
    sample_cumulative = np.zeros_like(sample_results)
    for i in range(n_samples):
        sample_cumulative[i, :] = np.cumsum(sample_results[i, ::-1])[::-1]

    theoretical_cumulative = np.zeros_like(theoretical_counts)
    theoretical_cumulative = np.cumsum(theoretical_counts[::-1])[::-1]

    # Calculate statistics for cumulative data
    mean_cumulative = np.mean(sample_cumulative, axis=0)
    std_cumulative = np.std(sample_cumulative, axis=0)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 7), facecolor='white')


    # Plot cumulative N(>psi) with dashed bounds
    ax1.plot(lg_x_bin_centers, mean_cumulative, 'b-', linewidth=2,
            label=f'Sampled mean ({n_samples} samples)', marker='o', markersize=4)

    # Plot ±1σ bounds with dashed lines
    ax1.plot(lg_x_bin_centers, mean_cumulative + std_cumulative, 'b--', linewidth=1.5, 
            alpha=0.8)
    ax1.plot(lg_x_bin_centers, mean_cumulative - std_cumulative, 'b--', linewidth=1.5, 
            alpha=0.8, label='±1σ with Poisson')
    
    ax1.plot(lg_x_bin_centers, theoretical_cumulative, 'r-', linewidth=3,
            label='Theoretical SHMF model', marker='s', markersize=4)
    
    sqrt_n_error = np.sqrt(theoretical_cumulative)  
    ax1.errorbar(lg_x_bin_centers, theoretical_cumulative, yerr=sqrt_n_error,
            fmt='rs-', linewidth=3, markersize=4, capsize=3, capthick=1,
            label=r'Theoretical ± $\sqrt{N}$', alpha=0.2)

    ax1.set_xlabel('lg(m/M)', fontsize=14)
    ax1.set_ylabel('N(> m/M)',fontsize=14)
    ax1.set_title(f'Cumulative SHMF: {n_samples} iterations, mean N={N_mean:.2f}', fontsize=16)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1e-3)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'SHMF_Poisson_sampling_z_{redshift}.png'), dpi=300)



#----------------------------------- Mass Limits -----------------------------------
def T_CMB(z):
    return T0_CMB*(1+z)

def T_K_Meiksin11(z):
    if z > 500:
        return T_CMB(z)
    elif 300 < z <= 500:
        return 1.46*(1+z)**1.10
    elif 60 < z <= 300:
        return 0.146*(1+z)**1.50
    elif 6 <= z <= 60:
        return 0.023*(1+z)**1.95
    else:
        raise ValueError("T_K_Meiksin11 Error: z out of range")

def plot_T_K():
    z_list = np.logspace(np.log10(6),np.log10(50),100)
    T_K_list = [T_K_Meiksin11(z) for z in z_list]

    #also read output from 21cmFAST fiducial run
    Tk_21cmFAST_filename = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/global_Tk_z_fid.txt'

    Tk_21cmFAST_data = np.loadtxt(Tk_21cmFAST_filename, skiprows=1)
    z_21cmFAST = Tk_21cmFAST_data[:,0]
    Tk_21cmFAST = Tk_21cmFAST_data[:,1]

    fig = plt.figure(facecolor='white')
    ax = fig.gca()
    ax.plot(z_list, T_K_list, color='r', label='Meiksin11')
    ax.plot(z_21cmFAST, Tk_21cmFAST, color='b', label='21cmFAST')
    ax.invert_xaxis()
    plt.xlabel('z', fontsize=14)
    plt.ylabel('T_K [K]', fontsize=14)
    plt.yscale('log')   
    ax.tick_params(direction='in', which='both', labelsize=12)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.savefig('/home/zwu/21cm_project/unified_model/Analytic_results/T_K_evolution.png',dpi=300)


def get_M_Jeans(z):
    #in units of Msun/h
    if 6 <= z <= 60:
        return 220*(1+z)**1.425*h_Hubble
    else:
        raise ValueError("M_Jeans Error: z out of range")

def get_M_Jeans_Meiksin11(z):
    #in units of Msun/h
    Cs = np.sqrt(5.0/3.0*kB*T_K_Meiksin11(z)/(mu*mp))
    rho_m = rho_m0*(1+z)**3*Msun/Mpc**3
    lambda_Jeans = np.sqrt(np.pi*Cs**2/(G_grav*rho_m))
    M_Jeans = 4/3*np.pi*(lambda_Jeans/2.0)**3*rho_m
    M_Jeans = M_Jeans/Msun*h_Hubble
    return M_Jeans

def get_M_Jeans_21cmFAST(z):
    #in units of Msun/h
    Tk_21cmFAST_filename = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/global_Tk_z_fid.txt'
    Tk_21cmFAST_data = np.loadtxt(Tk_21cmFAST_filename, skiprows=1)
    z_21cmFAST = Tk_21cmFAST_data[:,0]
    Tk_21cmFAST = Tk_21cmFAST_data[:,1]

    if z < min(z_21cmFAST) or z > max(z_21cmFAST): 
        raise ValueError("get_M_Jeans_21cmFAST Error: z out of range")
    
    interp_func = interp1d(z_21cmFAST, Tk_21cmFAST, kind='linear', assume_sorted=False)
    Tk_z_interp = interp_func(z)
    Cs = np.sqrt(5.0/3.0*kB*Tk_z_interp/(mu*mp))
    rho_m = rho_m0*(1+z)**3*Msun/Mpc**3
    lambda_Jeans = np.sqrt(np.pi*Cs**2/(G_grav*rho_m))
    M_Jeans = 4/3*np.pi*(lambda_Jeans/2.0)**3*rho_m
    M_Jeans = M_Jeans/Msun*h_Hubble
    return M_Jeans

def plot_M_Jeans():
    z_list = np.logspace(np.log10(6),np.log10(34),100)
    M_Jeans_list = [get_M_Jeans(z) for z in z_list]
    M_Jeans_Meiksin11_list = [get_M_Jeans_Meiksin11(z) for z in z_list]
    M_Jeans_21cmFAST_list = [get_M_Jeans_21cmFAST(z) for z in z_list]

    fig = plt.figure(facecolor='white')
    ax = fig.gca()
    ax.plot(z_list, M_Jeans_list, color='r', label='220*(1+z)^1.425')
    ax.plot(z_list, M_Jeans_Meiksin11_list, color='g', linestyle='--', label='Meiksin11')
    ax.plot(z_list, M_Jeans_21cmFAST_list, color='b', label='21cmFAST')
    ax.invert_xaxis()
    plt.xlabel('z', fontsize=14)
    plt.ylabel('M_Jeans [Msun/h]', fontsize=14)
    plt.yscale('log')   
    ax.tick_params(direction='in', which='both', labelsize=12)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.savefig('/home/zwu/21cm_project/unified_model/Analytic_results/M_Jeans_z.png',dpi=300)


if __name__ == "__main__":
    # HMF_ratio_2Dbestfit(9, 0.0)
    # run_shmf_redshift_evolution()
  
    # plot_M_Jeans()
    # run_hmf_redshift_evolution()
    # run_hmfhist()
    # run_shmfhist()

    test_SHMF_sampling()