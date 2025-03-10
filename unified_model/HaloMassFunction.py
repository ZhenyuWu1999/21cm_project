from physical_constants import *
from colossus.lss import mass_function
from scipy.special import gamma, expm1
import numpy as np
from Config import SHMF_model, simulation_set, hmf_ratio_params, \
p_evolved, p_unevolved, alpha_z_params, lgA_z_params, omega_z, beta_ln10_z
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from TNGDataHandler import get_simulation_resolution
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib

#output dn/dM in the unit of [(Mpc/h)^(-3) (Msun/h)^(-1)]
#input M in the unit of Msun/h
def HMF_Colossus(M, z, model = 'press74'):
    if model == 'press74' or model == 'sheth99':
        mfunc = mass_function.massFunction(M, z, model = model, q_out = 'M2dndM')
    elif model == 'reed07':
        ps_path = '/home/zwu/21cm_project/unified_model/TNG_results/TNG50-1/analysis/input_spectrum_PLANCK15.txt'
        #mfunc = mass_function.massFunction(M, z, model = model, ps_args={'model': 'test', 'path': ps_path}, q_out = 'M2dndM')
        mfunc = mass_function.massFunction(M, z, model = model, ps_args={'model': 'eisenstein98'}, q_out = 'M2dndM')
    elif model == 'tinker08':
        mfunc = mass_function.massFunction(M, z, model = model, mdef='200c', q_out = 'M2dndM')
    else:
        print('Error: model not supported')
        return -999
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
    ax = fig.gca()
 
    logM_limits = [6, np.log10(1.1*max_M)]  # Limits for log10(M [Msun/h])
    HMF_lgM_press74 = []
    HMF_lgM_sheth99 = []
    # HMF_lgM_reed07 = []
    # HMF_lgM_tinker08 = []
    logM_list = np.linspace(logM_limits[0], logM_limits[1],57)
    #plot analytical HMF
    for logM in logM_list:
        M = 10**(logM)
        HMF_lgM_press74.append(HMF_Colossus(10**logM, current_redshift, 'press74')* np.log(10)*M)  
        HMF_lgM_sheth99.append(HMF_Colossus(10**logM, current_redshift, 'sheth99')* np.log(10)*M)
        # HMF_lgM_reed07.append(HMF_Colossus(10**logM, current_redshift, 'reed07')* np.log(10)*M)
        # HMF_lgM_tinker08.append(HMF_Colossus(10**logM, current_redshift, 'tinker08')* np.log(10)*M)
    #plot the dark matter resolution and TNG HMF
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(100*dark_matter_resolution, color='black', linestyle='--')
    plt.scatter(bin_centers, number_density, c='none', edgecolor='blue', marker='o', label='All TNG halos')
    plt.scatter(bin_centers, number_density_selected, c='none', edgecolor='green', marker='^',label='Selected TNG halos')

    plt.plot(10**(logM_list),HMF_lgM_press74,color='k',linestyle='-',label='Press74')
    plt.plot(10**(logM_list),HMF_lgM_sheth99,color='red',linestyle='-',label='Sheth99')
    # plt.plot(10**(logM_list),HMF_lgM_reed07,color='orange',linestyle='-',label='Reed07')
    # plt.plot(10**(logM_list),HMF_lgM_tinker08,color='purple',linestyle='-',label='Tinker08')
    plt.legend(fontsize=13)
    
    plt.xlabel(r'Mass [$\mathrm{M}_{\odot}/\mathrm{h}$]', fontsize=14)
    plt.ylabel(r'$\frac{\text{dN}}{\text{ d\lg M}}$ [$(\text{Mpc/h})^{-3}$]',fontsize=14)
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
        #sheth99
        lgM = np.log10(bin_centers)
        HMF_lgM_sheth99 = np.array([HMF_Colossus(10**lgM[j], redshift, 'sheth99')* np.log(10)*10**lgM[j] for j in range(len(lgM))])
        
        plt.plot(10**lgM,HMF_lgM_sheth99*comoving_factor, color=colors[i],linestyle='-',label=labels[i])
        plt.scatter(bin_centers, number_density_all*comoving_factor, c='none', edgecolor=colors[i], marker='o')
        plt.scatter(bin_centers, number_density_selected*comoving_factor, c='none', edgecolor=colors[i], marker='^')
    
    plt.axvline(100*dark_matter_resolution, color='black', linestyle='--')
    plt.legend(fontsize=11)
    plt.xlabel(r'Mass [$\mathrm{M}_{\odot}/\mathrm{h}$]', fontsize=14)
    plt.ylabel(r'$\frac{\text{dN}}{\text{ d\lg M}}$ [$(\text{cMpc/h})^{-3}$]',fontsize=14)
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
        # Avoid zero division
        mask = number_density_all > 0
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


#----------------------------------- Subhalo Mass Function -----------------------------------

def get_M_Jeans(z):
    return 220*(1+z)**1.425*h_Hubble


#x  = m/M
#dN/dlgx = A* x**(-alpha) exp(-beta x**omega)
#lg[dN/dlgx] = lgA - alpha lgx - beta x**omega / ln(10)
def fitFunc_lg_dNdlgx(lgx,alpha,beta_ln10, omega, lgA):
    x = 10**lgx
    return lgA - alpha*lgx - beta_ln10*x**omega 


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
            label=r'$\alpha$'+f' fit: {popt[0]:.4f}z {"-" if popt[1] < 0 else "+"} {abs(popt[1]):.4f}')
    #also plot van den Bosch 2016
    ax.scatter(0, p_unevolved[0], facecolors='none', edgecolors='grey', marker='o')
    ax.scatter(0, p_evolved[0], c='grey', marker='o')
    
    ax2 = ax.twinx()
    ax2.scatter(redshifts, lgA_list, c='b', marker='^', label=r'$\lg\mathrm{A}$')
    popt, pcov = curve_fit(lambda x, a, b: a*x + b, redshifts, lgA_list)
    y_fit = popt[0]*x_fit + popt[1]
    ax2.plot(x_fit, y_fit, color='b', linestyle='--', 
         label=r'$\lg\mathrm{A}$'+f' fit: {popt[0]:.4f}z {"-" if popt[1] < 0 else "+"} {abs(popt[1]):.4f}')
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
    #then use a constant average to fit (only fit z>=6)
    redshifts_fit = np.array([redshift for redshift in redshifts if redshift >= 6])
    omega_list_fit = np.array([omega for i, omega in enumerate(omega_list) if redshifts[i] >= 6])
    x_fit = np.linspace(6.0,np.max(redshifts),100)
    y_fit = np.full_like(x_fit, np.mean(omega_list_fit))
    ax.plot(x_fit, y_fit, color='r', linestyle='--', label=r'$\omega$'+f' = {np.mean(omega_list_fit):.4f}')
    #also plot van den Bosch 2016
    ax.scatter(0, p_unevolved[2], facecolors='none', edgecolors='grey', marker='o')
    ax.scatter(0, p_evolved[2], c='grey', marker='o')

    ax2 = ax.twinx()
    ax2.scatter(redshifts, ln_beta_list, c='b', marker='^', label=r'$\ln\beta$')
    #then use a constant average to fit (only fit z>=6)
    ln_beta_list_fit = np.array([ln_beta for i, ln_beta in enumerate(ln_beta_list) if redshifts[i] >= 6])
    y_fit = np.full_like(x_fit, np.mean(ln_beta_list_fit))
    ax2.plot(x_fit, y_fit, color='b', linestyle='--', label=r'$\ln\beta$'+f' = {np.mean(ln_beta_list_fit):.4f}')
    ax2.scatter(0, np.log(np.log(10)*p_unevolved[1]), facecolors='none', edgecolors='grey', marker='^')
    ax2.scatter(0, np.log(np.log(10)*p_evolved[1]), c='grey', marker='^')

    y_min = min(0.3*np.min(omega_list), 0.3*np.min(ln_beta_list))
    y_max = max(1.1*np.max(omega_list), 1.1*np.max(ln_beta_list))
    ax.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    ax.set_xlabel('t', fontsize=14)
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



def SHMF_BestFit_dN_dlgx(x, redshift):
    '''
    Parameters:
    x: m/M
    redshift
    Returns:
    dN/dlgx(x, z)
    '''
    if SHMF_model == 'BestFit_z':
        if (redshift < 6):
            #omega and beta not available
            raise ValueError("SHMF Error: omega and beta not available for z<6")
        alpha_z = alpha_z_params[0] + alpha_z_params[1]*redshift 
        lgA_z = lgA_z_params[0] + lgA_z_params[1]*redshift
        lg_dNdlgx = fitFunc_lg_dNdlgx(np.log10(x), alpha_z, beta_ln10_z, omega_z, lgA_z)
        return 10**lg_dNdlgx

    
    elif SHMF_model == 'Bosch16evolved':
        alpha, beta_ln10, omega, lgA = p_evolved
        lg_dNdlgx = fitFunc_lg_dNdlgx(np.log10(x), alpha, beta_ln10, omega, lgA)
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

if __name__ == "__main__":
    # HMF_ratio_2Dbestfit(9, 0.0)
    # run_shmf_redshift_evolution()
    pass
