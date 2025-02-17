import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from TNGDataHandler import load_processed_data



def maxwell_boltzmann_pdf(x, sigma):
    """
    Maxwell-Boltzmann probability density function in 3D
    
    Parameters:
    -----------
    x : array_like
        Points at which to evaluate the distribution
    sigma : float
        Scale parameter (standard deviation)
        
    Returns:
    --------
    array_like
        Probability density at points x
    """
    return 4 * np.pi * x**2 * np.exp(-x**2 / (2 * sigma**2)) / (sigma**3 * (2 * np.pi)**(3/2))

def fit_maxwell_boltzmann(data, bins, range_fit=None, initial_guess=None):
    """
    Fit Maxwell-Boltzmann distribution to data
    
    Parameters:
    -----------
    data : array_like
        Data to fit
    bins : int
        Number of bins for histogram
    range_fit : tuple, optional
        (min, max) range for fitting
    initial_guess : float, optional
        Initial guess for sigma parameter
        
    Returns:
    --------
    tuple
        (fitted parameters, parameter covariances, x points, fitted y points)
    
    example usage:
    popt, pcov, x_fit, y_fit = fit_maxwell_boltzmann(
        data=mach_numbers,
        bins=50,
        range_fit=(0, 5),
        initial_guess=1.0
    )
    sigma_fit = popt[0]
    """
    if range_fit is None:
        range_fit = (np.min(data), np.max(data))
    
    # Create histogram of data
    hist, bin_edges = np.histogram(data, bins=bins, range=range_fit, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if initial_guess is None:
        initial_guess = np.std(data)
    
    # Fit the Maxwell-Boltzmann distribution
    popt, pcov = curve_fit(maxwell_boltzmann_pdf, bin_centers, hist, p0=[initial_guess])
    
    # Generate points for the fitted curve
    x_fit = np.linspace(range_fit[0], range_fit[1], 200)
    y_fit = maxwell_boltzmann_pdf(x_fit, popt[0])
    
    return popt, pcov, x_fit, y_fit
    

def plot_2D_histogram(data, output_dir):
    """
    Create 2D histograms of various TNG simulation properties.
    
    Parameters:
    -----------
    data : ProcessedTNGData
        Processed TNG data container
    output_dir : str
        Directory to save the output plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from the ProcessedTNGData container
    host_indices = data.subhalo_data['host_index'].value
    
    # Host halo properties (need to use host_indices to match with subhalos)
    host_mass = data.halo_data['GroupMass'].value[host_indices]
    host_M200 = data.halo_data['Group_M_Crit200'].value[host_indices]
    host_R200 = data.halo_data['Group_R_Crit200'].value[host_indices] 
    
    # Subhalo properties
    subhalo_mass = data.subhalo_data['SubMass'].value
    halfmass_radius = data.subhalo_data['SubHalfmassRad'].value
    vmaxrad = data.subhalo_data['SubVmaxRad'].value
    mach_number = data.subhalo_data['mach_number'].value
    vmaxrad_tcross = data.subhalo_data['vmaxrad_tcross'].value
    host_tff = data.subhalo_data['host_t_ff'].value
    a_number = data.subhalo_data['A_number'].value
    
    # Total mass vs subhalo mass
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist2d(np.log10(host_mass), np.log10(subhalo_mass), bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(M$_{host}$ [Msun/h])', fontsize=14)
    plt.ylabel(r'log$_{10}$(M$_{sub}$ [Msun/h])', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'Mtot_Msub.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # M200 vs subhalo mass
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist2d(np.log10(host_M200), np.log10(subhalo_mass), bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(M$_{200}$ [Msun/h])', fontsize=14)
    plt.ylabel(r'log$_{10}$(M$_{sub}$ [Msun/h])', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'M200_Msub.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # R200 vs halfmass radius
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist2d(np.log10(host_R200), np.log10(halfmass_radius), bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(R$_{200}$ [m])', fontsize=14)
    plt.ylabel(r'log$_{10}$(r$_{sub,halfmass}$ [m])', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'R200_rsubhalfmass.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # R200 vs Vmax radius
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist2d(np.log10(host_R200), np.log10(vmaxrad), bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(R$_{200}$ [m])', fontsize=14)
    plt.ylabel(r'log$_{10}$(r$_{sub,VmaxRad}$ [m])', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'R200_subhaloVmaxRad.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Free-fall time vs crossing time
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist2d(np.log10(host_tff), np.log10(vmaxrad_tcross), bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(t$_{ff,host}$ [s])', fontsize=14)
    plt.ylabel(r'log$_{10}$(t$_{cross}$ [s])', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'tff_tcross.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # M200 vs Mach number
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist2d(np.log10(host_M200), mach_number, bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(M$_{200}$ [Msun/h])', fontsize=14)
    plt.ylabel(r'Mach number', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'M200_Mach.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # M200 vs A number
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    plt.hist2d(np.log10(host_M200), np.log10(a_number), bins=50)
    plt.colorbar(label='Counts')
    plt.xlabel(r'log$_{10}$(M$_{200}$ [Msun/h])', fontsize=14)
    plt.ylabel(r'log$_{10}$(A number)', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'M200_Anumber.png'), dpi=300, bbox_inches='tight')
    plt.close()

    #plot Mach number in different bins of M200 (1D plot, divide M200 into 5 bins)
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    min_M200 = np.min(host_M200)
    max_M200 = np.max(host_M200)
    log_min_M200 = np.log10(min_M200)
    log_max_M200 = np.log10(max_M200)
    M200_bins = np.logspace(log_min_M200, log_max_M200, 6)

    #save best fit sigma to a file
    with open(os.path.join(output_dir, 'best_fit_Mach_sigma.txt'), 'w') as f:
        f.write("M200_min, M200_max, Best Fit Sigma\n")

    colors = plt.cm.plasma(np.linspace(0, 1, 5)) #color map for different bins

    for i in range(5):
        mask = (host_M200 > M200_bins[i]) & (host_M200 < M200_bins[i+1])
        plt.hist(mach_number[mask], bins=50, histtype='step', linewidth=2, density=True, color=colors[i],label=f'{M200_bins[i]:.2e} - {M200_bins[i+1]:.2e}')
    
        #fit Maxwell-Boltzmann distribution
        popt, pcov, x_fit, y_fit = fit_maxwell_boltzmann(
            data=mach_number[mask],
            bins=50,
            range_fit=(0, 5),
            initial_guess=1.0
        )
        plt.plot(x_fit, y_fit, linestyle='--', color=colors[i])

        with open(os.path.join(output_dir, 'best_fit_Mach_sigma.txt'), 'a') as f:
            f.write(f"{M200_bins[i]:.2e}, {M200_bins[i+1]:.2e}, {popt[0]}\n")

    plt.xlabel('Mach Number', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'M200_Mach_bins.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
       
    

def compare_mach_numbers(simulation_set, snapNums):
    """
    Compare Mach number distributions at different redshifts.
    
    Parameters:
    -----------
    simulation_set : str
        Name of the simulation set
    snapNums : list
        List of snapshot numbers to compare
    """
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    num_snapshots = len(snapNums)
    
    #write best fit sigma to a file
    with open(os.path.join(base_dir, simulation_set, 'analysis', 'best_fit_Mach_sigma.txt'), 'w') as f:
        f.write("Snapshot, Redshift, Best Fit Sigma\n")

    # Setup the plot
    fig = plt.figure(figsize=(8, 6), facecolor='w')
    colors = plt.cm.rainbow(np.linspace(0, 1, num_snapshots))
    
    for snapNum, color in zip(snapNums, colors):
        # Load data
        processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                    f'processed_halos_snap_{snapNum}.h5')
        data = load_processed_data(processed_file)
        
        # Get Mach numbers and filter out Mach > 5
        mach_numbers = data.subhalo_data['mach_number'].value
        mach_numbers = mach_numbers[mach_numbers < 5]
        redshift = data.header['Redshift']
        
        # Plot normalized distribution
        plt.hist(mach_numbers, bins=50, density=True, histtype='step', 
                color=color, label=f"z = {redshift:.2f}", linewidth=2)

        # Fit Maxwell-Boltzmann distribution and plot the best fit curve
        popt, pcov, x_fit, y_fit = fit_maxwell_boltzmann(
            data=mach_numbers,
            bins=50,
            range_fit=(0, 5),
            initial_guess=1.0
        )
        print("best fit sigma: ", popt[0])
        plt.plot(x_fit, y_fit, color=color, linestyle='--')

        #write best fit sigma to a file
        with open(os.path.join(base_dir, simulation_set, 'analysis', 'best_fit_Mach_sigma.txt'), 'a') as f:
            f.write(f"{snapNum}, {redshift}, {popt[0]}\n")

        
        # Print statistics
        print(f"\nSnapshot {snapNum}, z = {redshift:.2f}")
        print(f"Number of subhalos: {len(mach_numbers)}")
        print(f"Mean Mach number: {np.mean(mach_numbers):.2f}")
        print(f"Median Mach number: {np.median(mach_numbers):.2f}")
        print(f"Std Mach number: {np.std(mach_numbers):.2f}")
        print(f"Fraction subsonic (M < 1): {np.sum(mach_numbers < 1) / len(mach_numbers):.2%}")
    
    plt.xlabel('Mach Number')
    plt.ylabel('Probability Density')
    plt.title(f'Mach Number Distribution at Different Redshifts\n{simulation_set}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_dir = os.path.join(base_dir, simulation_set, 'analysis')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, 'mach_number_distribution_cutMach5.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    simulation_set = 'TNG50-1'
    
    snapNum = 13
    
    base_dir = '/home/zwu/21cm_project/unified_model/TNG_results/'
    processed_file = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 
                                f'processed_halos_snap_{snapNum}.h5')
    data = load_processed_data(processed_file)
    # Create plots
    output_dir = os.path.join(base_dir, simulation_set, f'snap_{snapNum}', 'analysis')
    plot_2D_histogram(data, output_dir)
    
    # Compare Mach numbers across snapshots
    # snapNums = [2, 3, 4, 6, 8, 11, 13]
    
    # compare_mach_numbers(simulation_set, snapNums)
