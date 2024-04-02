
import yt
import yt.extensions.legacy
import numpy as np
import matplotlib.pyplot as plt
import h5py
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from physical_constants import *
from scipy.special import gamma
import copy
from unyt import unyt_array
import os
#import yt.extensions.legacy

from scipy.special import gamma
from scipy.integrate import solve_ivp, quad
from scipy.integrate import nquad
from scipy.optimize import curve_fit

def print_attrs(name, obj):
    """Helper function to print the name of an HDF5 object and its attributes."""
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))

def display_hdf5_contents(filepath):
        # Open the HDF5 file in read-only mode
    with h5py.File(filepath, 'r') as f:       
        # Display all groups and datasets in the file
        for name, item in f.items():
            if isinstance(item, h5py.Group):
                print(f"Group: {name}")
                for key, val in item.attrs.items():
                    print(f"    {key}: {val}")
                for subname, subitem in item.items():
                    print_attrs(f"{name}/{subname}", subitem)
            elif isinstance(item, h5py.Dataset):
                print(f"Dataset: {name}")
                for key, val in item.attrs.items():
                    print(f"    {key}: {val}")

#output dn/dM in the unit of [(Mpc/h)^(-3) (Msun/h)^(-1)]
#input M in the unit of Msun/h
def HMF_Colossus(M, z):
    global rho_m0, h_Hubble
    mfunc = mass_function.massFunction(M, z, model = 'press74', q_out = 'M2dndM')
    return mfunc/M**2*rho_m0*(1+z)**3/h_Hubble**2

#dn/ d ln(m/M)
def Subhalo_Mass_Function(m_over_M):
    f0 = 0.1
    beta = 0.3
    gamma_value = 0.9
    x = m_over_M/beta
    return f0/(beta*gamma(1 - gamma_value)) * x**(-gamma_value) * np.exp(-x)

def Vel_Virial(M_vir_in_Msun, z):
    #M_vir in solar mass, return virial velocity in m/s    
    global h_Hubble, Omega_m
    #Delta_vir = Overdensity_Virial(z)
    Delta_vir = 200
    V_vir = 163*1e3 * (M_vir_in_Msun/1e12*h_Hubble)**(1/3) * (Delta_vir/200)**(1/6) * Omega_m**(1/6) *(1+z)**(1/2)
    return V_vir


#dN/ d ln(ln(m/M))
def Subhalo_Mass_Function_ln(ln_m_over_M,SHMF_model):
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



def integrand(ln_m_over_M, logM, z):
    global G_grav,rho_b0,h_Hubble, Mpc, Msun
    
    eta = 1.0
    I_DF = 1.0
    
    M = 10**logM
    m_over_M = np.exp(ln_m_over_M)
    m = m_over_M * M  
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    DF_heating =  eta * 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Vel_Virial(M/h_Hubble, z) *rho_g *I_DF
    SHMF_model = 'Bosch2016'
    DF_heating *= Subhalo_Mass_Function_ln(ln_m_over_M,SHMF_model) 
    DF_heating *= HMF_Colossus(M,z) * np.log(10)*M   #convert from M to log10(M)
    
    return DF_heating


def plot_hmf(group_data, ds_group, current_redshift, mass_resolution, host_halo_mask, filename):

    # Convert mass to 'Msun/h'
    group_mass_solar = group_data['halos', 'Mvir'][host_halo_mask].in_units('Msun/h')

    # Create a histogram (logarithmic bins and logarithmic mass)
    bins = np.logspace(np.log10(min(group_mass_solar)), np.log10(max(group_mass_solar)), num=50)
    hist, bin_edges = np.histogram(group_mass_solar, bins=bins)

    # Convert counts to number density
    volume = ds_group.domain_width.in_units('Mpc/h').prod() 
    print(  "volume = ", volume)
    log_bin_widths = np.diff(np.log10(bins))
    number_density = hist / volume / log_bin_widths

    # Plot the mass function
    
    fig = plt.figure(facecolor='white')
    

    cosmology.setCosmology('planck18')
    logM_limits = [5, 11]  # Limits for log10(M [Msun/h])
    HMF_lgM = []
    logM_list = np.linspace(logM_limits[0], logM_limits[1],100)
    for logM in logM_list:
        
        M = 10**(logM)
        HMF_lgM.append(HMF_Colossus(10**logM, current_redshift)* np.log(10)*M)  

    plt.axvline(100*mass_resolution, color='black', linestyle='--')
    plt.loglog(bins[:-1], number_density, marker='.')
    plt.plot(10**(logM_list),HMF_lgM,color='red',linestyle='-')
    plt.title(f'Host Halo Mass Function, z={current_redshift:.2f}')
    plt.xlabel('Mass [$M_\odot/h$]')
    plt.ylabel(r'$\frac{dN}{d\log_{10}M}$ [$(Mpc/h)^{-3}$]')
    plt.tight_layout()
    plt.savefig(filename)
    

def plot_DF_heating_per_logM_comparison(ds_group,logM_bins,subhalo_DF_heating_hostmassbin,hosthalo_DF_heating_hostmassbin,filename):

    logM_bin_width = logM_bins[1] - logM_bins[0]
    logM_bin_centers = (logM_bins[:-1] + logM_bins[1:]) / 2
    volume = ds_group.domain_width.in_units('Mpc/h').prod()
    subhalo_DF_heating_hostmassbin_perV_perBinsize = subhalo_DF_heating_hostmassbin/logM_bin_width/volume.value
    hosthalo_DF_heating_hostmassbin_perV_perBinsize = hosthalo_DF_heating_hostmassbin/logM_bin_width/volume.value

    print("subhalo DF_heating max: ",max(subhalo_DF_heating_hostmassbin_perV_perBinsize))
    print("hosthalo DF_heating max: ",max(hosthalo_DF_heating_hostmassbin_perV_perBinsize))
 
       
    # Plot DF heating as a function of host halo mass bin
    fig = plt.figure(facecolor='white')
    plt.plot(logM_bin_centers, 1e7*subhalo_DF_heating_hostmassbin_perV_perBinsize,'r-',label='Legacy l100n2048 subhalo')
    plt.plot(logM_bin_centers, 1e7*hosthalo_DF_heating_hostmassbin_perV_perBinsize,'b-',label='Legacy l100n2048 host halo')

    #check contribution to heating (analytical result)
    z_value = ds_group.current_redshift
    logM_limits = [2, 16]  # Limits for log10(M [Msun/h])
    ln_m_over_M_limits = [-12, 0]  # Limits for m/M

    logM_list = np.linspace(logM_limits[0], logM_limits[1],57)


    DF_heating_perlogM = []
    for logM in logM_list:
        result, error = quad(integrand, ln_m_over_M_limits[0], ln_m_over_M_limits[1], args=(logM, z_value))
        if (abs(error) > 0.01 * abs(result)):
            print("Possible large integral error at z = %f, relative error = %f\n", z_value, error/result)

        DF_heating_perlogM.append(result)

    DF_heating_perlogM = np.array(DF_heating_perlogM)
    plt.plot(logM_list,1e7*DF_heating_perlogM,'g-',label='analytic')
    plt.legend()
    plt.xlim([4,12])
    plt.ylim([1e37,1e43])
    plt.yscale('log')
    plt.ylabel(r'DF heating per logM [erg/s (Mpc/h)$^{-3}$]',fontsize=12)
    plt.xlabel('logM [Msun/h]',fontsize=12)
    plt.savefig(filename,dpi=300)


    #compare total heating rate
    subhalo_DF_heating_total_legacy = subhalo_DF_heating_hostmassbin.sum()/volume.value
    hosthalo_DF_heating_total_legacy = hosthalo_DF_heating_hostmassbin.sum()/volume.value

    print("subhalo_DF_heating_total_legacy: ",subhalo_DF_heating_total_legacy)
    print("hosthalo_DF_heating_total_legacy: ",hosthalo_DF_heating_total_legacy)

    DF_heating_total_analytic, error = nquad(integrand, [[ln_m_over_M_limits[0], ln_m_over_M_limits[1]], [logM_limits[0], logM_limits[1]]], args=(z_value,))

    if (abs(error) > 0.01 * abs(DF_heating_total_analytic)):
        print("possible large integral error at z = %f, relative error = %f\n",z_value,error/DF_heating_total_analytic)
    print("subhalo DF_heating_total_analytic: ",DF_heating_total_analytic)
    print("ratio: ",subhalo_DF_heating_total_legacy/DF_heating_total_analytic)
    

def plot_Mratio_cumulative(sub_host_Mratio,filename):
    if(np.max(sub_host_Mratio) > 1):
        print("Warning: m/M ratio > 1")
        bins = np.linspace(0,np.max(sub_host_Mratio),100)
    else:
        bins = np.linspace(0,1,100)

    # Calculate the histogram
    counts, bin_edges = np.histogram(sub_host_Mratio, bins=bins)

    cumulative_counts = np.cumsum(counts[::-1])[::-1]
    # Plot the cumulative histogram
    fig = plt.figure(facecolor='white') 
    scaled_counts = cumulative_counts * bin_edges[:-1]
    plt.plot(bin_edges[:-1],scaled_counts)
    plt.title(f'Subhalo m/M Distribution, z={current_redshift:.2f}')
    plt.xlabel(r'$\psi$ = m/M')
    plt.ylabel(r'$\psi$ * N(> $\psi$)')
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename,dpi=300)

#x  = m/M
#dN/dlgx = A* x**(-alpha) exp(-beta x**omega)
#lg[dN/dlgx] = lgA - alpha lgx - beta x**omega / ln(10)
def fitFunc_lg_dNdlgx(lgx,alpha,beta_ln10, omega, lgA):
    x = 10**lgx
    return lgA - alpha*lgx - beta_ln10*x**omega 

def plot_Mratio_dN_dlogMratio(All_sub_host_M,dark_matter_resolution,filename):
    print("total number of subhalos: ",len(All_sub_host_M))
    All_sub_host_Mratio = All_sub_host_M[:,0]
    All_host_M = All_sub_host_M[:,1]
    All_host_index = All_sub_host_M[:,2]
    All_host_logM = np.log10(All_host_M)
    #divide the host halos into 5 mass bins, and plot the distribution of m/M for each bin respectively
    logM_min = np.min(All_host_logM)
    logM_max = np.max(All_host_logM)
    num_M_bins = 5
    logM_bins = np.linspace(logM_min, logM_max, num=num_M_bins+1)
    sub_host_Mratio_list = []
    num_host_list = []
    for i in range(num_M_bins):
        mask = (All_host_logM >= logM_bins[i]) & (All_host_logM < logM_bins[i+1])
        sub_host_Mratio_list.append(All_sub_host_Mratio[mask])
        host_index = All_host_index[mask]
        num_host = len(set(host_index))
        num_host_list.append(num_host)
        print(f"number of host halos in bin {i}: {num_host}")
    tot_num_host = len(set(All_host_index))
    print(f"total number of host halos: {tot_num_host}")
    num_host_list.append(tot_num_host)

    #threshold for small halos
    critical_ratio_list = []
    subhalo_resolution = 50*dark_matter_resolution
    for i in range(num_M_bins):
        critical_ratio = subhalo_resolution/10**logM_bins[i]
        critical_ratio_list.append(critical_ratio)
    
    colors = ['r','orange','y','g','b']
    labels = [f'[{logM_bins[i]:.2f}, {logM_bins[i+1]:.2f}]' for i in range(num_M_bins)]

    colors = np.append(colors,'black')
    labels.append('All')
    sub_host_Mratio_list.append(All_sub_host_Mratio)   

    bins = np.linspace(-5,0,50)
    log_bin_widths = bins[1] - bins[0]
    min_number_density = 1e10

    number_density_list = []
    fig = plt.figure(facecolor='white')
    for i in range(num_M_bins+1):
        counts, bin_edges = np.histogram(np.log10(sub_host_Mratio_list[i]), bins=bins)
        counts = np.append(counts,counts[-1])
        number_density = counts/log_bin_widths
        #divide by num of host halos
        print("number of host halos: ",num_host_list[i])
        if num_host_list[i] == 0:
            continue
        number_density /= num_host_list[i]

        #min nonzero number density
        min_number_density = min(min_number_density,np.min(number_density[number_density > 0])) 
        #exclude zero counts, set to an artificial small number
        mask = number_density == 0
        artificial_small = 1e-10
        number_density[mask] = artificial_small
        plt.step(bin_edges, number_density, where='post',color=colors[i],label=labels[i])
        if (i< num_M_bins):
            number_density_list.append(number_density)
            plt.axvline(np.log10(critical_ratio_list[i]),color = colors[i],linestyle='--')

        print("sum counts: ",np.sum(counts))
        #plt.hist(np.log10(sub_host_Mratio_list[i]), bins=bins, histtype='step',color=colors[i],label=labels[i])
    
    #plot the initial guess fitting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    p_guess = [0.86, 50/np.log(10),4 ,np.log10(0.065)]
    fit_lg_number_density = fitFunc_lg_dNdlgx(bin_centers,*p_guess)
    plt.plot(bin_centers,10**fit_lg_number_density,linestyle='-',color='grey',label='van den Bosch+ 2016')

    plt.ylim(bottom=min_number_density/10)
    plt.legend(loc='lower left')
    plt.title(f'Subhalo m/M Distribution, z={current_redshift:.2f}')
    plt.xlabel(r'log10($\psi$) = log10(m/M)')
    plt.ylabel(r'$\frac{dN}{d\log_{10}(\psi)}$')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(filename,dpi=300)

    #use >resolution data to fit the distribution
    fig = plt.figure(facecolor='white')

    for i in range(3,5):
        number_density = number_density_list[i]
        fit_mask = (bin_centers > np.log10(critical_ratio_list[i])) & (number_density[:-1] != artificial_small)
        #fitFunc_lg_dNdlgx(lgx,alpha,beta_ln10, omega, lgA)
        #p_guess = [0.86, 50/np.log(10),4 ,np.log10(0.065)]
        popt, pcov = curve_fit(fitFunc_lg_dNdlgx, bin_centers[fit_mask], np.log10(number_density[:-1][fit_mask]), p0=p_guess)
        print("fit parameters: ",popt)
        fit_lg_number_density = fitFunc_lg_dNdlgx(bin_centers[fit_mask],*popt)
        plt.step(bin_edges, np.log10(number_density), where='post',color=colors[i],label=labels[i])
        plt.plot(bin_centers[fit_mask],fit_lg_number_density,linestyle='-.',color=colors[i])
        plt.axvline(np.log10(critical_ratio_list[i]),color = colors[i],linestyle='--')

        param_text = r'$\alpha$: {:.2f} $\beta/\ln10$: {:.1f} $\omega$: {:.1f} lgA: {:.1f}'.format(popt[0], popt[1], popt[2], popt[3])
        plt.text(-4.5, -3+i/4, param_text, fontsize=10, color=colors[i])
    
    #plot the initial guess fitting
    p_guess = [0.86, 50/np.log(10),4 ,np.log10(0.065)]
    fit_lg_number_density = fitFunc_lg_dNdlgx(bin_centers,*p_guess)

    plt.plot(bin_centers,fit_lg_number_density,linestyle='-',color='grey',label='van den Bosch+ 2016')
    plt.ylim(bottom=np.log10(min_number_density/10))
    plt.xlabel(r'log10($\psi$) = log10(m/M)')
    plt.ylabel(r'log10[$\frac{dN}{d\log_{10}(\psi)}$]')
    plt.legend(loc='lower left')
    plt.savefig(filename.replace('.png','_fit.png'),dpi=300)


if __name__ == "__main__":

    simulation = 'l10n1024'
    path_to_catalog = '/home/zwu/21cm_project/legacy/GVD_C700_'+simulation+'_SLEGAC/dm_gadget/'
    output_dir ='/home/zwu/21cm_project/compare_legacy/results_'+simulation+'/rockstar/'
    path_to_haloh5 = path_to_catalog + 'halos_h5/'
    path_to_rockstar = path_to_catalog + 'rockstar/'




    # Load the halo and subhalo catalogs using yt
    # rockstar_logfile = path_to_catalog + 'outrockstar_434756.log'
    # subhalofile = yt.load(path_to_catalog + 'out_000_subhalo.list')
    
    #subhalofile = path_to_catalog + 'out_001_subhalo.list'
    # ds_subhalo = yt.load(subhalofile)  #yt does not support reading subhalo catalog

    #check hlist file names under path_to_haloh5 and get their redshifts
    #list files: hlist_...
    hlist_files = []
    z_list = []
    for file in os.listdir(path_to_haloh5):
        if file.startswith("hlist_") and file.endswith(".h5"):
            hlist_files.append(file)
            scale_factor = float(file.split('_')[1].split('.h5')[0])
            z_list.append(1/scale_factor-1)
    
    #find the closest file to a given redshift
    z_target = 18
    z_diff = np.abs(np.array(z_list)-z_target)
    z_idx = np.argmin(z_diff)
    print(f"select redshift = {z_list[z_idx]}")
    hlistfile = path_to_haloh5 + hlist_files[z_idx]


    print("reading halo catalog ......  from file:", hlistfile)
    #hlistfile = path_to_catalog + 'rockstar/hlists/hlist_0.047830.list'  #ascii file
    #hlistfile = path_to_catalog + 'halos_h5/hlist_0.047830.h5'  #hdf5 file
    #hlistfile = path_to_catalog + 'halos_h5/hlist_0.090580.h5'

    ds_hlist = yt.load(hlistfile)
    print("before correction: domain width =", ds_hlist.domain_width.in_units('Mpc/h'))
    ds_hlist.domain_width *= ds_hlist.scale_factor
    print("after correction: domain width =", ds_hlist.domain_width.in_units('Mpc/h'))
    print("after correction: domain width =", ds_hlist.domain_width.in_units('Mpccm/h'))

    #display_hdf5_contents(hlistfile)
    
    ad = ds_hlist.all_data()
    current_redshift = ds_hlist.current_redshift
    print("redshift = ", current_redshift)
    if simulation == 'l10n1024':
        mass_resolution = ds_hlist.quan(1.06e5, 'Msun').in_units('Msun/h')
    elif simulation == 'l100n2048':
        mass_resolution = ds_hlist.quan(1.32e7, 'Msun').in_units('Msun/h')
    else:
        print("simulation not supported")
        exit()
    
    #print(ds_hlist.field_list)
        
    #assign subhalos to their host halos
    num_halo = len(ad['halos', 'id'])
    print("total number of halos = ", num_halo)

    #select host halos (pid = -1)
    host_halo_mask = ad['halos', 'pid'] == -1
    host_ids = ad['halos', 'id'][host_halo_mask]
    N_host = len(ad['halos', 'id'][host_halo_mask])
    print("number of host halos = ", N_host)

    #select subhalos (pid > 0)
    subhalo_mask = ad['halos', 'pid'] > 0
    subhalo_ids = ad['halos', 'id'][subhalo_mask]
    subhalo_pids = ad['halos', 'pid'][subhalo_mask]
    N_sub = len(ad['halos', 'id'][subhalo_mask])
    print("number of subhalos = ", N_sub)

    print("total number of halos: ", N_host + N_sub)

    # Initialize lists to store subhalo indices for each host halo
    # length: N_host, each element is a list of subhalo indices (range from 0 to N_sub-1)
    Subhalo_Index_List = [[] for _ in range(len(host_ids))]

    # Loop through subhalos and assign them to their corresponding host halos
    for subhalo_index, subhalo_id, pid in zip(range(len(subhalo_ids)), subhalo_ids, subhalo_pids):
        host_index = np.where(host_ids == pid)[0]
        if len(host_index) > 0:
            Subhalo_Index_List[host_index[0]].append(subhalo_index)

                
    

    
    hmf_filename = output_dir + f'hmf_file_{z_idx}_z_{current_redshift:.2f}.png'
    plot_hmf(ad, ds_hlist, current_redshift, mass_resolution, host_halo_mask, hmf_filename)

    vx = ad['halos','vx']  #default unit in km/s 
    vy = ad['halos','vy']
    vz = ad['halos','vz'] 
    vel = np.sqrt(vx**2+vy**2+vz**2)

    #plot velocity histogram
    fig = plt.figure(facecolor='white')
    plt.hist(vel[host_halo_mask].value, bins=50, color='blue', alpha=0.7)
    plt.title(f'Host Halo Velocity Distribution, z={current_redshift:.2f}')
    plt.xlabel('Velocity [km/s]')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(output_dir + f'velocity_hist_{z_idx}_z_{current_redshift:.2f}.png')


    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Calculate the total DF heating for each host halo mass bin
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    z = current_redshift
    GroupMassList = ad['halos','Mvir'][host_halo_mask].in_units('Msun/h')

    # Define the bins for the host halo masses
    logM_min = np.log10(GroupMassList.min())
    logM_max = np.log10(GroupMassList.max())
    logM_bins = np.linspace(logM_min, logM_max, num=20)  
    logM_bin_width = logM_bins[1] - logM_bins[0]
    logM_bin_centers = (logM_bins[:-1] + logM_bins[1:]) / 2

    # Initialize an array to store the total DF heating for each bin
    subhalo_DF_heating_hostmassbin = np.zeros_like(logM_bin_centers)
    hosthalo_DF_heating_hostmassbin = np.zeros_like(logM_bin_centers)


    # Initialize a list to store the host halos for each bin
    hosts_in_bins = [[] for _ in range(len(logM_bin_centers))]

    # Loop over each host halo
    for host in range(len(GroupMassList)):
        # Determine which bin this host belongs to
        host_logM = np.log10(GroupMassList[host])
        bin_index = np.searchsorted(logM_bins, host_logM) - 1
        # Add the host to the corresponding bin
        hosts_in_bins[bin_index].append(host)


    # Now loop over each bin and each host in the bin
    subhalo_DF_heating_list = []
    hosthalo_DF_heating_list = []

    print("calculate DF heating ......")
    M_host = ad['halos','Mvir'][host_halo_mask].in_units('Msun/h').value
    Vx_host = ad['halos','vx'][host_halo_mask].in_units('m/s').value
    Vy_host = ad['halos','vy'][host_halo_mask].in_units('m/s').value
    Vz_host = ad['halos','vz'][host_halo_mask].in_units('m/s').value
    Vel_host = np.sqrt(Vx_host**2+Vy_host**2+Vz_host**2)
    M_sub = ad['halos','Mvir'][subhalo_mask].in_units('Msun/h').value
    Vx_sub = ad['halos', 'vx'][subhalo_mask].in_units('m/s').value
    Vy_sub = ad['halos', 'vy'][subhalo_mask].in_units('m/s').value
    Vz_sub = ad['halos', 'vz'][subhalo_mask].in_units('m/s').value
    Vel_sub = np.sqrt(Vx_sub**2+Vy_sub**2+Vz_sub**2)

    All_sub_host_M  = []
    All_sub_host_Vratio = []

    for i, hosts_in_bin in enumerate(hosts_in_bins):
        # Loop over each host halo in this bin
        print(f"bin {i}: {len(hosts_in_bin)} host halos")
        for host in hosts_in_bin:

            #get host halo mass and velocity
            M = M_host[host]

            #exclude unresolved halos
            min_num_part = 100
            if M < min_num_part*mass_resolution:
                continue

        
            #calculate host halo DF heating, unit: J/s
            #use global average gas density instead of 200*rho_b(z)
            rho_g =  rho_b0*(1+z)**3 *Msun/Mpc**3
            I_DF = 1.0
            eta = 1.0
            vel_host = Vel_host[host]
            hosthalo_DF_heating = eta * 4 * np.pi * (G_grav * M *Msun/h_Hubble) ** 2 / vel_host *rho_g *I_DF

            hosthalo_DF_heating_list.append((host,hosthalo_DF_heating))
            hosthalo_DF_heating_hostmassbin[i] += hosthalo_DF_heating
            #Cs_sound = np.sqrt(5/3*kB*1e3/mp)
            #print("Mach number: ",vel_host/Cs_sound)


            
            #Now calculate subhalo DF heating
            # Get the subhalos of this host
           
            # Exclude empty lists
            num_subs = len(Subhalo_Index_List[host])
            if num_subs == 0:
                continue
            

            subhalos_of_host = [(j, M_sub[j]) for j in Subhalo_Index_List[host]]
            

            #do not exclude the first subhalo
            # subhalos_of_host.sort(key=lambda x: x[1])
            # maxsub_index = subhalos_of_host[-1][0]
            # subhalo_DF_heating_list.append((maxsub_index,0.0))
            # subhalos_of_host = subhalos_of_host[:-1]
            
    
            # Loop over each subhalo and calculate subhalo DF heating
            for (subhalo_index, m) in subhalos_of_host:
                if(m < 50*mass_resolution):
                    continue
                #incorrect subhalo, print index
                if(m/M > 1):
                    print("\nsubhalo mass > host halo mass")
                    print("subhalo mass: ",m)
                    print("host halo mass: ",M)
                    print("# of particle in host halo:",int(M/mass_resolution.value))
                    print("subhalo index: ",subhalo_index)
                    print("host halo index: ",host)
                    print("subindex list of the host:",Subhalo_Index_List[host])

                    print("subhalo ID:",ad["halos","id"][subhalo_mask][subhalo_index])
                    print("subhalo PID: ",ad["halos","pid"][subhalo_mask][subhalo_index])
                    print("host ID: ",ad["halos","id"][host_halo_mask][host])
                    print("\n")
                    continue

                    # print(len(subhalo_mask))
                    # print(len(host_halo_mask))
                    # #find the global index (the array of all halos) of this subhalo and host halo
                    # #(the #subhalo_index of True in subhalo_mask)
                    # print("gloabl index of the subhalo: ",np.where(subhalo_mask)[0][subhalo_index])
                    # print("gloabl index of the host halo: ",np.where(host_halo_mask)[0][host])
                    # exit()
                    
                All_sub_host_M.append([m/M, M, host])

                # Use the same settings for Vel_Virial, rho_g, I_DF, and eta
                #use subhalo velocity ???
                vel_analytic = Vel_Virial(M/h_Hubble, z)

                vel_sub = Vel_sub[subhalo_index]
                All_sub_host_Vratio.append(vel_sub/vel_host)

                vel = np.sqrt((Vx_host[host]-Vx_sub[subhalo_index])**2 + (Vy_host[host]-Vy_sub[subhalo_index])**2 + (Vz_host[host]-Vz_sub[subhalo_index])**2)

                rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
                I_DF = 1.0
                eta = 1.0

                # Calculate DF_heating and add it to the total for this bin
                DF_heating =  eta * 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / vel *rho_g *I_DF

                subhalo_DF_heating_hostmassbin[i] += DF_heating
                subhalo_DF_heating_list.append((subhalo_index,DF_heating))
            #end of subhalo loop
        #end of host loop
    #end of host mass bin loop
 
    filename = output_dir+f"DF_heating_usevel_perlogM_comparison_hlist{z_idx}_z{current_redshift:.2f}.png"
    plot_DF_heating_per_logM_comparison(ds_hlist,logM_bins,subhalo_DF_heating_hostmassbin,hosthalo_DF_heating_hostmassbin,filename)

    #plot histogram of m/M ratios 
    All_sub_host_M = np.array(All_sub_host_M)
    All_sub_host_Mratio = All_sub_host_M[:,0]
    All_host_M = All_sub_host_M[:,1]


    plot_Mratio_cumulative(All_sub_host_Mratio,output_dir+f'Mratio_cumulative_snap{z_idx}_z{current_redshift:.2f}.png')

    plot_Mratio_dN_dlogMratio(All_sub_host_M,mass_resolution, output_dir+f'Average_Mratio_dN_dlogMratio_snap{z_idx}_z{current_redshift:.2f}.png')



    #velocity
    '''
    #plot cumulative distribution of x = vel_sub/vel_host (N(>x))
    fig = plt.figure(facecolor='white')

    # Convert to numpy array for easier manipulation
    All_sub_host_Vratio = np.array(All_sub_host_Vratio)
    # Calculate unique mass ratios and their occurrences
    unique_vel_ratios, counts = np.unique(All_sub_host_Vratio, return_counts=True)
    # Sort unique mass ratios
    sorted_unique_vel_ratios = np.sort(unique_vel_ratios)
    # Calculate cumulative counts
    cumulative_counts = np.cumsum(counts)
    # Calculate total number of occurrences
    total_occurrences = len(All_sub_host_Vratio)
    # Calculate the number of occurrences where mass ratio is greater than or equal to each value of mu
    N_gt_x = total_occurrences - np.searchsorted(sorted_unique_vel_ratios, sorted_unique_vel_ratios, side='right')

    # Normalize by total occurrences
    #N_gt_x = N_gt_x / total_occurrences

    # Plot
    # plt.plot(sorted_unique_vel_ratios, N_gt_x)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel('Velocity Ratio (x = v_sub/v_host)')
    # plt.ylabel('N(≥x)')
    # plt.title('Distribution of Velocity Ratio')
    # plt.grid(True)
    # plt.savefig(output_dir+f"velocity_ratio_distribution_hlist{z_idx}_z{current_redshift:.2f}.png")

    '''

    # Open the output file
    # with open('results_l100n2048/fields_h5.txt', 'w') as f:
    #     for field in ds_hlist.field_list:
    #         # Write the field to the file
    #         f.write(str(field) + '\n')

    #display_hdf5_contents(hlistfile)   

    # for field in ds_hlist.field_list:
    #     print(field)

    # print("\nhalos ID:")
    # print(ad['halos', 'id'][0:50])

    # print("\nhalos pid:  (ID of least massive host halo (-1 if distinct halo))")
    # print(ad['halos', 'pid'][0:50])

    # print("\nhalos upid: (Upid: ID of most massive host halo (different from Pid when the halo is within two or more larger halos))")
    # print(ad['halos', 'upid'][0:50])

    # print("\nhalos desc_id: (Descendant halo ID)")
    # print(ad['halos', 'desc_id'][0:50])

    # print("\nhalos mvir:")
    # print(ad['halos', 'Mvir'][0:50])

    










