import yt
from yt import YTQuantity
#import yt.extensions.legacy
import ytree
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from physical_constants import *
from scipy.special import gamma
from scipy.integrate import solve_ivp, quad
from scipy.integrate import nquad

def test_volume_unit(ds):
    volume = ds.domain_width.in_units('Mpc/h').prod()
    print("1/volume: ",(1/volume))
    print("1/volume in unit (Mpc/h)^(-3): ",(1/volume).in_units('h**3*Mpc**-3'))
    print("1/volume.value: ",1/volume.value)

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

    
def Subhalo_Mass_Function_ln(ln_m_over_M):
    m_over_M = np.exp(ln_m_over_M)
    f0 = 0.1
    beta = 0.3
    gamma_value = 0.9
    x = m_over_M/beta
    return f0/(beta*gamma(1 - gamma_value)) * x**(-gamma_value) * np.exp(-x)

def integrand(ln_m_over_M, logM, z):
    global G_grav,rho_b0,h_Hubble, Mpc, Msun
    
    eta = 1.0
    I_DF = 1.0
    
    M = 10**logM
    m_over_M = np.exp(ln_m_over_M)
    m = m_over_M * M  
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    DF_heating =  eta * 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Vel_Virial(M/h_Hubble, z) *rho_g *I_DF
    
    DF_heating *= Subhalo_Mass_Function_ln(ln_m_over_M) 
    DF_heating *= HMF_Colossus(M,z) * np.log(10)*M   #convert from M to log10(M)
    
    return DF_heating


#plot functions

def plot_hmf(group_data, ds_group, current_redshift,filename):

    # Convert mass to 'Msun/h'
    group_mass_solar = group_data['Group', 'GroupMass'].in_units('Msun/h')

    # Create a histogram (logarithmic bins and logarithmic mass)
    bins = np.logspace(np.log10(min(group_mass_solar)), np.log10(max(group_mass_solar)), num=50)
    hist, bin_edges = np.histogram(group_mass_solar, bins=bins)

    # Convert counts to number density
    volume = ds_group.domain_width.in_units('Mpc/h').prod() 
    log_bin_widths = np.diff(np.log10(bins))
    number_density = hist / volume.value / log_bin_widths

    # Plot the mass function
    
    fig = plt.figure(facecolor='white')
    mass_resolution = ds_group.quan(1.06e5, 'Msun').in_units('Msun/h')

    cosmology.setCosmology('planck18')
    logM_limits = [5, 11]  # Limits for log10(M [Msun/h])
    HMF_lgM = []
    logM_list = np.linspace(logM_limits[0], logM_limits[1],57)
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
    
def plot_subhalo_histogram(ad, ds_group):
    #read subhalo data in the group file
    # Extract the number of subhalos for each group
    num_subhalos = ad['Group', 'GroupNsubs']

    # Convert min and max to integers
    min_subhalos = int(num_subhalos.min())
    max_subhalos = int(num_subhalos.max())

    fig = plt.figure(facecolor='white')
    # Create a histogram of the number of subhalos
    hist, bins = np.histogram(num_subhalos, bins=range(min_subhalos, max_subhalos + 1))

    # Plot the histogram
    plt.bar(bins[:-1], hist, width=1)
    plt.xlabel('Number of Subhalos')
    plt.ylabel('Number of Groups')
    #plt.savefig("subhalo_number_histogram.png",dpi=300)

    # Write the histogram data to a text file
    # np.savetxt('subhalo_number_histogram.txt', np.column_stack((bins[:-1], hist)), fmt='%d', header='Number_of_Subhalos Number_of_Groups')    

def plot_subhalo_mass_for_a_group(ad, group_id):
    # Extract the masses of its subhalos
    first_sub = int(ad['Group', 'GroupFirstSub'][group_id])
    num_subs = int(ad['Group', 'GroupNsubs'][group_id])
    subhalo_masses = ad['Subhalo', 'SubhaloMass'][first_sub:first_sub + num_subs]

    # Calculate m/M for each subhalo
    group_mass = ad['Group', 'GroupMass'][group_id]
    mass_ratio = subhalo_masses / group_mass

    fig = plt.figure(facecolor='white')
    # Create a histogram of the mass ratios
    plt.hist(mass_ratio, bins=np.linspace(min(mass_ratio), max(mass_ratio), 10))
    plt.xlabel('ln(Subhalo Mass / Group Mass)')
    plt.ylabel('Number of Subhalos in Group id = {}'.format(group_id))
    #plt.savefig("subhalo_mass_ratio_histogram.png",dpi=300)


    # Sort the mass ratios and exclude the most massive subhalo
    sorted_mass_ratios = np.sort(mass_ratio)
    mass_ratio_without_max = sorted_mass_ratios[:-1]
    
    fig = plt.figure(facecolor='white')
    # Create a histogram of the mass ratios
    plt.hist(np.log(np.array(mass_ratio_without_max)), bins=np.linspace(np.log(float(min(mass_ratio_without_max))), np.log(float(max(mass_ratio_without_max))), 20))
    plt.xlabel('ln(Subhalo Mass / Group Mass)')
    plt.ylabel('Number of Subhalos in Group id = {}'.format(group_id))
    #plt.savefig("subhalo_mass_ratio_histogram_without_max.png", dpi=300)


    # Create a histogram of the mass ratios
    #fig = plt.figure(facecolor='white')
    hist, bins = np.histogram(np.log(mass_ratio_without_max), bins=np.linspace(np.log(min(mass_ratio_without_max)), np.log(max(mass_ratio_without_max)), 20))

    # Calculate the bin widths
    bin_widths = np.diff(bins)

    # Calculate dn/dln(m/M)
    density = hist / bin_widths

    #Plot dn/dln(m/M)
    fig = plt.figure(facecolor='white')
    plt.plot(bins[:-1], density,linestyle='-')
    # plt.xlabel('ln(Subhalo Mass / Group Mass)')
    # plt.ylabel(r'$\frac{\Delta n}{\Delta \ln(m/M)}$')
    # plt.savefig("smf_without_max.png", dpi=300)

    #plot analytic subhalo mass function
    sub_host_ratio = np.linspace(1e-5,1,100)
    subHMF = np.array([Subhalo_Mass_Function(m_over_M) for m_over_M in sub_host_ratio])
    plt.plot(np.log(sub_host_ratio), subHMF,'r-')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig("compare_smf.png", dpi=300)


def plot_subhalo_mass_fraction_distribution(ad):
        # Initialize an empty list to store the mass fractions
    mass_fractions_without_max = []

    # Loop over all groups
    for group_id in range(len(ad['Group', 'GroupNsubs'])):
        # Extract the masses of the subhalos
        first_sub = int(ad['Group', 'GroupFirstSub'][group_id])
        num_subs = int(ad['Group', 'GroupNsubs'][group_id])
        subhalo_masses = ad['Subhalo', 'SubhaloMass'][first_sub:first_sub + num_subs]

        # Sort the subhalo masses and exclude the most massive subhalo
        sorted_subhalo_masses = np.sort(subhalo_masses)
        subhalo_masses_without_max = sorted_subhalo_masses[:-1]

        # Calculate the group mass
        group_mass = ad['Group', 'GroupMass'][group_id]

        # Calculate the mass fraction of the subhalos and add it to the list
        mass_fraction_without_max = np.sum(subhalo_masses_without_max) / group_mass
        mass_fractions_without_max.append(mass_fraction_without_max)

    # Create a histogram of the mass fractions
    fig = plt.figure(facecolor='white')
    plt.hist(mass_fractions_without_max, bins=50)
    plt.xlabel('Subhalo Mass Fraction (excluding most massive subhalo)')
    plt.ylabel('Number of Groups')
    #plt.savefig("subhalo_mass_fraction_without_max_histogram.png", dpi=300)
    

def plot_subhalo_mass_fraction_hostmassbin(ad,filename):
    GroupMassList = ad['Group', 'GroupMass'].in_units('Msun/h')
    # Define the bins for the host halo masses
    logM_min = np.log10(GroupMassList.min())
    logM_max = np.log10(GroupMassList.max())
    logM_bins = np.linspace(logM_min, logM_max, num=20)  
    logM_bin_width = logM_bins[1] - logM_bins[0]
    logM_bin_centers = (logM_bins[:-1] + logM_bins[1:]) / 2

    subhalo_mass_fraction = np.zeros_like(logM_bin_centers)

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
    for i, hosts_in_bin in enumerate(hosts_in_bins):
        # Loop over each host halo in this bin
        m_sum_thisbin = 0.0
        M_sum_thisbin = 0.0
        for host in hosts_in_bin:
            # Get the subhalos of this host
            first_sub = int(ad['Group', 'GroupFirstSub'][host])
            num_subs = int(ad['Group', 'GroupNsubs'][host])
            subhalos_of_host_masslist = ad['Subhalo', 'SubhaloMass'][first_sub:first_sub + num_subs].in_units('Msun/h')

            # Exclude the most massive subhalo
            subhalos_of_host_masslist.sort()
            subhalos_of_host_masslist = subhalos_of_host_masslist[:-1]

            #Calculate M and m
            M = ad['Group', 'GroupMass'][host].in_units('Msun/h')        
            m_sum_this_Group  = subhalos_of_host_masslist.sum()
    
            m_sum_thisbin += m_sum_this_Group
            M_sum_thisbin += M
        sub_fraction_thisbin = m_sum_thisbin/M_sum_thisbin
        subhalo_mass_fraction[i] = sub_fraction_thisbin 

    fig = plt.figure(facecolor='white')
    plt.bar(logM_bin_centers, subhalo_mass_fraction, width=logM_bin_width, align='center')
    plt.xlim([4,12])
    plt.ylim([0,0.5])
    plt.title('Legacy l10n1024')
    plt.ylabel('subhalo mass fraction',fontsize=12)
    plt.xlabel('logM [Msun/h]',fontsize=12)
    plt.savefig(filename,dpi=300) 


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
    plt.plot(logM_bin_centers, 1e7*subhalo_DF_heating_hostmassbin_perV_perBinsize,'r-',label='Legacy l10n1024 subhalo')
    plt.plot(logM_bin_centers, 1e7*hosthalo_DF_heating_hostmassbin_perV_perBinsize,'b-',label='Legacy l10n1024 host halo')

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
    

def write_heating_to_file(ds_group,ad,hosthalo_DF_heating_sorted,subhalo_DF_heating_sorted,filename):
    #write subhalo DF heating to file
    with h5py.File(filename, 'w') as f:
        header = f.create_group("Header") 
        header_attrs_list = ['BoxSize','FlagDoubleprecision','HubbleParam','Ngroups_Total','Nsubgroups_Total','Omega0','OmegaLambda','Redshift','Time']
        for attr in header_attrs_list:
            header.attrs[attr] = ds_group.parameters[attr]

        group = f.create_group("Group")
        group.create_dataset("GroupMass", data=ad['Group', 'GroupMass'].in_units('Msun/h'))
        group.create_dataset("Group_R_Crit200", data=ad['Group', 'Group_R_Crit200'])
        group.create_dataset("GroupPos_0", data=ad['Group', 'GroupPos_0'].in_units('Mpccm/h'))
        group.create_dataset("GroupPos_1", data=ad['Group', 'GroupPos_1'].in_units('Mpccm/h'))
        group.create_dataset("GroupPos_2", data=ad['Group', 'GroupPos_2'].in_units('Mpccm/h'))
        group.create_dataset("GroupVel_0", data=ad['Group', 'GroupVel_0'].in_units('km/s'))
        group.create_dataset("GroupVel_1", data=ad['Group', 'GroupVel_1'].in_units('km/s'))
        group.create_dataset("GroupVel_2", data=ad['Group', 'GroupVel_2'].in_units('km/s'))
        group.create_dataset("GroupFirstSub", data=ad['Group', 'GroupFirstSub'])
        group.create_dataset("GroupNsubs", data=ad['Group', 'GroupNsubs'])
        group.create_dataset("GroupDFHeating", data=hosthalo_DF_heating_sorted)

        subhalo = f.create_group("Subhalo")
        subhalo.create_dataset("SubhaloMass", data=ad['Subhalo', 'SubhaloMass'].in_units('Msun/h'))
        subhalo.create_dataset("SubhaloHalfmassRad", data=ad['Subhalo', 'SubhaloHalfmassRad'].in_units('Mpccm/h'))
        subhalo.create_dataset("SubhaloGrNr", data=ad['Subhalo', 'SubhaloGrNr'])
        subhalo.create_dataset("SubhaloPos_0", data=ad['Subhalo', 'SubhaloPos_0'].in_units('Mpccm/h'))
        subhalo.create_dataset("SubhaloPos_1", data=ad['Subhalo', 'SubhaloPos_1'].in_units('Mpccm/h'))
        subhalo.create_dataset("SubhaloPos_2", data=ad['Subhalo', 'SubhaloPos_2'].in_units('Mpccm/h'))
        subhalo.create_dataset("SubhaloVel_0", data=ad['Subhalo', 'SubhaloVel_0'].in_units('km/s'))
        subhalo.create_dataset("SubhaloVel_1", data=ad['Subhalo', 'SubhaloVel_1'].in_units('km/s'))
        subhalo.create_dataset("SubhaloVel_2", data=ad['Subhalo', 'SubhaloVel_2'].in_units('km/s'))
        subhalo.create_dataset("SubhaloDFHeating", data=subhalo_DF_heating_sorted)






#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#                                           main program
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

if __name__ == "__main__":

    print(yt.__version__)
    print(sys.executable)
    cosmology.setCosmology('planck18')



    # simulation_set = 'GVD_C700_l10n1024_SLEGAC'  #group_129 - 323
    simulation_set = 'GVD_C700_l100n256_SLEGAC'  #groups_063 - 299
    # simulation_set = 'GVD_C700_l1600n256_SLEGAC'   #groups_216 - 299

    output_dir = "/home/zwu/21cm_project/compare_legacy/results_l100n256/subfind/debug/"

    filedir = "/home/zwu/21cm_project/legacy/"+simulation_set+"/dm_gadget/"
    redshift_file = '/home/zwu/21cm_project/compare_legacy/group_redshifts_'+simulation_set+'.txt'
    redshifts_index = np.loadtxt(redshift_file,skiprows=1)
    snapnums = redshifts_index[:,0].astype(int)
    redshifts = redshifts_index[:,1]

    #group_filenumber_list = [135,156,171,192]
    #find file index closet to the given redshifts
    target_z_list = [20, 15, 12, 0]
    group_filenumber_list = []
    for target_z in target_z_list:
        idx = (np.abs(redshifts - target_z)).argmin()
        group_filenumber_list.append(snapnums[idx])
    
    group_filenumber = str(group_filenumber_list[3]).zfill(3)

    file_group_0 = filedir + "data/groups_"+group_filenumber+"/fof_subhalo_tab_"+group_filenumber+".0.hdf5"


    display_hdf5_contents(file_group_0)

    #read group data
    print("\n\nexample: load groups:")
    ds_group = yt.load(file_group_0)
    ad = ds_group.all_data()

    # for field in ds_group.field_list:
    #     print(field)

    current_redshift = ds_group.current_redshift
    print("current redshift: ",current_redshift)
    
    print(len(ad['Group','GroupMass']))
    #print(len(ad['Subhalo','SubhaloMass']))

    #plot_hmf(ad, ds_group, current_redshift,output_dir+"hmf_l10n1024_z15.png")
    #plot_subhalo_histogram(ad, ds_group)

    # Find the group with the most subhalos
    #group_id = np.argmax(ad['Group', 'GroupNsubs'])
    #print("Group id = {} has the most subhalos".format(group_id))
    #plot_subhalo_mass_for_a_group(ad, group_id)
 
    # mass fraction in subhalos
    #plot_subhalo_mass_fraction_without_max_distribution(ad)
    #plot_subhalo_mass_fraction_hostmassbin(ad,output_dir+"subhalo_mass_fraction_hostmassbin_z15.png")


    

    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    # Calculate the total DF heating for each host halo mass bin
    #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    z = current_redshift
    GroupMassList = ad['Group', 'GroupMass'].in_units('Msun/h')

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
    vel_host_list = []

    for i, hosts_in_bin in enumerate(hosts_in_bins):
        # Loop over each host halo in this bin
        for host in hosts_in_bin:

            #get host halo mass and velocity
            M = ad['Group', 'GroupMass'][host].in_units('Msun/h').value
            group_vel0 = ad['Group', 'GroupVel_0'][host].in_units('m/s').value
            group_vel1 = ad['Group', 'GroupVel_1'][host].in_units('m/s').value
            group_vel2 = ad['Group', 'GroupVel_2'][host].in_units('m/s').value

            #calculate host halo DF heating, unit: J/s
            #use global average gas density instead of 200*rho_b(z)
            rho_g =  rho_b0*(1+z)**3 *Msun/Mpc**3
            I_DF = 1.0
            eta = 1.0
            vel_host = np.sqrt(group_vel0**2+group_vel1**2+group_vel2**2)
            hosthalo_DF_heating =  eta * 4 * np.pi * (G_grav * M *Msun/h_Hubble) ** 2 / vel_host *rho_g *I_DF
            vel_host_list.append(vel_host)

            hosthalo_DF_heating_list.append((host,hosthalo_DF_heating))
            hosthalo_DF_heating_hostmassbin[i] += hosthalo_DF_heating
            #Cs_sound = np.sqrt(5/3*kB*1e3/mp)
            #print("Mach number: ",vel_host/Cs_sound)


            
            #Now calculate subhalo DF heating
            # Get the subhalos of this host
           
            first_sub = int(ad['Group', 'GroupFirstSub'][host])
            num_subs = int(ad['Group', 'GroupNsubs'][host])
            # Exclude the most massive subhalo
            if (num_subs == 0):
                continue
            subhalos_of_host = [(j, ad['Subhalo', 'SubhaloMass'][j].in_units('Msun/h')) for j in range(first_sub, first_sub + num_subs)]
            


            subhalos_of_host.sort(key=lambda x: x[1])
            maxsub_index = subhalos_of_host[-1][0]
            #heating = 0 for the most massive subhalo as it's not a real subhalo
            subhalo_DF_heating_list.append((maxsub_index,0.0))

            subhalos_of_host = subhalos_of_host[:-1]
            
    

            # Loop over each subhalo and calculate subhalo DF heating
            for (subhalo_index, subhalo_mass) in subhalos_of_host:
                m = subhalo_mass.value

                # Use the same settings for Vel_Virial, rho_g, I_DF, and eta
                #use subhalo velocity ???
                vel_analytic = Vel_Virial(M/h_Hubble, z)
                sub_vel0 = ad['Subhalo', 'SubhaloVel_0'][subhalo_index].in_units('m/s').value
                sub_vel1 = ad['Subhalo', 'SubhaloVel_1'][subhalo_index].in_units('m/s').value
                sub_vel2 = ad['Subhalo', 'SubhaloVel_2'][subhalo_index].in_units('m/s').value
                vel = np.sqrt((group_vel0-sub_vel0)**2+(group_vel1-sub_vel1)**2+(group_vel2-sub_vel2)**2)

                rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
                I_DF = 1.0
                eta = 1.0

                # Calculate DF_heating and add it to the total for this bin
                DF_heating =  eta * 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / vel *rho_g *I_DF

                subhalo_DF_heating_hostmassbin[i] += DF_heating
                subhalo_DF_heating_list.append((subhalo_index,DF_heating))

 
    filename = output_dir+f"DF_heating_usevel_perlogM_comparison_groups{group_filenumber}_z{current_redshift:.2f}.png"
    plot_DF_heating_per_logM_comparison(ds_group,logM_bins,subhalo_DF_heating_hostmassbin,hosthalo_DF_heating_hostmassbin,filename)


    #plot histogram of host vel
    vel_host_list = np.array(vel_host_list)
    fig = plt.figure(facecolor='white')
    plt.hist(vel_host_list/1e3, bins=50)
    plt.title(f'Host Halo Velocity Distribution, z={current_redshift:.2f}')
    plt.xlabel('Host Halo Velocity [km/s]')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(output_dir+f'Host_vel_groups{group_filenumber}_z{current_redshift:.2f}.png',dpi=300)

    
    #rearange subhalo_DF_heating_list according to subhalo index
    subhalo_DF_heating_list.sort(key=lambda x: x[0])
    subhalo_DF_heating_sorted = np.array([x[1] for x in subhalo_DF_heating_list])
    #print(len(subhalo_DF_heating_list))

    #rearange hosthalo_DF_heating_list according to hosthalo index
    hosthalo_DF_heating_list.sort(key=lambda x: x[0])
    hosthalo_DF_heating_sorted = np.array([x[1] for x in hosthalo_DF_heating_list])
    #print(len(hosthalo_DF_heating_list))

    #heating_filename =output_dir+ 'subhalo_DF_heating_'+simulation_set+group_filenumber+'.hdf5'
    heating_filename =output_dir+ 'subhalo_DF_heating_usevel_'+simulation_set+group_filenumber+'.hdf5'


    #write_heating_to_file(ds_group,ad,hosthalo_DF_heating_sorted, subhalo_DF_heating_sorted,heating_filename)
    
