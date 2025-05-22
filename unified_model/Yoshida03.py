import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator

from colossus.cosmology import cosmology
from physical_constants import h_Hubble, H0_s, Omega_m, Omega_b, Myr, cosmo, Msun
from HaloProperties import Temperature_Virial_analytic, get_gas_lognH_analytic, \
inversefunc_Temperature_Virial_analytic, get_mass_density_analytic
from Grackle_cooling import run_constdensity_model
from Analytic_Model import integrate_SHMF_heating_for_single_host

def get_Hubble_timescale_Tegmark97(z):
    #return Hubble timescale in Myr, modify z to 1+z for accuracy at low z
    #Tegmark 1997 Eq. 10

    return 6.5/h_Hubble * ((1+z)/100)**(-1.5)


def get_nH_Tegmark97(z):
    #Tegmark 1997 Eq. 1, modify z to 1+z for accuracy at low z
    #nH in cm^-3
    return 23*((1+z)/100)**3*h_Hubble**2*Omega_b/0.015

def get_fH2_Tegmark97(T, z=None):
    #H2 mass fraction for Tvir
    #Tegmark 1997 Eq.17
    #T in K, return fH2
    if z is None:
        high_order_term = 0.0
    else:
        n = get_nH_Tegmark97(z)
        high_order_term = 7.4e8/n*(1+z)**2.13*np.exp(-3173/(1+z))
        # print("high_order_term = ", high_order_term)
    
    return 3.5e-4 * (T/1e3)**1.52*(1+high_order_term)**(-1)

def get_critical_fH2_Tegmark97(T, z):
    #where cooling time = Hubble time(z), modify z to 1+z for accuracy at low z
    #Eq.11 Tegmark 1997
    T3 = T/1e3
    fc = 0.00016/(h_Hubble*Omega_b/0.03) * ((1+z)/100)**(-1.5) / (1+10*T3**3.5/(60+T3**4)) *np.exp(512/T)
    return fc
    
def get_Tvir_Tegmark97(M, z):
    #M in Msun, Tegmark Eq. 28
    return 485*h_Hubble**(2/3) * (M/1e4)**(2/3) * (1+z)/100


def get_Tvir_Yoshida03(M, z, mean_molecular_weight):
    #M in Msun
    Delta = 200
    T =  1.98e4*(mean_molecular_weight/0.6)*(M*h_Hubble/1e8)**(2/3) \
        *(Omega_m*Delta/18/np.pi**2)**(1/3)*(1+z)/10
    return T

def get_fH2_Yoshida03(T):
    
    T3 = T/1e3
    fH2 = 4.7e-5 * T3**1.52
    return fH2

def test_Tvir():
    #compare Tvir from Yoshida03, Tegmark97 and our model
    z = 15
    M_list = np.logspace(4, 8, 50)
    Tvir_Yoshida = np.array([get_Tvir_Yoshida03(M, z, 1.0) for M in M_list])
    Tvir_Tegmark = np.array([get_Tvir_Tegmark97(M, z) for M in M_list])
    Tvir_analytic = np.array([Temperature_Virial_analytic(M, z, mean_molecular_weight=1.0) for M in M_list])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(M_list, Tvir_Yoshida, label='Yoshida 2003', color='blue')
    ax.plot(M_list, Tvir_Tegmark, label='Tegmark 1997', color='red')
    ax.plot(M_list, Tvir_analytic, label='our model, mu = 1.0', color='green')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    ax.set_xlabel('M [Msun]', fontsize=14)
    ax.set_ylabel('Tvir [K]', fontsize=14)
    ax.set_title(f'z = {z}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join("Analytic_results/Yoshida03", "compare_Tvir.png")
    plt.savefig(filename, dpi=300)
    

def compare_Hubble_timescale_and_nH():
    z_list = np.linspace(0, 30, 100)
    t_Hubble_Tegmark = get_Hubble_timescale_Tegmark97(z_list)
    t_Hubble_colossus = np.array([cosmo.age(z)*1e3 for z in z_list])
    # print("t_Hubble_Tegmark = ", t_Hubble_Tegmark)
    # print("t_Hubble_colossus = ", t_Hubble_colossus)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(z_list, t_Hubble_Tegmark, label='Tegmark 1997', color='blue')
    ax.plot(z_list, t_Hubble_colossus, label='Colossus', color='red')
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    ax.set_xlabel('z', fontsize=14)
    ax.set_ylabel('Hubble time [Myr]', fontsize=14)
    ax.set_title('Hubble time vs redshift', fontsize=16)
    plt.tight_layout()
    filename = os.path.join("Analytic_results/Yoshida03", "Hubble_time_vs_z.png")
    if not os.path.exists("Analytic_results/Yoshida03"):
        os.makedirs("Analytic_results/Yoshida03")
    plt.savefig(filename, dpi=300)

    #also compare nH in Tegmark 1997 and our model
    nH_Tegmark = get_nH_Tegmark97(z_list)
    lognH_analytic = get_gas_lognH_analytic(z_list)
    nH_analytic = 10**lognH_analytic
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(z_list, nH_Tegmark, label='Tegmark 1997', color='blue')
    ax.plot(z_list, nH_analytic, label='our model', color='red')
    ax.set_yscale('log')
    ax.legend()
    ax.grid()
    ax.set_xlabel('z', fontsize=14)
    ax.set_ylabel('nH [cm^-3]', fontsize=14)
    ax.set_title('nH vs redshift', fontsize=16)
    plt.tight_layout()
    filename = os.path.join("Analytic_results/Yoshida03", "nH_vs_z.png")
    plt.savefig(filename, dpi=300)


def test_fH2_Tegmark():
    z_list = np.array([100, 50, 25])
    nH_Tegmark = get_nH_Tegmark97(z_list)
    print("nH_Tegmark = ", nH_Tegmark)
    T_list = np.logspace(2, 4, 50)
    All_critical_fH2 = []
    All_H2_formation = []
    for z in z_list:
        fH2_critical = np.array([get_critical_fH2_Tegmark97(T, z) for T in T_list])
        fH2_formation = np.array([get_fH2_Tegmark97(T, z) for T in T_list])
        All_critical_fH2.append(fH2_critical)
        All_H2_formation.append(fH2_formation)
    All_critical_fH2 = np.array(All_critical_fH2)
    All_H2_formation = np.array(All_H2_formation)

    fig = plt.figure(figsize=(8, 6))
    linestyles = ['-',':','--']
    for i, z in enumerate(z_list):
        plt.plot(T_list, All_critical_fH2[i], label='critical fraction, z='+str(z), linestyle=linestyles[i], color='k')
        plt.plot(T_list, All_H2_formation[i], label='H2 produced, z='+str(z), linestyle=linestyles[i], color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('T [K]')
    plt.ylabel('fH2')
    plt.title('Tegmark 1997 model')
    plt.legend()
    plt.tight_layout()
    filename = os.path.join("Analytic_results/Yoshida03", "Tegmark_fH2_vs_T_z.png")
    plt.savefig(filename, dpi=300)

def test_fH2():
    z = 20
    M = 1.0e6/h_Hubble #Msun
    T1 = get_Tvir_Yoshida03(M, z, 1.0)
    T2 = Temperature_Virial_analytic(M, z, 1.0)
    print("T_vir_Yoshida03 = ", T1)
    print("T_vir_analytic = ", T2)
    print("T_vir_Yoshida03 / T_vir_analytic = ", T1/T2)

    M_test = 5.0e5/h_Hubble #Msun
    T_test = get_Tvir_Yoshida03(M_test, z, 1.0)

    fH2_test = get_fH2_Yoshida03(T_test)
    print("fH2_test = ", fH2_test)
    fH2_test_Tegmark = get_fH2_Tegmark97(T_test)
    print("fH2_test_Tegmark = ", fH2_test_Tegmark)
    print("fH2_test / fH2_test_Tegmark = ", fH2_test/fH2_test_Tegmark)

def plot_fH2_vs_T():
    output_dir = "/home/zwu/21cm_project/unified_model/Analytic_results/Yoshida03"

    z = 15
    #test different Hubble timescale
    t_Hubble_Tegmark = get_Hubble_timescale_Tegmark97(z)
    t_Hubble_colossus = cosmo.age(z)*1e3
    print(f"t_Hubble_Tegmark = {t_Hubble_Tegmark} Myr")
    print(f"t_Hubble_colossus = {t_Hubble_colossus} Myr")
    # t_Hubble_Myr = get_Hubble_timescale_Tegmark97(z)
    t_Hubble_Myr = t_Hubble_colossus
    # t_Hubble_Myr = 1.0/H0_s/Myr

    lognH = get_gas_lognH_analytic(z)
    # lognH = np.log10(500)
    print("lognH = ", lognH)
    nH = 10**lognH
    print("nH = ", nH)


    #plot fH2 vs T
    T_list = np.logspace(2, 5, 50)
    fH2_list = np.logspace(-6, -1, 50)
    cooling_timescale_all = np.zeros((len(T_list), len(fH2_list)))
    cooling_rate_all = np.zeros((len(T_list), len(fH2_list)))
    for j, fH2 in enumerate(fH2_list):

        cooling_data = run_constdensity_model(
        False, z, lognH, 0.0, 0.0, 
        T_list, 0.0, f_H2=fH2, UVB_flag=False, 
        Compton_Xray_flag=False, dynamic_final_flag=False)
        cooling_time = cooling_data['cooling_time']
        # print("cooling_time = ", cooling_time.in_units('Myr'))
        cooling_timescale_all[:, j] = cooling_time.in_units('Myr')
        # print("cooling_timescale_all = ", cooling_timescale_all[:,j])
        cooling_rate_all[:, j] = cooling_data['cooling_rate']


    # print("cooling_timescale_all = ", cooling_timescale_all)
    # print("cooling_rate_all = ", cooling_rate_all)
    # print("energy in the unit cell = ",cooling_timescale_all*cooling_rate_all) 

    #include DF heating
    DF_heating_list = np.zeros(len(T_list))
    normalized_heating_list = np.zeros(len(T_list))
    for i in range(len(T_list)):
        Tvir = T_list[i]
        M_in_Msun = inversefunc_Temperature_Virial_analytic(Tvir, z, mean_molecular_weight=1.0)
        Mvir = M_in_Msun * h_Hubble #Msun/h
        lgM = np.log10(Mvir)
        DF_heating = integrate_SHMF_heating_for_single_host(z, -3, 0, lgM, "BestFit_z")
        DF_heating_erg = DF_heating * 1.0e7 # erg/s
        halo_density = get_mass_density_analytic(z)
        halo_volume = M_in_Msun * Msun / halo_density
        halo_volume_cm3 = halo_volume * 1.0e6 # cm^3
        DF_heating_density = DF_heating_erg / halo_volume_cm3 # erg/s/cm^3
        normalized_heating = DF_heating_density/nH**2 # erg/s*cm^3

        DF_heating_list[i] = DF_heating_density
        normalized_heating_list[i] = normalized_heating

    Cooling_rate_with_DF_all = np.zeros((len(T_list), len(fH2_list)))
    Cooling_timescale_with_DF_all = np.zeros((len(T_list), len(fH2_list)))
    for i in range(len(T_list)):
        for j in range(len(fH2_list)):
            cooling_rate = cooling_rate_all[i, j]
            cooling_timescale = cooling_timescale_all[i, j]
            energy_density_in_cell = cooling_rate * cooling_timescale
            cooling_rate_with_DF = cooling_rate + normalized_heating_list[i]
            cooling_timescale_with_DF = energy_density_in_cell / cooling_rate_with_DF
            Cooling_rate_with_DF_all[i, j] = cooling_rate_with_DF
            Cooling_timescale_with_DF_all[i, j] = cooling_timescale_with_DF

            
    #find the critical fH2 for each T
    fH2_critical = np.zeros(len(T_list))  
    for i in range(len(T_list)):
        fH2_critical[i] = 1.0
        find_critical_flag = False
        # for j in range(len(fH2_list)-1):
        #     if cooling_timescale_all[i, j] < 0.0 and cooling_timescale_all[i, j+1] <0.0:
        #         if (-cooling_timescale_all[i, j]>t_Hubble_Myr and -cooling_timescale_all[i, j+1]<=t_Hubble_Myr):
        #             fH2_critical[i] = fH2_list[j]
        #             find_critical_flag = True
        #             break

        for j in range(len(fH2_list)-1,0,-1):
            if cooling_timescale_all[i, j] < 0.0 and cooling_timescale_all[i, j-1] <0.0:
                if (-cooling_timescale_all[i, j]<=t_Hubble_Myr and -cooling_timescale_all[i, j-1]>t_Hubble_Myr):
                    fH2_critical[i] = fH2_list[j]
                    find_critical_flag = True
                    break
            if j == 1 and cooling_timescale_all[i, j] < 0.0 and -cooling_timescale_all[i, j] <= t_Hubble_Myr:
                fH2_critical[i] = fH2_list[j] #the lowest fH2

        # if find_critical_flag == False and -cooling_timescale_all[i, 0] <= t_Hubble_Myr:
        #     fH2_critical[i] = 1.0e-7
        
    fH2_with_DF_critical = np.zeros(len(T_list))
    for i in range(len(T_list)):
        fH2_with_DF_critical[i] = 1.0
        find_critical_flag = False
        for j in range(len(fH2_list)-1,0,-1):
            if Cooling_timescale_with_DF_all[i, j] < 0.0 and Cooling_timescale_with_DF_all[i, j-1] <0.0:
                if (-Cooling_timescale_with_DF_all[i, j]<=t_Hubble_Myr and -Cooling_timescale_with_DF_all[i, j-1]>t_Hubble_Myr):
                    fH2_with_DF_critical[i] = fH2_list[j]
                    find_critical_flag = True
                    break
            if j == 1 and Cooling_timescale_with_DF_all[i, j] < 0.0 and -Cooling_timescale_with_DF_all[i, j] <= t_Hubble_Myr:
                fH2_with_DF_critical[i] = fH2_list[j]

    # print("T_list = ", T_list)  
    # print("fH2_critical = ", fH2_critical)

    #first check if all the cooling timescale and cooling rate are negative
    for i in range(len(T_list)):
        for j in range(len(fH2_list)):
            if cooling_timescale_all[i, j] > 0.0:
                print("cooling_timescale_all[", i, ",", j, "] = ", cooling_timescale_all[i, j])
            if cooling_rate_all[i, j] > 0.0:
                print("cooling_rate_all[", i, ",", j, "] = ", cooling_rate_all[i, j])


    
    fH2_Tegmark = np.array([get_fH2_Tegmark97(T, z) for T in T_list])
    fH2_Yoshida = get_fH2_Yoshida03(T_list)
    fH2_critical_Tegmark = np.array([get_critical_fH2_Tegmark97(T, z) for T in T_list])

    #2D color plot for cooling rate
    T_grid, fH2_grid = np.meshgrid(T_list, fH2_list, indexing='ij')

    fig = plt.figure(figsize=(8, 6))
    lg_cooling_rate_all = np.log10(-cooling_rate_all)
    c = plt.pcolormesh(
        T_grid, fH2_grid, lg_cooling_rate_all, 
        cmap='seismic', shading='auto',
        vmin=np.min(lg_cooling_rate_all), vmax=np.max(lg_cooling_rate_all)
    )
    cb = plt.colorbar(c, label=r'log$_{10}$(cooling rate [erg cm$^3$ /s])')  # Add units if known
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Temperature [K]', fontsize=15)
    plt.ylabel('Molecular Hydrogen Fraction $f_{H_2}$', fontsize=15)
    plt.title('Cooling Rate vs Temperature and $f_{H_2}$')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cooling_rate_vs_T_fH2_z{z}.png"), dpi=300)
    


    #2D color plot for cooling timescale
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.gca()
    lg_cooling_timescale_all = np.log10(-cooling_timescale_all)
    c = plt.pcolormesh(
        T_grid, fH2_grid, lg_cooling_timescale_all, 
        cmap='seismic', shading='auto',
        vmin=np.min(lg_cooling_timescale_all), vmax=np.max(lg_cooling_timescale_all)
    )
    cb = plt.colorbar(c, label=r'log$_{10}$(cooling timescale [Myr])') 
    #add fH2_critical_Tegmark on top of the plot
    ax1.plot(T_list[T_list<=1e4], fH2_critical_Tegmark[T_list<=1e4], color='k', linestyle='-', label='critical line (Tegmark97)')
    ax1.plot(T_list, fH2_critical, color='green', linestyle='-', label='critical line (Grackle)')
    ax1.plot(T_list, fH2_with_DF_critical, color='orange', linestyle='-', label='critical line with DF heating (Grackle)')
    # ax1.plot(T_list[T_list<=1e4], fH2_Tegmark[T_list<=1e4], color='k', linestyle='--', label='H2 fraction produced (Tegmark97)')
    ax1.plot(T_list[T_list<=1e4], fH2_Yoshida[T_list<=1e4], color='grey', linestyle='--', label='H2 produced ~ T^1.52')
    ax1.set_ylim(1e-6, 1e-1)
    plt.legend()
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Temperature [K]', fontsize=15)
    ax1.set_ylabel('Molecular Hydrogen Fraction $f_{H_2}$', fontsize=15)
    # ax2 = ax1.twiny()
    # ax2.set_xlim(ax1.get_xlim())
    # Tvir_min, Tvir_max = ax1.get_xlim()
    # lgM_min = 3  # 1e3 Msun
    # lgM_max = 7  # 1e7 Msun
    # lgM_locator = LogLocator(base=10)
    # lgM_ticks_top = lgM_locator.tick_values(lgM_min, lgM_max)
    # Tvir_ticks = [Temperature_Virial_analytic(10**lgM, z, mean_molecular_weight=1.0) for lgM in lgM_ticks_top]

    # Filter to only keep Tvir ticks that lie within current temperature axis limits
    # valid_ticks = [(lgM, Tvir) for lgM, Tvir in zip(lgM_ticks_top, Tvir_ticks) if Tvir_min <= Tvir <= Tvir_max]
    # if valid_ticks:
    #     lgM_ticks_top, Tvir_ticks = zip(*valid_ticks)
    #     ax2.set_xticks(Tvir_ticks)
    #     ax2.set_xticklabels([f"$10^{{{int(lgM)}}}$" for lgM in lgM_ticks_top])
    #     ax2.set_xlabel(r'Halo Mass $M_{\rm vir}$ [$M_\odot$]', fontsize=14)

    plt.title(r'Cooling Timescale vs Temperature and $f_{H_2}$ at z=' + str(z)+r' (t$_{H}$ = ' + f'{t_Hubble_Myr:.0f}' + 'Myr)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cooling_timescale_vs_T_fH2_z{z}.png"), dpi=300)
    


    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(T_list, fH2_Tegmark, label='Tegmark 1997', color='blue')
    ax.plot(T_list, fH2_Yoshida, label='Yoshida 2003', color='red')
    ax.scatter(T_list, fH2_critical, label='fH2 critical', color='green')
    ax.plot(T_list, fH2_critical_Tegmark, label='fH2 critical Tegmark', color='orange')

    ax.plot(T_list, fH2_critical, color='green')

    ax.legend()
    ax.grid()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('T [K]', fontsize=14)
    ax.set_ylabel('fH2', fontsize=14)
    filename = os.path.join(output_dir, "fH2_vs_T.png")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(filename)
    """
    


    #save cooling rate and cooling timescale to txt file
    """
    cooling_rate_filename = os.path.join(output_dir, f"cooling_rate_z{z}.txt")
    cooling_timescale_filename = os.path.join(output_dir, f"cooling_timescale_z{z}.txt")
    
    # Suppose cooling_rate_all has shape (len(T_list), len(fH2_list))
    n_T = len(T_list)
    n_fH2 = len(fH2_list)

    # Create new array with one extra row and column
    cooling_rate_with_labels = np.empty((n_T + 1, n_fH2 + 1))
    cooling_timescale_with_labels = np.empty((n_T + 1, n_fH2 + 1))

    # Fill in headers
    cooling_rate_with_labels[0, 0] = np.nan  # Top-left corner empty or NaN
    cooling_rate_with_labels[0, 1:] = fH2_list  # First row = fH2
    cooling_rate_with_labels[1:, 0] = T_list    # First column = T
    cooling_rate_with_labels[1:, 1:] = cooling_rate_all  # Fill in data

    # Repeat for cooling timescale
    cooling_timescale_with_labels[0, 0] = np.nan
    cooling_timescale_with_labels[0, 1:] = fH2_list
    cooling_timescale_with_labels[1:, 0] = T_list
    cooling_timescale_with_labels[1:, 1:] = cooling_timescale_all

    # Save to file
    np.savetxt(cooling_rate_filename, cooling_rate_with_labels, fmt="%.4e")
    np.savetxt(cooling_timescale_filename, cooling_timescale_with_labels, fmt="%.4e")
    """


if __name__ == "__main__":
    # compare_Hubble_timescale_and_nH()
    plot_fH2_vs_T()
    # test_fH2_Tegmark()
    # test_fH2()
    # test_Tvir()