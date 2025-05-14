import numpy as np
import matplotlib.pyplot as plt
import os

from physical_constants import h_Hubble, H0_s, Omega_m
from HaloProperties import Temperature_Virial_analytic, get_gas_lognH_analytic
from Grackle_cooling import run_constdensity_model

def get_Tvir_Yoshida03(M, z):
    #M in Msun
    mu = 1.0
    Delta = 200
    T =  1.98e4*(mu/0.6)*(M*h_Hubble/1e8)**(2/3) \
        *(Omega_m*Delta/18/np.pi**2)**(1/3)*(1+z)/10
    return T

def get_fH2_Tegmark97(T):
    #Tegmark 1997 Eq.17
    #do not consider redshift dependence now, to be added later
    #T in K, return fH2
    return 3.5e-4 * (T/1e3)**1.52


def get_fH2_Yoshida03(T):
    
    T3 = T/1e3
    fH2 = 4.7e-5 * T3**1.52
    return fH2

def test_fH2():
    z = 20
    M = 1.0e6/h_Hubble #Msun
    T1 = get_Tvir_Yoshida03(M, z)
    T2 = Temperature_Virial_analytic(M, z)
    print("T_vir_Yoshida03 = ", T1)
    print("T_vir_analytic = ", T2)
    print("T_vir_Yoshida03 / T_vir_analytic = ", T1/T2)

    M_test = 5.0e5/h_Hubble #Msun
    T_test = get_Tvir_Yoshida03(M_test, z)

    fH2_test = get_fH2_Yoshida03(T_test)
    print("fH2_test = ", fH2_test)
    fH2_test_Tegmark = get_fH2_Tegmark97(T_test)
    print("fH2_test_Tegmark = ", fH2_test_Tegmark)
    print("fH2_test / fH2_test_Tegmark = ", fH2_test/fH2_test_Tegmark)

def plot_fH2_vs_T():
    output_dir = "/home/zwu/21cm_project/unified_model/Analytic_results/Yoshida03"

    t_Hubble = 1./H0_s
    t_Hubble_Myr = t_Hubble / 3.15576e13 #seconds to Myr
    print("t_Hubble = ", t_Hubble_Myr, "Myr")
    z = 17
    lognH = get_gas_lognH_analytic(z)
    # lognH = np.log10(500)
    print("lognH = ", lognH)
    nH = 10**lognH
    print("nH = ", nH)

    
    #plot fH2 vs T
    T_list = np.logspace(1, 4, 50)
    fH2_list = np.logspace(-6, -2, 50)
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


    print("cooling_timescale_all = ", cooling_timescale_all)

    #find the critical fH2 for each T
    fH2_critical = np.zeros(len(T_list))  
    for i in range(len(T_list)):
        fH2_critical[i] = 1.0
        find_critical_flag = False
        for j in range(len(fH2_list)-1):
            if cooling_timescale_all[i, j] < 0.0 and cooling_timescale_all[i, j+1] <0.0:
                if (-cooling_timescale_all[i, j]>t_Hubble_Myr and -cooling_timescale_all[i, j+1]<=t_Hubble_Myr):
                    fH2_critical[i] = fH2_list[j]
                    find_critical_flag = True
                    break
        # if find_critical_flag == False and -cooling_timescale_all[i, 0] <= t_Hubble_Myr:
        #     fH2_critical[i] = 1.0e-7
        
    
    print("T_list = ", T_list)  
    print("fH2_critical = ", fH2_critical)


    fH2_Tegmark = get_fH2_Tegmark97(T_list)
    fH2_Yoshida = get_fH2_Yoshida03(T_list)
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(T_list, fH2_Tegmark, label='Tegmark 1997', color='blue')
    ax.plot(T_list, fH2_Yoshida, label='Yoshida 2003', color='red')
    ax.scatter(T_list, fH2_critical, label='fH2 critical', color='green')
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

    #save cooling rate and cooling timescale to txt file
    cooling_rate_filename = os.path.join(output_dir, "cooling_rate.txt")
    cooling_timescale_filename = os.path.join(output_dir, "cooling_timescale.txt")
    np.savetxt(cooling_rate_filename, cooling_rate_all)
    np.savetxt(cooling_timescale_filename, cooling_timescale_all)



if __name__ == "__main__":
    plot_fH2_vs_T()