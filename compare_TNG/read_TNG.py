import numpy as np
import h5py
import matplotlib.pyplot as plt
import illustris_python as il
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from physical_constants import *
from scipy.special import gamma
from scipy.integrate import solve_ivp, quad
from scipy.integrate import nquad
from scipy.optimize import curve_fit
import sys
import os

#output dn/dM in the unit of [(Mpc/h)^(-3) (Msun/h)^(-1)]
#input M in the unit of Msun/h
def HMF_Colossus(M, z):
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

def Temperature_Virial(M,z):
    #M in solar mass, return virial temperature in K
    return 3.6e5 * (Vel_Virial(M,z)/1e3/100)**2



#dN/ d ln(ln(m/M))
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
        



#use Bosch2016 model to calculate the DF heating
def integrand(ln_m_over_M, logM, z, *bestfitparams):
    if not bestfitparams:
        bestfitparams = None
    global SHMF_model
    global G_grav,rho_b0,h_Hubble, Mpc, Msun
    
    eta = 1.0
    I_DF = 1.0
    
    M = 10**logM
    m_over_M = np.exp(ln_m_over_M)
    m = m_over_M * M  
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    DF_heating =  eta * 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Vel_Virial(M/h_Hubble, z) *rho_g *I_DF
    if SHMF_model == 'BestFit':
        DF_heating *= Subhalo_Mass_Function_ln(ln_m_over_M,bestfitparams)
    else:
        DF_heating *= Subhalo_Mass_Function_ln(ln_m_over_M) 


    DF_heating *= HMF_Colossus(M,z) * np.log(10)*M   #convert from M to log10(M)
    
    return DF_heating


def plot_hmf(halos, current_redshift, dark_matter_resolution,hmf_filename):

    #plot HMF (halo mass function)
    group_mass_solar = halos['GroupMass']*1e10   #unit: 1e10 Msun/h
    
    # Create a histogram (logarithmic bins and logarithmic mass)
    bins = np.logspace(np.log10(min(group_mass_solar)), np.log10(max(group_mass_solar)), num=50)
    hist, bin_edges = np.histogram(group_mass_solar, bins=bins)

    # Convert counts to number density
    log_bin_widths = np.diff(np.log10(bins))
    number_density = hist / volume / log_bin_widths

    # Plot the mass function
    fig = plt.figure(facecolor='white')
 
    logM_limits = [5, 12]  # Limits for log10(M [Msun/h])
    HMF_lgM = []
    logM_list = np.linspace(logM_limits[0], logM_limits[1],57)
    for logM in logM_list:
        
        M = 10**(logM)
        HMF_lgM.append(HMF_Colossus(10**logM, current_redshift)* np.log(10)*M)  

    plt.axvline(100*dark_matter_resolution, color='black', linestyle='--')
    plt.loglog(bins[:-1], number_density, marker='.')
    plt.plot(10**(logM_list),HMF_lgM,color='red',linestyle='-')
    plt.title(f'Host Halo Mass Function, z={current_redshift:.2f}')
    plt.xlabel('Mass [$M_\odot/h$]')
    plt.ylabel(r'$\frac{dN}{d\log_{10}M}$ [$(Mpc/h)^{-3}$]')
    plt.tight_layout()
    plt.savefig(hmf_filename,dpi=300)
    





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
    M_Jeans = 220*(1+z_value)**1.425*h_Hubble
    print("Jeans mass: ",M_Jeans)
    logM_limits = [np.log10(M_Jeans), 15]  # Limits for log10(M [Msun/h])
    ln_m_over_M_limits = [np.log(1e-3), 0]  # Limits for m/M

    logM_list = np.linspace(logM_limits[0], logM_limits[1],57)


    DF_heating_perlogM = []
    for logM in logM_list:
        if SHMF_model == 'BestFit':
            result, error = quad(integrand, ln_m_over_M_limits[0], ln_m_over_M_limits[1], args=(logM, z_value,*bestfitparams))
        else:
            result, error = quad(integrand, ln_m_over_M_limits[0], ln_m_over_M_limits[1], args=(logM, z_value))



        if (abs(error) > 0.01 * abs(result)):
            print("Possible large integral error at z = %f, relative error = %f\n", z_value, error/result)

        DF_heating_perlogM.append(result)

    DF_heating_perlogM = np.array(DF_heating_perlogM)
    plt.plot(logM_list,1e7*DF_heating_perlogM,'g-',label='subhalo analytic')
    plt.axvline(np.log10(100*dark_matter_resolution), color='black', linestyle='--')
    plt.legend()
    #plt.xlim([4,12])
    xlim_min = min(np.min(logM_list),4)
    xlim_max = max(np.max(logM_list),12)
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
        DF_heating_total_analytic, error = nquad(integrand, [[ln_m_over_M_limits[0], ln_m_over_M_limits[1]], [logM_limits[0], logM_limits[1]]], args=(z_value,*bestfitparams))
    else:
        DF_heating_total_analytic, error = nquad(integrand, [[ln_m_over_M_limits[0], ln_m_over_M_limits[1]], [logM_limits[0], logM_limits[1]]], args=(z_value,))


    if (abs(error) > 0.01 * abs(DF_heating_total_analytic)):
        print("possible large integral error at z = %f, relative error = %f\n",z_value,error/DF_heating_total_analytic)
    print("subhalo DF_heating_total_analytic: ",DF_heating_total_analytic)
    print("ratio: ",subhalo_DF_heating_total_TNG/DF_heating_total_analytic)
    
#estimate mass weighted average IGM gas velocity within r = 5*R_crit200 (periodic boundary considered)
def calc_gas_vel(gas_all,halo_pos,R_crit200):
    gas_pos = gas_all['Coordinates']
    gas_vel = gas_all['Velocities']
    gas_mass = gas_all['Masses']
    box_size = 75000  #ckpc/h
    gas_vel_weighted = np.zeros(3)
    total_mass = 0
    for i in range(len(gas_pos)):
        #progress bar (current i / len(gas_pos)) at every 5% progress
        if(i % (len(gas_pos)//20) == 0):
            print(f"progress: {i}/{len(gas_pos)}")
        
        pos = gas_pos[i]
        vel = gas_vel[i]
        mass = gas_mass[i]
        r = np.sqrt(np.sum((pos-halo_pos)**2))
        if(r < 5*R_crit200):
            gas_vel_weighted += vel*mass
            total_mass += mass
        else:
            #periodic boundary
            for j in range(3):
                if(pos[j] - halo_pos[j] > box_size/2):
                    pos[j] -= box_size
                elif(pos[j] - halo_pos[j] < -box_size/2):
                    pos[j] += box_size
            r = np.sqrt(np.sum((pos-halo_pos)**2))
            if(r < 5*R_crit200):
                gas_vel_weighted += vel*mass
                total_mass += mass
    gas_vel_weighted /= total_mass
    return gas_vel_weighted

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

#fix omega = 4 and beta_ln10 = 50/ln(10)
def fitFunc_lg_dNdlgx_fixomega(lgx,alpha,lgA):
    x = 10**lgx
    beta_ln10 = 50/np.log(10)
    omega = 4
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

    all_fitting_params = []
    for i in range(3,5):
        number_density = number_density_list[i]
        #fitFunc_lg_dNdlgx(lgx,alpha,beta_ln10, omega, lgA)
        #p_guess = [0.86, 50/np.log(10),4 ,np.log10(0.065)]
        try:
            if current_redshift > 10:
                fit_mask = (bin_centers > np.log10(2.0*critical_ratio_list[i])) & (number_density[:-1] != artificial_small)
            else:
                fit_mask = (bin_centers > np.log10(critical_ratio_list[i])) & (number_density[:-1] != artificial_small)
            popt, pcov = curve_fit(fitFunc_lg_dNdlgx, bin_centers[fit_mask], np.log10(number_density[:-1][fit_mask]), p0=p_guess, maxfev= 1000)
            alpha = popt[0]
            beta_ln10 = popt[1]
            omega = popt[2]

            #if (alpha < 0 and beta_ln10>0):
            if (False):
                print("i = {}: fit failed, increase critical ratio".format(i))
                popt, pcov = curve_fit(fitFunc_lg_dNdlgx, bin_centers[fit_mask], np.log10(number_density[:-1][fit_mask]), p0=p_guess, maxfev= 1000)
            #elif(beta_ln10 <= 0 or omega > 10):
            elif(True):
                print("i = {}: fit failed, increase critical ratio and fix the exponential slope".format(i))
                fit_mask = (bin_centers > np.log10(2.0*critical_ratio_list[i])) & (number_density[:-1] != artificial_small)
                p0_fixomega = [0.86, np.log10(0.065)]
                popt, pcov = curve_fit(fitFunc_lg_dNdlgx_fixomega, bin_centers[fit_mask], np.log10(number_density[:-1][fit_mask]), p0=p0_fixomega, maxfev= 1000)
                alpha = popt[0];  lgA = popt[1]
                popt = [alpha,50/np.log(10),4,lgA]

        except:
            print("i = {}: fit failed, increase critical ratio and fix the exponential slope".format(i))
            fit_mask = (bin_centers > np.log10(2.0*critical_ratio_list[i])) & (number_density[:-1] != artificial_small)
            p0_fixomega = [0.86, np.log10(0.065)]
            popt, pcov = curve_fit(fitFunc_lg_dNdlgx_fixomega, bin_centers[fit_mask], np.log10(number_density[:-1][fit_mask]), p0=p0_fixomega, maxfev= 1000)
            alpha = popt[0];  lgA = popt[1]
            popt = [alpha,50/np.log(10),4,lgA]

        


            #popt = advanced_fit(bin_centers[fit_mask],number_density[:-1][fit_mask],p_guess)

        print("fit parameters: ",popt)
        all_fitting_params.append([snapNum,i,logM_bins[i],logM_bins[i+1],*popt])

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

    #write_SHMF_fit_parameters(filename.replace('.png','_fit_parameters.txt'),all_fitting_params)
    write_SHMF_fit_parameters(filename.replace('.png','_2paramfit_parameters.txt'),all_fitting_params)

def advanced_fit(x,y,p_guess):
    pass

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

simulation_set = 'TNG50-1'
SHMF_model = 'BestFit'
if __name__ == "__main__":
    if simulation_set == 'TNG100-1':
        gas_resolution = 1.4e6 * h_Hubble  # Msun/h 
        dark_matter_resolution = 7.5e6 * h_Hubble  # Msun/h
    elif simulation_set == 'TNG50-1':
        gas_resolution = 4.5e5 * h_Hubble
        dark_matter_resolution = 8.5e4 * h_Hubble

    basePath = '/home/zwu/21cm_project/TNG_data/'+simulation_set+'/output'
    #snapNum = 99
    #get snapNum as a parameter when running the file
    #snapNum = int(sys.argv[1])
    snapNum = 3
    output_dir = '/home/zwu/21cm_project/compare_TNG/results/'+simulation_set+'/'
    output_dir += f'snap_{snapNum}/'
    #mkdir for this snapshot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("output_dir: ",output_dir)


    # Load the group catalog data
    print("loading header ...")
    header = il.groupcat.loadHeader(basePath, snapNum)
    print("loading halos ...")
    halos = il.groupcat.loadHalos(basePath, snapNum, fields=['GroupFirstSub', 'GroupNsubs', 'GroupPos', 'GroupMass', 'GroupMassType','Group_M_Crit200','Group_R_Crit200','Group_R_Crit500','GroupVel'])
    print("loading subhalos ...")
    subhalos = il.groupcat.loadSubhalos(basePath, snapNum, fields=['SubhaloMass', 'SubhaloPos', 'SubhaloVel', 'SubhaloGrNr', 'SubhaloMassType'])
    #Type: 0 gas , 1 dark matter, 2 gas tracers, 3 stellar,  4 stellar wind particles, 5 black hole sinks
    

    # print(halos['GroupMassType'][10])
    # print(halos['GroupMassType'][10].sum())
    # print(halos['GroupMass'][10])

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
    hmf_filename= output_dir+ f'HMF_snap_{snapNum}_z_{current_redshift:.2f}.png'
    #plot_hmf(halos, current_redshift,dark_matter_resolution,hmf_filename)

    
    #Load snapshot data
    # print("loading gas ...")
    # gas_all = il.snapshot.loadSubset(basePath, snapNum, 'gas', fields=['Masses','Coordinates','Velocities'])
    #gas_all['Masses'] gas particle mass in 1e10 Msun/h
    #gas_all['Coordinates'] gas particle position in ckpc/h
    #gas_all['Velocities'] gas peculiar vel in sqrt(a)*km/s
    #print("finish loading gas.")



    All_sub_host_M  = []
    #calculate the average m/M ratio for each host halo
    for host in range(len(halos['GroupMass'])):
        #Calculate M and m
        M = halos['GroupMass'][host]*1e10
        if (M < 100*dark_matter_resolution):
            continue
        # Get the subhalos of this host
        first_sub = int(halos['GroupFirstSub'][host])
        num_subs = int(halos['GroupNsubs'][host])
        subhalos_of_host = [(j, subhalos['SubhaloMass'][j]*1e10) for j in range(first_sub, first_sub + num_subs)]   #to Msun/h
        if (num_subs == 0):
            continue
        subhalos_of_host.sort(key=lambda x: x[1])
        subhalos_of_host = subhalos_of_host[:-1] #exclude the most massive subhalo (not a real subhalo)
        #calculate m/M ratio
        for (subhalo_index, subhalo_mass) in subhalos_of_host:
            m = subhalo_mass
            if (m < 50*dark_matter_resolution):
                continue
            if (m/M >= 1):
                print("Warning: m/M > 1")
                continue
            All_sub_host_M.append([m/M, M, host])

    #plot histogram of m/M ratios 
    All_sub_host_M = np.array(All_sub_host_M)
    All_sub_host_Mratio = All_sub_host_M[:,0]
    All_host_M = All_sub_host_M[:,1]

    #plot_Mratio_cumulative(All_sub_host_Mratio,output_dir+f'Mratio_cumulative_snap{snapNum}_z{current_redshift:.2f}.png')

    # plot_Mratio_dN_dlogMratio(All_sub_host_M,dark_matter_resolution, output_dir+f'Average_Mratio_dN_dlogMratio_snap{snapNum}_z{current_redshift:.2f}.png')


    #Calculate the total DF heating for each host halo mass bin 
    #Define the bins for the host halo masses
    logM_min = np.log10(halos['GroupMass'].min()*1e10)
    logM_max = np.log10(halos['GroupMass'].max()*1e10)
    logM_bins = np.linspace(logM_min, logM_max, num=20)
    logM_bin_width = logM_bins[1] - logM_bins[0]
    logM_bin_centers = (logM_bins[:-1] + logM_bins[1:]) / 2

    # Initialize an array to store the total DF heating for each bin
    subhalo_DF_heating_hostmassbin = np.zeros_like(logM_bin_centers)
    hosthalo_DF_heating_hostmassbin = np.zeros_like(logM_bin_centers)

    # Initialize a list to store the host halos for each bin
    hosts_in_bins = [[] for _ in range(len(logM_bin_centers))]

    # Loop over each host halo
    for host in range(len(halos['GroupMass'])):
        # Determine which bin this host belongs to
        host_logM = np.log10(halos['GroupMass'][host]*1e10)
        bin_index = np.searchsorted(logM_bins, host_logM) - 1
        # Add the host to the corresponding bin
        hosts_in_bins[bin_index].append(host)

    # Now loop over each bin and each host in the bin
    subhalo_DF_heating_list = []
    hosthalo_DF_heating_list = []
    vel_host_list = []
    vel_host_gas_list = []
    T_DFheating_list = []
    DF_thermal_energy_list = []
    Mach_rel_list = []
    
    for i, hosts_in_bin in enumerate(hosts_in_bins):
        # Loop over each host halo in this bin
        print("\n\nbin: ",i)
        print("\n")
        for host in hosts_in_bin:
            
            #Calculate M and m
            M = halos['GroupMass'][host]*1e10  #Msun/h
            M_crit200 = halos['Group_M_Crit200'][host]*1e10 #Msun/h
            M_gas = halos['GroupMassType'][host][0]*1e10  #Msun/h
            R_crit200 = halos['Group_R_Crit200'][host]  #ckpc/h
            R_crit200 = R_crit200/1e3 * scale_factor / h_Hubble  #Mpc
            R_crit500 = halos['Group_R_Crit500'][host]  #ckpc/h
            R_crit500 = R_crit500/1e3 * scale_factor / h_Hubble
            
            rho_g_analytic = rho_b0*(1+current_redshift)**3 *Msun/Mpc**3
            rho_m_analytic = rho_m0*(1+current_redshift)**3 *Msun/Mpc**3
            rho_g_analytic_200 = 200 *rho_g_analytic
            rho_m_analytic_200 = 200 *rho_m_analytic
            group_vel = halos['GroupVel'][host]*1e3/scale_factor  #km/s/a to m/s
            vel_analytic = Vel_Virial(M/h_Hubble, current_redshift)

            #exclude small halos not resolved
            if(M < 100*dark_matter_resolution):
                continue
           
            #calculate host halo DF heating, unit: J/s
            #use global average gas density instead of 200*rho_b(z)
            rho_g = rho_g_analytic
            vel_host = np.sqrt(np.sum(group_vel**2))
            vel_host_list.append(vel_host)
            
            Tvir_host = Temperature_Virial(M/h_Hubble, current_redshift)
            Cs_host = np.sqrt(5.0/3.0 *kB *Tvir_host/(mu*mp))
            #surrounding_gas_vel = calc_gas_vel(gas_all,halos['GroupPos'][host],R_crit200)
            #surrounding_gas_vel = surrounding_gas_vel*1e3*np.sqrt(scale_factor)  #sqrt(a)*km/s to m/s
            # print("surrounding gas vel: ",surrounding_gas_vel)
            # vel = np.sqrt(np.sum((group_vel - surrounding_gas_vel)**2))
            # print("host vel relative to gas: ",vel_host_gas)
            # vel_host_gas_list.append(vel_host_gas)
            
            I_DF = 1.0
            eta = 1.0  

            hosthalo_DF_heating =  eta * 4 * np.pi * (G_grav * M *Msun/h_Hubble) ** 2 / vel_host *rho_g *I_DF
            #use relative velocity between host halo and surrounding gas
            # hosthalo_DF_heating =  eta * 4 * np.pi * (G_grav * M *Msun/h_Hubble) ** 2 / vel_host_gas *rho_g *I_DF

            hosthalo_DF_heating_list.append((host,hosthalo_DF_heating))
            hosthalo_DF_heating_hostmassbin[i] += hosthalo_DF_heating

            #Now calculate subhalo DF heating

            # Get the subhalos of this host
            first_sub = int(halos['GroupFirstSub'][host])
            num_subs = int(halos['GroupNsubs'][host])
            subhalos_of_host = [(j, subhalos['SubhaloMass'][j]*1e10) for j in range(first_sub, first_sub + num_subs)]   #to Msun/h

            if (num_subs == 0):
                continue
            subhalos_of_host.sort(key=lambda x: x[1])
            maxsub_index = subhalos_of_host[-1][0]
            #heating = 0 for the most massive subhalo as it's not a real subhalo
            subhalo_DF_heating_list.append((maxsub_index,0.0))

            subhalos_of_host = subhalos_of_host[:-1]

            # Loop over each subhalo
            for (subhalo_index, subhalo_mass) in subhalos_of_host:
                m = subhalo_mass   
                #exclude small subhalos not resolved
                if(m < 50*dark_matter_resolution):
                    continue
                #exclude possible incorrect subhalos
                if (m/M >= 1):
                    print("Warning: m/M > 1")
                    continue



                #use simulation velocity and gas density
                subhalo_vel = subhalos['SubhaloVel'][subhalo_index]*1e3  #km/s to m/s (default unit for subhalo vel is km/s, see TNG data specification)

                vel = np.sqrt(np.sum((group_vel - subhalo_vel)**2))
                Mach_rel = vel/Cs_host
                Mach_rel_list.append(Mach_rel)
                rho_g = rho_g_analytic_200

                # if(R_crit200 > 0):
                #     rho_g_numerical = M_gas*(Msun/h_Hubble)/(4/3*np.pi*R_crit200**3)  #kg/Mpc^3
                #     rho_g_numerical = rho_g_numerical/Mpc**3  #kg/m^3
                #     print("rho_g ratio: ",rho_g_numerical/rho_g_analytic_200) #Mgas > Mgas in R_crit200
                # else:
                #     rho_g = rho_g_analytic_200

                I_DF = 1.0
                eta = 1.0

                # Calculate DF_heating and add it to the total for this bin
                DF_heating =  eta * 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / vel *rho_g *I_DF

                subhalo_DF_heating_hostmassbin[i] += DF_heating
                #estimate the temperature of the DF wake 
                #dynamical time scale
                if (R_crit200 > 0):
                    R_crit200_m = R_crit200 * Mpc
                    rho_halo = (M_crit200*Msun/h_Hubble)/(4/3*np.pi*R_crit200_m**3)
                    print("rho halo ratio: ",rho_halo/rho_m_analytic_200) #always 1
                else:
                    rho_halo = rho_m_analytic_200
                t_dyn = 1/np.sqrt(G_grav*rho_halo)
                #t_DF (to be continued)
                DF_thermal_energy = DF_heating * t_dyn
                DF_thermal_energy_list.append(DF_thermal_energy)
                #size of the wake and density fluctuations? (to be continued)
                N_gas = M_gas*Msun/h_Hubble/(mu*mp)
                T_DFheating = DF_thermal_energy * 2/(3*kB*N_gas)  #K
                T_DFheating_list.append(T_DFheating)
                print("temperature of DF heating: ",T_DFheating)
                print(f"t_dyn: {t_dyn:.5e} s = ",t_dyn/1e9/3.15e7," Gyr")
                
                
                
                subhalo_DF_heating_list.append((subhalo_index,DF_heating,T_DFheating))
            #end of subhalo loop
        #end of host loop
    #end of host mass bin loop
    print("max T: ",np.max(T_DFheating_list))
    print("min T: ",np.min(T_DFheating_list))
    print("max DF thermal energy: ",np.max(DF_thermal_energy_list))
    print("min DF thermal energy: ",np.min(DF_thermal_energy_list))
    

    
    #plot DF heating per logM
    '''    
    if SHMF_model == 'BestFit':
        filename = output_dir+f"DF_heating_perlogM_comparison_BestFit_snap{snapNum}_z{current_redshift:.2f}.png"
        #read bestfit parameters
        SHMF_fit_params_file = output_dir+f"Average_Mratio_dN_dlogMratio_snap{snapNum}_z{current_redshift:.2f}_fit_parameters.txt"
        bestfitparams = read_SHMF_fit_parameters(SHMF_fit_params_file)
        plot_DF_heating_per_logM_comparison(volume, current_redshift,dark_matter_resolution,logM_bins,subhalo_DF_heating_hostmassbin,hosthalo_DF_heating_hostmassbin,filename,bestfitparams)
    else:
        filename = output_dir+f"DF_heating_perlogM_comparison_Bosch16_snap{snapNum}_z{current_redshift:.2f}.png"
        plot_DF_heating_per_logM_comparison(volume, current_redshift,dark_matter_resolution,logM_bins,subhalo_DF_heating_hostmassbin,hosthalo_DF_heating_hostmassbin,filename)
    '''
    
    #plot histogram of Mach number
    Mach_rel_list = np.array(Mach_rel_list)
    fig = plt.figure(facecolor='white')
    plt.hist(Mach_rel_list, bins=50)
    plt.title(f'Mach Number Distribution, z={current_redshift:.2f}')
    plt.xlabel('Mach Number')
    plt.ylabel('Counts')
    plt.tight_layout()
    plt.savefig(output_dir+f'Mach_number_snap{snapNum}_z{current_redshift:.2f}.png',dpi=300)
    
    
    
    
    exit(0)
    T_DFheating_list = np.array(T_DFheating_list)
    DF_thermal_energy_list = np.array(DF_thermal_energy_list)
    DF_thermal_energy_erg_list = DF_thermal_energy_list * 1e7  #erg
    
    #plot T_DFheating histogram (in lgT bin)
    bins = np.logspace(np.log10(T_DFheating_list.min()), np.log10(T_DFheating_list.max()), num=50)
    hist, bin_edges = np.histogram(T_DFheating_list, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    log_bin_centers = np.log10(bin_centers)

    # Calculate the width of each bar
    width = np.diff(log_bin_centers)
    width = np.append(width, width[-1])
    # Add a zero at the end of the width array to match the shape of log_bin_centers

    fig = plt.figure(facecolor='white',figsize=(10,6))
    plt.bar(log_bin_centers, hist, align='center', width=width)
    plt.title(f'T_DF Distribution, z={current_redshift:.2f}')
    plt.xlabel('log10 (T_DF) [K]',fontsize=13)
    plt.ylabel('Counts',fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir+f'T_DF_snap{snapNum}_z{current_redshift:.2f}.png',dpi=300)
    
        
        
    #plot 2D histogram of T_DF and thermal energy    
    # Convert the temperature list to logarithmic scale
    log_T_DFheating_list = np.log10(T_DFheating_list)

    # Convert the thermal energy list to erg
    DF_thermal_energy_erg_list = DF_thermal_energy_list * 1e7

    # Define the number of bins for the 2D histogram
    num_bins_x = 50
    num_bins_y = 40

    # Define logarithmic bins for both dimensions
    xedges = np.logspace(np.log10(T_DFheating_list.min()), np.log10(T_DFheating_list.max()), num_bins_x+1)
    yedges = np.logspace(np.log10(DF_thermal_energy_erg_list.min()), np.log10(DF_thermal_energy_erg_list.max()), num_bins_y+1)

    # Create the 2D histogram
    hist2d, xedges, yedges = np.histogram2d(T_DFheating_list, DF_thermal_energy_erg_list, bins=[xedges, yedges])
    
    total_thermal_energy_bin = np.zeros_like(hist2d)
    thermal_energy_midpoints = 0.5 * (yedges[:-1] + yedges[1:])
    print(num_bins_x,num_bins_y)
    print(hist2d.shape)
    print(total_thermal_energy_bin.shape)
    print(thermal_energy_midpoints.shape)
    for i in range(num_bins_x):
        for j in range(num_bins_y):
            total_thermal_energy_bin[i, j] = hist2d[i, j] * thermal_energy_midpoints[j]

    
    # Plot the 2D histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use a colormap to show the counts in each bin
    #pcm = ax.pcolormesh(np.log10(xedges), np.log10(yedges), hist2d.T, cmap='viridis', shading='auto')
    
    # Use a colormap to show the total thermal energy in each bin
    pcm = ax.pcolormesh(np.log10(xedges), np.log10(yedges), total_thermal_energy_bin.T, cmap='viridis', shading='auto')
    

    # Add a color bar to show the count scale
    #cbar = plt.colorbar(pcm, ax=ax, label='Counts')
    cbar = plt.colorbar(pcm, ax=ax, label='thermal energy in bin [erg]')
    
    # Set plot labels and title
    ax.set_xlabel('log10 (T_DF) [K]',fontsize=13)
    ax.set_ylabel('log10 (Thermal Energy) [erg]',fontsize=13)
    ax.set_title(f'T_DF and Thermal Energy Distribution, z={current_redshift:.2f}')

    # Save the figure
    plt.tight_layout()
    #plt.savefig(output_dir + f'T_DF_ThermalEnergy_snap{snapNum}_z{current_redshift:.2f}.png', dpi=300)
    plt.savefig(output_dir + f'T_DF_ThermalEnergy_newcolor_snap{snapNum}_z{current_redshift:.2f}.png', dpi=300)
    
    
    #plot histogram of host vel
    # vel_host_list = np.array(vel_host_list)
    # fig = plt.figure(facecolor='white')
    # plt.hist(vel_host_list/1e3, bins=50)
    # plt.title(f'Host Halo Velocity Distribution, z={current_redshift:.2f}')
    # plt.xlabel('Host Halo Velocity [km/s]')
    # plt.ylabel('Counts')
    # plt.tight_layout()
    # plt.savefig(output_dir+f'Host_vel_snap{snapNum}_z{current_redshift:.2f}.png',dpi=300)

    #plot histogram of host vel relative to gas
    # vel_host_gas_list = np.array(vel_host_gas_list)
    # fig = plt.figure(facecolor='white')
    # plt.hist(vel_host_gas_list/1e3, bins=50)
    # plt.title(f'Host Halo Velocity Relative to Gas Distribution, z={current_redshift:.2f}')
    # plt.xlabel('Host Halo Velocity Relative to Gas [km/s]')
    # plt.ylabel('Counts')
    # plt.tight_layout()
    # plt.savefig(output_dir+f'Host_gas_vel_snap{snapNum}_z{current_redshift:.2f}.png',dpi=300)

    #plot heating
    # fig = plt.figure(facecolor='white')
    # plt.plot(logM_bin_centers, DF_heating_hostmassbin, marker='.')
    # plt.title(f'DF Heating, z={current_redshift:.2f}')
    # plt.xlabel('Host Halo Mass [$M_\odot/h$]')
    # plt.ylabel(r'$\epsilon_{DF}$ [erg/s]')
    # plt.tight_layout()
    # plt.savefig(f'./figures/DF_heating_usegas_z{current_redshift:.2f}.png',dpi=300)


