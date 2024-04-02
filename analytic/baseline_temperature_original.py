import numpy as np
import matplotlib.pyplot as plt
from math import pi, erfc
from scipy.integrate import solve_ivp, quad
import warnings
from scipy.special import gamma
from scipy.integrate import nquad
from physical_constants import *
from linear_evolution import *
from colossus.cosmology import cosmology
from colossus.lss import mass_function



z_list = np.logspace(0, 3.0, num=100)

T_CMB_list = T0_CMB * (1 + z_list)


# plt.plot(1+z_list, T_CMB_list,label='CMB')
# plt.xscale('log')
# plt.yscale('log')

# plt.xlabel('1+z')
# plt.ylabel('T')



# Define the ODE function (Compton scattering only)
def ode_function_Compton_only(z, T, xe, t_gamma_inv, H0):
    
    return 2 * T / (1 + z) - xe / (1 + xe + 0.079) * (2.7255 * (1 + z) - T) * t_gamma_inv * (1 + z) ** 3 / (H0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7))

# Define the range of z values
z_start = 1000.0
z_end = 6.0
num_points = 1000
z_values = np.logspace(np.log10(z_start), np.log10(z_end), num = num_points, base = 10)

# Define the initial condition
T0_z1000 = 2728.2255  # Initial temperature at z = 1000

# Define the parameters
xe = 1e-3
t_gamma_inv = 8.55e-13 / 3.1536e7  # Convert t_gamma_inv from yr^(-1) to s^(-1)

# Solve the ODE
solution = solve_ivp(lambda z, T: ode_function_Compton_only(z, T, xe, t_gamma_inv, H0_s), [z_start, z_end], [T0_z1000], t_eval=z_values)

# Extract the solution
z_values = solution.t
T_values = solution.y[0]

# Plot the Compton Scattering solution 
fig = plt.figure(facecolor='white')
plt.plot(1+z_values, T_values,label='gas')
plt.plot(1+z_list, T_CMB_list,label='CMB')
plt.xlabel('1+z')
plt.ylabel('T(z)')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.title('Temperature Evolution')
plt.grid(True)
plt.savefig("Temperature_Evolution_Compton_only.png",dpi=300)


#normalization
sigma8_nonormalization =  sigma_R_z_uselog(8/h_Hubble, 0)


sigma_normalization_factor = sigma8/sigma8_nonormalization
P_normalization_factor = sigma_normalization_factor**2



def sigma_M_z_normalized(M, z):
    #M in the unit of Msun
    global Omega_m, rho_crit_z0, sigma_normalization_factor
    Omega_m_z = Omega_m*(1+z)**3
    rho_m_z = Omega_m_z * rho_crit_z0
    R = (M/(4/3*np.pi*rho_m_z))**(1/3)
    
    return sigma_R_z_uselog(R, z)*sigma_normalization_factor

    
def logsigma_logM_z_normalized(logM, z):
    #M in the unit of Msun
    M = 10**(logM)
    global Omega_m, rho_crit_z0, sigma_normalization_factor
    Omega_m_z = Omega_m*(1+z)**3
    rho_m_z = Omega_m_z * rho_crit_z0
    R = (M/(4/3*np.pi*rho_m_z))**(1/3)
    
    return np.log10(sigma_R_z_uselog(R, z)*sigma_normalization_factor)


def derivative_ln_sigma_ln_M(logM, z=0):
    
    #M = 10**(logM)
    delta = 1e-5  # Choose an appropriate small value for delta
    
    logsigma_plus_delta = logsigma_logM_z_normalized(logM + delta, z)
    logsigma_minus_delta = logsigma_logM_z_normalized(logM - delta, z)
    
    derivative = (logsigma_plus_delta - logsigma_minus_delta) / (2 * delta)
    return derivative


def M_crit_z(z):
    #in the unit of Msun
    global Omega_m, Tvir_crit, mu, h_Hubble
    d = Omega_m_z(z) -1
    Delta_crit = 18*np.pi**2 + 82*d - 39*d**2
    
    factor = 1.98e4 * (mu/0.6) *(Omega_m/Omega_m_z(z) * Delta_crit/18/np.pi**2)**(1/3) * (1+z)/10
    Mcrit = (Tvir_crit/factor)**(3/2)
    Mcrit *= 1e8*h_Hubble**(-1)
    
    return Mcrit
     

def f_coll(z):
    global delta_c
    delta_c_z = delta_c/D_z(z)
    #delta_c_z = delta_c
    M_crit = M_crit_z(z)
    sigma_M = sigma_M_z_normalized(M_crit/(1+z)**3, 0.0) 
    #sigma is evaluated at z=0 since linear evolution has been included in D(z)
    #filter radius R is set s.t. 4/3 pi R^3 rho(z) = M(z), which is equivalent to 
    #4/3 pi R^3 rho(0) = M(z)/(1+z)^3
    fcoll = erfc(delta_c_z/np.sqrt(2)/sigma_M)
    
    return fcoll


def f_coll2(z):
    global delta_c
    #delta_c_z = delta_c/D_z(z)
    delta_c_z = delta_c
    M_crit = M_crit_z(z)
    sigma_M = sigma_M_z_normalized(M_crit, z) 
    fcoll = erfc(delta_c_z/np.sqrt(2)/sigma_M)
    
    return fcoll


def dfcoll_dz(z, h=1e-5):
    if (z > 50):
        return 0.0
    else:
        return (f_coll(z+h/2) - f_coll(z-h/2))/h
    


# Define the ODE function (Compton scattering + X-ray heating)
def ode_Compton_Xray(z, T, xe, t_gamma_inv, H0):
    dTdz_Compton = xe / (1 + xe + 0.079) * (2.7255 * (1 + z) - T) * t_gamma_inv * (1 + z) ** 3 / (H0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7))
    f_X = 1
    f_star = 0.1
    f_X_heating = 0.2
    dfcolldz = dfcoll_dz(z, 1e-5)
    dTdz_Xray = - 1e3 * f_X *(f_star/0.1)*(f_X_heating/0.2)*(dfcolldz/0.01)*(1/10)    
    return 2 * T / (1 + z) - dTdz_Compton - dTdz_Xray

# Define the range of z values
z_start = 1000.0
z_end = 10.0
num_points = 1000
z_values = np.logspace(np.log10(z_start), np.log10(z_end), num = num_points, base = 10)

# Define the parameters
xe = 1e-3
t_gamma_inv = 8.55e-13 / 3.1536e7  # Convert t_gamma_inv from yr^(-1) to s^(-1)

# Solve the ODE
solution = solve_ivp(lambda z, T: ode_Compton_Xray(z, T, xe, t_gamma_inv, H0_s), [z_start, z_end], [T0_z1000], t_eval=z_values)

# Extract the solution
z_values = solution.t
T_values = solution.y[0]


fig = plt.figure(facecolor='white')
plt.plot(1+z_values, T_values,label='gas')
plt.plot(1+z_list, T_CMB_list,label='CMB')
plt.xlabel('1+z')
plt.ylabel('T(z)')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.title('Temperature Evolution')
plt.grid(True)
plt.savefig("Temperature_Evolution_Compton_Xray.png",dpi=300)


def Overdensity_Virial(z):
    x = Omega_m_z(z) - 1
    return (18*np.pi**2 + 82 *x - 39*x**2)/(x+1)
    

def Vel_Virial(M_vir_in_Msun, z):
    #M_vir in solar mass, return virial velocity in m/s    
    global h_Hubble, Omega_m
    #Delta_vir = Overdensity_Virial(z)
    Delta_vir = 200
    V_vir = 163*1e3 * (M_vir_in_Msun/1e12*h_Hubble)**(1/3) * (Delta_vir/200)**(1/6) * Omega_m**(1/6) *(1+z)**(1/2)
    return V_vir


cosmology.setCosmology('planck18')
# mfunc_so = mass_function.massFunction(1E12, 0.0, mdef = 'vir', model = 'tinker08')
# mfunc_fof = mass_function.massFunction(1E12, 0.0, mdef = 'fof', model = 'watson13')

#dN/dM
def Halo_Mass_Function(M,z):
    #M in Msun
    #return number of halos in [M, M+dM] per (Mpc)^3
    global Omega_m, rho_crit_z0, delta_c
    average_density = Omega_m*(1+z)**3*rho_crit_z0
    delta_c_z = delta_c/D_z(z)
    sigma_M = sigma_M_z_normalized(M, 0)
    dlnSigma_dlnM = derivative_ln_sigma_ln_M(np.log10(M), 0)
    return np.sqrt(2/np.pi)*average_density/M**2 *delta_c_z/sigma_M *np.exp(-delta_c_z**2/2/sigma_M**2) *abs(dlnSigma_dlnM) 



z_PS = [0.0, 2.0, 4.0, 6.0]
logM_test = np.linspace(9.0, 16, 100)
M_test = 10**(logM_test)

M2_HMF_PS = []

for z in z_PS:
    M2_HMF_PS_z = np.array([M**2/(rho_m0*(1+z)**3) *Halo_Mass_Function(M,z) for M in M_test])
    M2_HMF_PS.append(M2_HMF_PS_z)

z_colossus = [0.0, 2.0, 4.0, 6.0]
M_colossus = 10**np.linspace(9.0, 16, 100)

fig = plt.figure(facecolor='white')
plt.xlabel('M [Msun/h]',fontsize=14)
plt.ylabel(r'$\frac{M^2}{\bar{\rho}_m} \ \frac{dn}{dM}$',fontsize = 16)
plt.loglog()
# plt.xlim(1E9, 1E16)
plt.ylim(1E-8, 1E-1)
for i in range(len(z_colossus)):
    mfunc = mass_function.massFunction(M_colossus, z_colossus[i], model = 'press74', q_out = 'M2dndM')
    plt.plot(M_colossus, mfunc, '-', label = 'z = %.1f' % (z_colossus[i]))

for i in range(len(z_PS)):
    plt.plot(M_test*h_Hubble,M2_HMF_PS[i],linestyle='--',color='grey')

plt.legend()
plt.title("solid: Colossus press74; dashed: press74")
plt.tight_layout()
plt.savefig("hmf_comparison_press74.png",dpi=300)


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
    

def Subhalo_Mass_Function_ln(ln_m_over_M):
    m_over_M = np.exp(ln_m_over_M)
    f0 = 0.1
    beta = 0.3
    gamma_value = 0.9
    x = m_over_M/beta
    return f0/(beta*gamma(1 - gamma_value)) * x**(-gamma_value) * np.exp(-x)



#subhalo mass function
sub_host_ratio = np.linspace(1e-5,1,100)
subHMF = np.array([Subhalo_Mass_Function(m_over_M) for m_over_M in sub_host_ratio])

fig = plt.figure(facecolor='white')
plt.plot(np.log(sub_host_ratio), subHMF)
#plt.xscale('log')
plt.yscale('log')
plt.xlabel('ln(m/M)',fontsize = 14)
plt.ylabel(r'$\frac{dn}{ d\ln (\frac{m}{M})}$',fontsize=16)

plt.tight_layout()
plt.savefig("subhalo_mass_function.png",dpi=300)

current_redshift = 15.0
logM_limits = [5, 11]  # Limits for log10(M [Msun/h])
HMF_lgM = []
logM_list = np.linspace(logM_limits[0], logM_limits[1],57)
for logM in logM_list:
    
    M = 10**(logM)
    HMF_lgM.append(HMF_Colossus(10**logM, current_redshift)* np.log(10)*M)  

fig = plt.figure(facecolor='white')
plt.plot(logM_list,HMF_lgM,label='z=15.0')
plt.legend()
plt.ylim([1e0,1e9])
plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$\log_{10}M \ (M_{\odot}/h)$',fontsize=14)
plt.ylabel(r'$\frac{dN}{d \log_{10}M} \ \ [(Mpc/h)^{-3}]$',fontsize=16)

plt.tight_layout()
plt.savefig("HMF_Colossus_z15.png",dpi=300)


#Dynamical Friction Heating
def I_DF(Vel,z):
    return 1.0


#input  ln(m/M), log10(M[Msun/h]), z
#output heating rate per volume [ J/s (Mpc/h)^3 ] after integrating host halo mass and subhalo mass

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


def integrand_singlehost(ln_m_over_M, logM, z):  #integrand without dM/dlgM (= HMF(M,z)*dM/dlgM)
    global G_grav,rho_b0,h_Hubble, Mpc, Msun
    
    eta = 1.0
    I_DF = 1.0
    
    M = 10**logM
    m_over_M = np.exp(ln_m_over_M)
    m = m_over_M * M  
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    DF_heating =  eta * 4 * np.pi * (G_grav * m *Msun/h_Hubble) ** 2 / Vel_Virial(M/h_Hubble, z) *rho_g *I_DF
    
    DF_heating *= Subhalo_Mass_Function_ln(ln_m_over_M) 
   
    
    return DF_heating


def integrated_subhalo(logM, z):  #without dM/dlgM (= HMF(M,z)*dM/dlgM)
    global G_grav,rho_b0,h_Hubble, Mpc, Msun
    
    eta = 1.0
    I_DF = 1.0
    
    M = 10**logM
#     m_over_M = np.exp(ln_m_over_M)
#     m = m_over_M * M  
    rho_g = 200 * rho_b0*(1+z)**3 *Msun/Mpc**3
    DF_heating =  eta * 4 * np.pi * (G_grav * M *Msun/h_Hubble) ** 2 / Vel_Virial(M/h_Hubble, z) *rho_g *I_DF
    
    DF_heating *= 0.00286998  #\int m^2 * subhalo_hmf(m/M) d(ln(m/M))
   
    
    return DF_heating



#check contribution to heating
z_values = [0,5,10]
z_value = 5.0
logM_limits = [2, 16]  # Limits for log10(M [Msun/h])
ln_m_over_M_limits = [-12, 0]  # Limits for m/M

logM_list = np.linspace(logM_limits[0], logM_limits[1],57)

DF_heating_perlogM_list = []

for z_value in z_values:
    DF_heating_perlogM = []
    
    for logM in logM_list:
        result, error = quad(integrand, ln_m_over_M_limits[0], ln_m_over_M_limits[1], args=(logM, z_value))
        if (abs(error) > 0.01 * abs(result)):
            print("Possible large integral error at z = %f, relative error = %f\n", z_value, error/result)

        DF_heating_perlogM.append(result)
    
    DF_heating_perlogM_list.append(np.array(DF_heating_perlogM))


fig = plt.figure(facecolor='white')

plt.plot(logM_list,1e7*DF_heating_perlogM_list[0],label='z=0')
plt.plot(logM_list,1e7*DF_heating_perlogM_list[1],label='z=5')
plt.plot(logM_list,1e7*DF_heating_perlogM_list[2],label='z=10')


plt.legend()
# plt.xlim([4,12])
plt.ylim([1e37,1e41])
plt.yscale('log')
plt.ylabel('DF heating per logM [erg/s (Mpc/h)^3]',fontsize=12)
plt.xlabel('logM [Msun/h]]',fontsize=12)


plt.savefig("DF_heating_perlogM_analytic.png",dpi=300)





              
'''              
#read data

logM_limits = [2, 16]  # Limits for log10(M [Msun/h])

ln_m_over_M_limits = [-12, 0]  # Limits for m/M
#m_over_M_limits = [1e-12,1]

# Define the value of z
z_value_list = np.linspace(0,40,201) 
DF_heating_list = []
error_list = []

SubhaloDF_data = np.loadtxt('Subhalo_DF_heating.txt', skiprows=1)

# Extract the columns
z_value_list = SubhaloDF_data[:, 0]
DF_heating_list = SubhaloDF_data[:, 1]
error_list = SubhaloDF_data[:, 2]




plt.scatter(logM_list,DF_heating_perlogM_list[0],label='z=0')
plt.scatter(logM_list,DF_heating_perlogM_list[1],label='z=5')
plt.scatter(logM_list,DF_heating_perlogM_list[2],label='z=10')



plt.legend()
# plt.xlim([4,12])
plt.ylim([1e25,1e34])
plt.yscale('log')





#plt.scatter(logM_list,DF_heating_singlehost_perlogM * HMF_Mlist)
plt.scatter(logM_list,DF_heating_singlehost_perlogM_list[0],s=10,label='z=0')
plt.scatter(logM_list,DF_heating_singlehost_perlogM_list[1],s=10,label='z=5')
plt.scatter(logM_list,DF_heating_singlehost_perlogM_list[2],s=10,label='z=10')

# plt.plot(logM_list,DF_heating_singlehost_perlogM_list_check[0],label='z=0')
# plt.plot(logM_list,DF_heating_singlehost_perlogM_list_check[1],label='z=5')
# plt.plot(logM_list,DF_heating_singlehost_perlogM_list_check[2],label='z=10')

plt.legend()
#plt.ylim([1e25,1e34])

plt.yscale('log')
#plt.xscale('log')



#plt.scatter(logM_list,DF_heating_singlehost_perlogM * HMF_Mlist)
plt.plot(logM_list,HMF_lgM_list[0],label='z=0')
plt.plot(logM_list,HMF_lgM_list[1],label='z=5')
plt.plot(logM_list,HMF_lgM_list[2],label='z=10')
plt.legend()
#plt.ylim([1e25,1e34])
plt.ylim([1e-12,1e15])
plt.yscale('log')
#plt.xscale('log')
plt.xlabel(r'$\log_{10}M \ (M_{\odot}/h)$',fontsize=14)
plt.ylabel(r'$\frac{dN}{d \log_{10}M} \ \ [(Mpc/h)^{-3}]$',fontsize=16)

#output dn/dM in the unit of [(Mpc/h)^(-3) (Msun/h)^(-1)]
plt.tight_layout()




# with open('Subhalo_DF_heating.txt', 'w') as file:
    
#     file.write("1. redshift z\t2. Subhalo DF heating rate [J/s (Mpc/h)^3]\t 3. error of Subhalo DF heating \n")
#     for z, DF_heating, error in zip(z_value_list, DF_heating_list, error_list):
#         file.write("{:.2f}\t{:.8e}\t{:.8e}\n".format(z, DF_heating, error))



plt.plot(z_value_list, DF_heating_list)
#plt.yscale('log')
plt.xlabel('z',fontsize=14)
plt.ylabel(r'subhalo DF heating rate [ J/s (Mpc/h)$^3$]',fontsize=12)

#plt.savefig("SubhaloDF_heating_rate.png",dpi=300)



#number of baryon per Mpc^3
def ngas(z):
    global rho_b0, mu, Msun,mp
    return rho_b0*(1+z)**3*Msun/(mu*mp)
    

# only for 10 < z < 30
def func_dTdz_SubhaloDF(z):
    global kB, h_Hubble
    dTdz = 0.0
    if(z > 10 and z<30):
        epsilon_SubhaloDF = 10**(-0.11*z + 34.8)
        dTdt = 2/3*epsilon_SubhaloDF/(kB*ngas(z)) *h_Hubble**3
        dTdz = dTdt/(1+z)/H(z)
    
    return dTdz




z_list = np.logspace(0, 3.0, num=100)

T_CMB_list = T0_CMB * (1 + z_list)


# Define the ODE function
def ode_Compton_Xray(z, T, xe, t_gamma_inv, H0):
    dTdz_Compton = xe / (1 + xe + 0.079) * (2.7255 * (1 + z) - T) * t_gamma_inv * (1 + z) ** 3 / (H0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7))
    f_X = 1
    f_star = 0.1
    f_X_heating = 0.2
    dfcolldz = dfcoll_dz(z, 1e-5)
    
    dTdz_Xray = 1e3 * f_X *(f_star/0.1)*(f_X_heating/0.2)*(-dfcolldz/0.01)*(1/10)
    
    return 2 * T / (1 + z) - dTdz_Compton - dTdz_Xray


def ode_Compton_Xray_SubhaloDF(z, T, xe, t_gamma_inv, H0):
    dTdz_Compton = xe / (1 + xe + 0.079) * (2.7255 * (1 + z) - T) * t_gamma_inv * (1 + z) ** 3 / (H0 * np.sqrt(0.3 * (1 + z) ** 3 + 0.7))
    f_X = 1
    f_star = 0.1
    f_X_heating = 0.2
    dfcolldz = dfcoll_dz(z, 1e-5)
    
    dTdz_Xray = 1e3 * f_X *(f_star/0.1)*(f_X_heating/0.2)*(-dfcolldz/0.01)*(1/10)
    
    dTdz_SubhaloDF = func_dTdz_SubhaloDF(z)    
    
    return 2 * T / (1 + z) - dTdz_Compton - dTdz_Xray -dTdz_SubhaloDF

# Define the range of z values
z_start = 1000.0
z_end = 10
num_points = 1000
z_values = np.logspace(np.log10(z_start), np.log10(z_end), num = num_points, base = 10)
t_span=[z_start, z_end*0.99]

# Define the initial condition

# Solve the ODE
solution = solve_ivp(lambda z, T: ode_Compton_Xray(z, T, xe, t_gamma_inv, H0_s), t_span,[T0_z1000], t_eval=z_values,method='RK45')

# Extract the solution
z_values_noDF = solution.t
T_values_noDF = solution.y[0]


solution = solve_ivp(lambda z, T: ode_Compton_Xray_SubhaloDF(z, T, xe, t_gamma_inv, H0_s),t_span, [T0_z1000], t_eval=z_values,method='RK45')
z_values_DF = solution.t
T_values_DF = solution.y[0]




0.99*30.3



plt.plot(1+z_values_noDF, T_values_noDF,'b-',label='gas, no Subhalo DF')
plt.plot(1+z_values_DF, T_values_DF,'r--',label='gas, Subhalo DF')

plt.plot(1+z_list, T_CMB_list,'g-',label='CMB')
plt.xlabel('1+z')
plt.ylabel('T(z)')
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.title('Temperature Evolution')
plt.grid(True)

#plt.savefig("SubhaloDF_heating_impact.png",dpi=300)





relative_T_DF = (T_values_DF - T_values_noDF)/T_values_noDF
plt.plot(1+z_values_DF, relative_T_DF)
plt.xscale('log')
plt.xlabel("1+z",fontsize=14)
plt.ylabel(r"$\frac{T_{k,DF}-T_{k,fid}}{T_{k,fid}}$",fontsize=17)
plt.tight_layout()
#plt.savefig("Tk_SubhaloDF.png",dpi=300)


np.where(relative_T_DF<0)




adiabatic_index = 5/3
#gas_constant = 8.314462618
mu = 1.
xe_z0 = 1
eV = 1.60218e-19
cm = 1e-2









4*np.pi*G_grav**2*(1e11*Msun)**2*500*2*rho/Cs


HMF_lgM_list[0][49]*rho_gas_ratio*Mass_ratio





'''
