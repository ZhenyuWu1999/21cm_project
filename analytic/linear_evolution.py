import numpy as np
import matplotlib.pyplot as plt
from math import pi, erfc
from scipy.integrate import solve_ivp, quad
import warnings
from scipy.special import gamma
from scipy.integrate import nquad
from physical_constants import *
from colossus.cosmology import cosmology
from colossus.lss import mass_function

def E(z):
    global Omega_lambda,Omega_m,Omega_k,Omega_r
    return np.sqrt(Omega_lambda + Omega_k*(1+z)**2 + Omega_m*(1+z)**3 + Omega_r*(1+z)**4)

def H(z):
    global Omega_lambda,Omega_m,Omega_k,Omega_r
    global H0_s
    return H0_s*E(z)

def dt_dz(z):
    return -1/(1+z)/H(z)

def Omega_lambda_z(z):
    global Omega_lambda,Omega_m,Omega_k,Omega_r
    return Omega_lambda/E(z)**2
    

def Omega_m_z(z):
    global Omega_lambda,Omega_m,Omega_k,Omega_r
    return Omega_m*(1+z)**3/E(z)**2
    
                                                                        
def g_z(z):  #Carroll; MBW4.1.6
    global Omega_lambda,Omega_m,Omega_k,Omega_r
    Omz = Omega_m_z(z)
    Olz = Omega_lambda_z(z)
    return 2.5*Omz * (Omz**(4/7) - Olz + (1 + Omz/2)*(1 + Olz/70) )**(-1)

def D_z(z):
    global Omega_lambda,Omega_m,Omega_k,Omega_r
    D_0 = g_z(0)
    return g_z(z) / (1+z) / D_0

    

# double check: Peebles 1980; Barkana 2001

# Define the function to integrate
def D_z_integral_func(a, Omega_lambda, Omega_k, Omega_m):
    return a**(3/2) / (Omega_lambda * a**3 + Omega_k *a + Omega_m )**(3/2)

def D_z_Peebles_nonormalization(z):
    global Omega_lambda,Omega_m,Omega_k,Omega_r
    a = 1/(1+z)
    integral, error =  quad(D_z_integral_func, 0, a, args=(Omega_lambda, Omega_k, Omega_m))
    D = np.sqrt(Omega_lambda*a**3 + Omega_k*a + Omega_m)/a**(3/2) * integral
    return D

def D_z_Peebles(z):
    global Omega_lambda,Omega_m,Omega_k,Omega_r
    a = 1/(1+z)    
    # Perform the integration from a = 0 to a = a(z)
    integral, error =  quad(D_z_integral_func, 0, a, args=(Omega_lambda, Omega_k, Omega_m))
    D = np.sqrt(Omega_lambda*a**3 + Omega_k*a + Omega_m)/a**(3/2) * integral
    D /= D_z_Peebles_nonormalization(0)
    return D


#initial power spectrum:
def P1(k):
    n = 1
    return k**n

#linear transfer function:
#MBW 4.3.2, CDM
def Transfer_func(k):
    #k in the unit of Mpc^(-1)
    global Omega_m, Omega_b, h_Hubble
    k /= h_Hubble**2
    shape_parameter = Omega_m * np.exp(- Omega_b *(1 + np.sqrt(2*h_Hubble)/Omega_m))
    q = k/shape_parameter
    
    T_k = np.log(1 + 2.34*q)/(2.34*q) *(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4 )**(-1/4) 
    
    return T_k

def Power_Spectrum(k, z):
    #k in the unit of Mpc^(-1)
    return P1(k) * Transfer_func(k)**2 * D_z(z)**2
    

# window function (top hat) in k-space
def W_th_k(k, R):
    #k in the unit of Mpc^(-1)
    #R in the unit of Mpc
    return 3/(k*R)**3 * (np.sin(k*R) - k*R*np.cos(k*R))
    


def sigma_integral(k,R,z):
    return Power_Spectrum(k, z)* W_th_k(k,R)**2 * k**2

def sigma_integral_log(lgk, R, z):
    k = 10**(lgk)
    return Power_Spectrum(k, z)* W_th_k(k,R)**2 * k**2 *k*np.log(10)  ## change to variable (log10 k)

    
def sigma_R_z(R, z):
    #R in the unit of Mpc
    sigma2, error = quad(sigma_integral, 0.0, np.inf, args=(R, z))
#     if error > 1.0e-3*sigma2:
#         warnings.warn("density variance integral error may be too large!")
#         print("integrate in linear scale: R, relative error = ",R, error/sigma2)
    sigma2 /= (2*np.pi**2)
    sigma = np.sqrt(sigma2)
    return sigma
    

def sigma_R_z_uselog(R, z):
    #R in the unit of Mpc
    sigma2, error = quad(sigma_integral_log, -10, 10, args=(R, z))
#     if error > 1.0e-3*sigma2:
#         warnings.warn("density variance integral error may be too large!")
#         print("integrate in log scale: R, relative error = ",R, error/sigma2)
    sigma2 /= (2*np.pi**2)
    sigma = np.sqrt(sigma2)
    return sigma   


#normalization
sigma8_nonormalization =  sigma_R_z_uselog(8/h_Hubble, 0)

sigma8 = 0.812  #Planck 2018  ???

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


def df_dz(z):
    delta_z = 1.0e-2
    return (f_coll(z+delta_z) - f_coll(z-delta_z))/(2*delta_z)