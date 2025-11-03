import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator
from TNG_plots import maxwell_boltzmann_pdf, truncated_gaussian_pdf

def h_test(sz, rcyl, mach):
    """The constant for the integrand of the dynamical friction integral.
    h = 1: the ice-cream
    h = 2: the cone."""
    h = 0
    if (rcyl**2 + (sz + mach)**2) <= 1 or mach <= 1:
        h = 1
    elif (rcyl**2 + (sz + mach)**2) > 1 and (sz + mach)>1/mach and sz/rcyl < -np.sqrt(mach**2 - 1):
        h = 2
    else:
        return h
    # a = (mach**2*rcyl*sz)/((rcyl**2 + sz**2)**1.5*np.sqrt((1 - mach**2)*rcyl**2 + sz**2))
    return h 

def integrandmux(mu, x, mach, h):
    """The integrand as a function of mu and x (as well as h and the mach number)."""

    return h * mu * mach**2/np.sqrt(1 - mach**2 + mu**2 * mach**2)/x

def Idf_Ostriker99(mach, xmin):
    """The full solution for the normalised force from Ostriker 1999"""
    #xmin = rmin/(Cs*t)
    if mach > 1.:
        fdf = 0.5 * np.log((1. + mach)/(mach - 1.)) + np.log((mach - 1.)) - np.log(xmin)
    elif mach == 1.0:
        fdf = np.inf
    else:
        fdf = 0.5 * np.log((1. + mach)/(1 - mach)) - mach
    return fdf

def Idf_Ostriker99_nosingularity(mach, xmin):
    if not (mach >= 0):
        raise ValueError("mach must be positive")
    delta = 0.05 #avoid singularity at mach = 1
    if mach > 1 - delta and mach <= 1:
        mach = 1 - delta
    if mach < 1 + delta and mach >= 1:
        mach = 1 + delta
    if (mach > 1 and mach -1 < xmin):
        print("Warning: Mach -1 < xmin in I_Ostriker99_supersonic; reset mach to 1+xmin")
        print("Mach:", mach, "xmin:", xmin)
        mach = 1 + xmin
    return Idf_Ostriker99(mach, xmin)       

def Idf_Ostriker99_nosingularity_Vtrmin(mach, Vt_rmin):
    Cst_rmin = Vt_rmin / mach
    xmin = 1/Cst_rmin
    return Idf_Ostriker99_nosingularity(mach, xmin)


def Idf_Ostriker99_wrapper(mach, rmin, Cs, t):
    if not (mach >= 0):
        raise ValueError("mach must be positive")
    delta = 0.05 #avoid singularity at mach = 1
    if mach > 1 - delta and mach <= 1:
        mach = 1 - delta
    if mach < 1 + delta and mach >= 1:
        mach = 1 + delta
    if (mach > 1 and mach -1 < rmin/t/Cs):
        print("Warning: Mach -1 < rmin/t/Cs in I_Ostriker99_supersonic.")
        raise ValueError("Mach -1 < rmin/t/Cs")
    
    xmin = rmin/(Cs*t)
    return Idf_Ostriker99(mach, xmin)
    
def Idf_R_Kim2007(mach):
    """
    Kim&Kim 2007 model for point mass perturber in a circular orbit
    mach: Mach number; Rp_rmin: ratio of orbital radius to rmin
    """
    if mach < 1.1:
        I_R = mach**2 * 10**(3.51*mach - 4.22)
    elif 1.1 <= mach < 4.4:
        I_R = 0.5 * np.log(9.33 * mach**2*(mach**2 - 0.95))
    elif mach >= 4.4:
        I_R = 0.3 *mach**2
    else:
        raise ValueError("mach must be positive")
    I_R = max(0, I_R)
    return I_R

def Idf_phi_Kim2007(mach, Rp_rmin):
    """
    Kim&Kim 2007 model for point mass perturber in a circular orbit
    mach: Mach number; Rp_rmin: ratio of orbital radius to rmin
    """
    #smooth between 1-delta and 1+delta to make the function continuous
    delta = 0.05 
    if mach > 1 - delta and mach <= 1:
        mach = 1 - delta
    if mach < 1 + delta and mach >= 1:
        mach = 1 + delta

    if mach < 1.0:
        I_phi = 0.7706 * np.log((1+mach)/(1.0004 - 0.9185*mach)) - 1.4703*mach
    elif 1.0 <= mach < 4.4:
        I_phi = np.log(330*Rp_rmin*(mach - 0.71)**5.72*mach**(-9.58))
    elif mach >= 4.4:
        I_phi = np.log(Rp_rmin/(0.11*mach + 1.65))
    else:
        raise ValueError("mach must be positive")
    I_phi = max(0, I_phi)

    return I_phi



def iA_analytic(mach):
    """The analytic expression for the force from the c_s * t circle (the ice cream)"""

    iA = 0.5 * np.log((mach + 1.)/(mach - 1.))

    return iA

def iAii_analytic(mach):
    """The analytic expression for the force from the region of the c_s * t circle
    within (M^2 - 1) * c_s * t from the perturber"""

    iA =  0.5 *(np.sqrt(-1 + mach**2) - mach + np.log(np.sqrt(-1 + mach**2)/(mach - 1)))

    return iA

def iAi_analytic(mach):
    """The analytic expression for the force from the region of the c_s * t circle
    beyond (M^2 - 1) * c_s * t from the perturber"""

    iA =  0.5 *(np.sqrt(-1 + mach**2) - mach + np.log((1 + mach)/np.sqrt(-1 + mach**2)))

    return iA

def iBii_analytic(mach):
    """The analytic expression for the force from the cone region further than
    V * t - c_s * t away from the perturber"""

    iBii = (mach - np.sqrt(mach**2 - 1))

    return iBii

def iBi_analytic(mach, xmin):
    """The analytic expression for the force from the cone region closer than
    V * t - c_s * t away from the perturber """

    iBi = np.log((mach - 1.)/xmin)

    return iBi

def fdf_to_x(xval, mach, xmin):
    """Calculate the contribution to the force from within x=r/(c_s * t).

    Makes use of scipy function 'dblquad':
    dblquad(integrand,
            lower limit of x,
            upper limit of x,
            lower limit of mu (a function of x),
            upper limit of mu (a function of x),
            arguments of the integrand function (mach and h))

    """

    # Only works for supersonic setup
    if mach >= 1:
        # Inside of xmin there is no force
        if xval <= xmin:
            integral = 0.
        # between x= xmin and x = mach - 1
        elif xval <= mach -1:
            iBi = dblquad(integrandmux, 
                          xmin,
                          xval,
                          lambda mu: -1,
                          lambda mu: -np.sqrt(mach**2-1)/mach,
                          args=(mach, 2))
            integral = -0.5 * iBi[0]
        # between x = mach - 1 and x = mach**2 - 1
        elif xval <= np.sqrt(mach**2 - 1):
            iBi = iBi_analytic(mach, xmin) 
            iBii = dblquad(integrandmux,
                          mach - 1,
                          xval,
                          lambda x: (1 - mach**2-x**2)/(2.*x*mach),
                          lambda x: -np.sqrt(mach**2-1)/mach,
                          args=(mach, 2))
            iA = dblquad(integrandmux,
                         mach - 1,
                         xval,
                         lambda x: -1,
                         lambda x: (1 - mach**2-x**2)/(2.*x*mach),
                         args=(mach, 1))
            integral = iBi - 0.5 * iBii[0] - 0.5 * iA[0]
        # between x = mach**2 - 1 and x = mach + 1
        elif xval <= mach + 1:
            iBi = iBi_analytic(mach, xmin) 
            iBii = iBii_analytic(mach)
            iA = dblquad(integrandmux,
                         mach - 1,
                         xval,
                         lambda x: -1,
                         lambda x: (1 - mach**2-x**2)/(2.*x*mach),
                         args=(mach, 1))
            integral = iBi + iBii - 0.5 * iA[0]
        # Larger x than the wake
        else:
            iBi = iBi_analytic(mach, xmin) 
            iBii = iBii_analytic(mach)
            iA = iAi_analytic(mach) + iAii_analytic(mach)
            integral = iBi + iBii + iA

    else:
        integral = 0 
        print("Error, only applicable for mach > 1")

    return integral

def i_indef_sol(x, m):
    """Analytic calculation from integrating Ostriker1999 equation 13 without
    setting x limits.
    """
    sol = -0.5*((m**2 + x**2 - 1.)/x/2.  - np.log(x))
    return sol

def fdf_subsonic_from_xinn_to_xout(xout, xinn, mach):
    """For the subsonic case, calculates the contribution to the force from
    between xout=rout/(c_s * t) and xinn.
    """
    xmin = 1. - mach
    if xinn < xmin:
        xinn = xmin # no contribution from inside 1 - mach

    if xout <= xinn:
        print("Warning, xout must be larger than xinn and xmin; return 0")
        return 0.

    if xout > mach + 1.:
        xout = mach + 1 # no contribution from outside 1 + mach 

    # Only works for subsonic setup
    if mach > 1:
        print("Error, only applicable for mach < 1")
        return 0.

    # analytic calculation from integrating Ostriker1999 equation 13 and
    # setting the x limits as xin and xout
    return i_indef_sol(xout, mach) - i_indef_sol(xinn, mach)

def fdf_supersonic_from_xinn_to_xout(xout, xinn, mach, xmin):
    """Calculate the contribution to the force from between xout=rout/(c_s * t) and xinn.

    Makes use of scipy function 'dblquad':
    dblquad(integrand,
            lower limit of x,
            upper limit of x,
            lower limit of mu (a function of x),
            upper limit of mu (a function of x),
            arguments of the integrand function (mach and h))

    Splits up integral into the 4 regions and tests which region xinn and xout are in before
    calculating the full force.

    """

    if xinn < xmin:
        xinn = xmin

    if xout <= xinn:
        print("Warning, xout must be larger than xinn and xmin; return 0")
        return 0.

    # Only works for supersonic setup
    if mach <= 1:
        print("Error, only applicable for mach > 1; return 0")
        return 0.
    
    # Calculate IBi
    if xinn < mach - 1:
        if xout < mach - 1:
            iBi = - 0.5 * dblquad(integrandmux, 
                          max(xinn, xmin),
                          xout,
                          lambda mu: -1,
                          lambda mu: -np.sqrt(mach**2-1)/mach,
                          args=(mach, 2))[0]
        else:
            iBi = iBi_analytic(mach, xinn) 
    else:
        iBi = 0.

    # Calculate IBii
    if xinn < np.sqrt(mach**2 - 1) and xout > mach - 1:
        if xout < np.sqrt(mach**2 - 1):
            iBii = - 0.5 * dblquad(integrandmux,
                          max((mach - 1, xinn)),
                          xout,
                          lambda x: (1 - mach**2-x**2)/(2.*x*mach),
                          lambda x: -np.sqrt(mach**2-1)/mach,
                          args=(mach, 2))[0]
        elif xinn > mach - 1:
            iBii = - 0.5 * dblquad(integrandmux,
                          xinn,
                          np.sqrt(mach**2 - 1),
                          lambda x: (1 - mach**2-x**2)/(2.*x*mach),
                          lambda x: -np.sqrt(mach**2-1)/mach,
                          args=(mach, 2))[0]
        else:
            iBii = iBii_analytic(mach)
    else:
        iBii = 0.

    # Calculate IA

    if xinn < mach + 1 and xout > mach - 1:
        if xout < mach + 1:
            iA = - 0.5 * dblquad(integrandmux,
                         max((mach - 1, xinn)),
                         xout,
                         lambda x: -1,
                         lambda x: (1 - mach**2-x**2)/(2.*x*mach),
                         args=(mach, 1))[0]
        elif xinn > mach - 1:
            iA = - 0.5 * dblquad(integrandmux,
                          xinn,
                          mach + 1,
                         lambda x: -1,
                         lambda x: (1 - mach**2-x**2)/(2.*x*mach),
                          args=(mach, 1))[0]
        else:
            iA = iAi_analytic(mach) + iAii_analytic(mach)
    else:
        iA = 0.


    integral = iBi + iBii + iA

    return integral


def volume_integral_subsonic(x, mach):
    #in unit of (c_s * t)**3
    if x <= 1 - mach:
        return 4*np.pi/3*x**3
    
    elif x <= 1 + mach:
        inner_sphere = 4*np.pi/3*(1 - mach)**3
        Cs_constained = np.pi/mach
        xin = 1 - mach
        term1 = -1.0/4.0*(x**4 - xin**4)
        term2 = 2.0*mach/3.0*(x**3 - xin**3)
        term3 = (1 - mach**2)/2.0*(x**2 - xin**2)
        Cs_constained *= (term1 + term2 + term3)
        return inner_sphere + Cs_constained
    
    elif x > 1+mach:
        return 4*np.pi/3
    else:
        raise ValueError("x must be positive")

def volume_integral_supersonic(x, mach, xmin):
    #in unit of (c_s * t)**3
    cos_theataM = np.sqrt(mach**2 - 1)/mach
    if x <= np.sqrt(mach**2 - 1):
        return 2*np.pi/3*(1 - cos_theataM)*x**3
    elif x <= 1 + mach:
        Vol_Mcone = 2*np.pi/3*(1 - cos_theataM)*np.sqrt(mach**2 - 1)**3
        Vol_Cssphere = np.pi/mach
        xin = np.sqrt(mach**2 - 1)
        term1 = -1.0/4.0*(x**4 - xin**4)
        term2 = 2.0*mach/3.0*(x**3 - xin**3)
        term3 = (1 - mach**2)/2.0*(x**2 - xin**2)
        Vol_Cssphere *= (term1 + term2 + term3)
        return Vol_Mcone + Vol_Cssphere
    elif x > 1+mach:
        return np.pi/3*(mach + 2.0 + 1.0/mach)
    else:
        raise ValueError("x must be positive")
        


def plot_subsonic_schematic(mach):
    
    #plot schematic figure of the different regions
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    x_list = np.linspace(1e-10, (1+mach)*1.5, 100)
    
    print(x_list)
    fdf_x = np.array([fdf_subsonic_from_xinn_to_xout(xout=xval, xinn=0.0, mach=mach) for xval in x_list])
    
    ax.plot(x_list, fdf_x)
    ymin = -0.01
    ax.set_ylim(ymin, None)
    #horizontal line 
    max_I_DF = 0.5 * np.log((1 + mach)/(1 - mach)) - mach
    ax.hlines(y=max_I_DF, xmin=0, xmax=(1+mach), color='k', linestyle='--')
    ax.text(0.5, max_I_DF*0.9, r'$\frac{1}{2}$ ln($\frac{1+\mathcal{M}}{1-\mathcal{M}}$) - $\mathcal{M}$', fontsize=12, ha='center', color='#1f77b4')
    
    # vertical line at x = 1 - mach and x = 1 + mach
    ax.vlines(x=1-mach, ymin=ymin, ymax=0, color='k', linestyle='--')
    ax.vlines(x=1+mach, ymin=ymin, ymax=max_I_DF, color='k', linestyle='--')
    #add text to the vertical lines
    ax.text(1-mach, ymin/2, r'x = 1-$\mathcal{M}$', fontsize=12, ha='right')
    ax.text(1+mach, max_I_DF/2, r'x = 1+$\mathcal{M}$', fontsize=12, ha='left')
    
    ax.set_xlabel(r"x = r/($c_s$ t)",fontsize=14)
    ax.set_ylabel(r"$I_{DF}$ (x)",fontsize=14,color= '#1f77b4')
    ax.set_title(r"Subsonic DF force ($\mathcal{M} = $"+f"{mach})",fontsize=14)
    
    #then plot the volume on the right side
    ax2 = ax.twinx()
    ax2.plot(x_list, [volume_integral_subsonic(x, mach) for x in x_list], color='r')
    ax2.set_ylabel(r"Volume [($C_s$ t)$^3$]",fontsize=14,color='r',alpha=0.8)
    ax2.set_ylim(-1, 5)
    #horizontal line
    ax2.hlines(y=4*np.pi/3*(1-mach)**3, xmin=(1-mach), xmax=x_list[-1], color='r', linestyle='--')
    ax2.text((1-mach)*3, 4*np.pi/3*(1-mach)**3*1.5, r'4$\pi$ (1-$\mathcal{M}$)$^3$/3', fontsize=12, ha='right', color='r')
    ax2.text(1+mach, 4*np.pi/3, r'4$\pi$/3', fontsize=12, ha='left', color='r')
    
    plt.tight_layout()
    filename = f"subsonic_schematic_M{mach}.pdf"
    plt.savefig(filename)
    
      
def plot_supersonic_schematic(mach, xmin):
    if (mach -1 <= xmin):
        print("Error: xmin must be smaller than mach - 1;")
        raise ValueError("xmin must be smaller than mach - 1")
    
    print(f"calculating for mach = {mach}, xmin = {xmin} ...")
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    x_list = np.linspace(1e-5, (1+mach)*1.5, 40)
    
    fdf_x = np.array([fdf_supersonic_from_xinn_to_xout(xout=xval, xinn=xmin, mach=mach, xmin=xmin) for xval in x_list])
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.plot(x_list, fdf_x)
    
    #horizontal line
    max_I_DF = Idf_Ostriker99(mach, xmin)
    ax.hlines(y=max_I_DF, xmin=0, xmax=(1+mach), color='#1f77b4', linestyle='--')
    ax.text(0.5, max_I_DF*0.9, r'$\frac{1}{2}$ ln($\frac{\mathcal{M}+1}{\mathcal{M}}-1$) + ln($\frac{\mathcal{M} - 1}{x_{\min}}$)', fontsize=12, ha='center', color='#1f77b4')
    
    ax.vlines(x=xmin, ymin=-0.01, ymax=max_I_DF, color='k', linestyle='--')
    ax.text(xmin, 0.0, r'$x_{\min}$', fontsize=12, ha='right')
    ax.vlines(x=np.sqrt(mach**2 - 1.), ymin=-0.01, ymax=max_I_DF, color='k', linestyle='--')
    ax.text(np.sqrt(mach**2 - 1.), 0.5, r'$\sqrt{\mathcal{M}^2 - 1}$', fontsize=12, ha='right')
    ax.vlines(x=1+mach, ymin=-0.01, ymax=max_I_DF, color='k', linestyle='--')
    ax.text(1+mach, 1.0, r'$1+\mathcal{M}$', fontsize=12, ha='right')
    
    ax.set_xlabel(r"x = r/($c_s$ t)",fontsize=14)
    ax.set_ylabel(r"$I_{DF}$ (x)",fontsize=14,color= '#1f77b4')
    ax.set_title(r"Supersonic DF force ($\mathcal{M} = $"+f"{mach}"+r" $x_{\min} = $"+f"{xmin})",fontsize=14)
    
    #then plot the volume on the right side
    
    ax2 = ax.twinx()
    ax2.plot(x_list, [volume_integral_supersonic(x, mach, xmin) for x in x_list], color='r')
    ax2.set_ylabel(r"Volume [($C_s$ t)$^3$]",fontsize=14,color='r',alpha=0.8)
    Vmax = np.pi/3*(mach + 2.0 - 1.0/mach)
    ax2.set_ylim(0, Vmax*1.5)
    #horizontal line
    ax2.hlines(y=2*np.pi/3*(1 - np.sqrt(mach**2 - 1)/mach)*np.sqrt(mach**2 - 1)**3, xmin=np.sqrt(mach**2 - 1), xmax=x_list[-1], color='r', linestyle='--')
    ax2.text(1, 2*np.pi/3*(1 - np.sqrt(mach**2 - 1)/mach)*np.sqrt(mach**2 - 1)**3*1.5, r'2$\pi$/3 (1 - $\cos(\theta_{\mathcal{M}}$))$\sqrt{\mathcal{M}^2 - 1}^3$', fontsize=12, ha='left', color='r')
    ax2.text(1+mach, Vmax*1.2, r'$\frac{\pi}{3}(\mathcal{M} + 2 + \frac{1}{\mathcal{M}})$', fontsize=12, ha='left', color='r')
    
    
    plt.tight_layout()
    filename = f"supersonic_schematic_M{mach}_xmin{xmin}.pdf"
    plt.savefig(filename)
   
   
def plot_subsonic_allmach():
    mach_array = np.array([0.1,0.5,0.9,0.99])
    colors = ['blue', 'green', 'yellow', 'red']
    
    x_list = np.linspace(1e-10, 1+np.max(mach_array)*1.2, 100)
    x_low = [1-mach for mach in mach_array]  
    x_high = [1+mach for mach in mach_array]
    max_I_DF = [0.5 * np.log((1 + mach)/(1 - mach)) - mach for mach in mach_array]
    
    I_for_all_mach = []
    for mach in mach_array:
        print(f"calculating for mach = {mach} ...")
        I_for_all_mach.append(np.array([fdf_subsonic_from_xinn_to_xout(xout=xval, xinn=1-mach, mach=mach) for xval in x_list]))
    
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for i, mach in enumerate(mach_array):
        ax.plot(x_list, I_for_all_mach[i], color=colors[i], label=f"$\mathcal{{M}} = {mach}$")
       
        ax.vlines(x=x_high[i], ymin=-0.01, ymax=max_I_DF[i], color=colors[i], linestyle='--')
        
    ax.set_xlabel(r"x = r/($c_s$ t)",fontsize=14)
    ax.set_ylabel(r"$I_{DF}$ (x)",fontsize=14)
    ax.set_title(r"Subsonic DF force",fontsize=14)
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    filename = "subsonic_FDF_allmach_ylog.pdf"
    plt.savefig(filename)
    
    #then plot the volume
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for i, mach in enumerate(mach_array):
        ax.plot(x_list, [volume_integral_subsonic(x, mach) for x in x_list], color=colors[i], label=f"$\mathcal{{M}} = {mach}$")
        ax.vlines(x=x_low[i], ymin=-1, ymax=4*np.pi/3*(1-mach)**3, color=colors[i], linestyle='--')
        ax.vlines(x=x_high[i], ymin=-1, ymax=4*np.pi/3, color=colors[i], linestyle='--')

    ax.set_xlabel(r"x = r/($c_s$ t)",fontsize=14)
    ax.set_ylabel(r"Volume [($C_s$ t)$^3$]",fontsize=14)
    ax.set_title(r"Subsonic Wake Volume",fontsize=14)
    ax.legend()
    plt.tight_layout()
    filename = "subsonic_volume_allmach.pdf"
    plt.savefig(filename)   
       

def plot_supersonic_allmach():
    mach_array = np.array([1.2, 1.5, 2.0, 4.0])
    xmin = 0.1
    colors = ['blue', 'green', 'yellow', 'red']
    
    x_list = np.linspace(1e-10, 1+np.max(mach_array)*1.2, 100)
    x_low = [np.sqrt(mach**2-1) for mach in mach_array]  
    x_high = [1+mach for mach in mach_array]
    max_I_DF = [Idf_Ostriker99(mach, xmin) for mach in mach_array]
    
    I_for_all_mach = []
    for mach in mach_array:
        print(f"calculating for mach = {mach} ...")
        I_for_all_mach.append(np.array([fdf_supersonic_from_xinn_to_xout(xout=xval, xinn=xmin, mach=mach, xmin=xmin) for xval in x_list]))
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for i, mach in enumerate(mach_array):
        ax.plot(x_list, I_for_all_mach[i], color=colors[i], label=f"$\mathcal{{M}} = {mach}$")
       
        ax.vlines(x=x_high[i], ymin=-0.01, ymax=max_I_DF[i], color=colors[i], linestyle='--')
        
    ax.set_xlabel(r"x = r/($c_s$ t)",fontsize=14)
    ax.set_ylabel(r"$I_{DF}$ (x)",fontsize=14)
    ax.set_title(r"Supersonic DF force",fontsize=14)
    ax.legend()
    plt.tight_layout()
    filename = "supersonic_FDF_allmach.pdf"
    plt.savefig(filename)
    
    #then plot the volume
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for i, mach in enumerate(mach_array):
        ax.plot(x_list, [volume_integral_supersonic(x, mach, xmin) for x in x_list], color=colors[i], label=f"$\mathcal{{M}} = {mach}$")
        ax.vlines(x=x_low[i], ymin=-1, ymax=2*np.pi/3*(1 - np.sqrt(mach**2 - 1)/mach)*np.sqrt(mach**2 - 1)**3, color=colors[i], linestyle='--')
        ax.vlines(x=x_high[i], ymin=-1, ymax=np.pi/3*(mach + 2.0 - 1.0/mach), color=colors[i], linestyle='--')
        
    ax.set_xlabel(r"x = r/($c_s$ t)",fontsize=14)
    ax.set_ylabel(r"Volume [($C_s$ t)$^3$]",fontsize=14)
    ax.set_title(r"Supersonic Wake Volume",fontsize=14)
    ax.legend()
    plt.tight_layout()
    filename = "supersonic_volume_allmach.pdf"
    plt.savefig(filename)
  
def plot_I_Mach(output_dir):
    mach_array = np.logspace(-2, np.log10(5.0), 100)
    # Cst_rmin_ratios = np.array([10, 20, 50, 100])
    Vt_rmin_ratios = np.array([20, 40, 100, 200])
    I_DF_list = []
    for Vt_rmin in Vt_rmin_ratios:
        I_DF = []
        for mach in mach_array:
            Cst_rmin = Vt_rmin / mach
            xmin = 1/Cst_rmin
            I_DF.append(Idf_Ostriker99_nosingularity(mach, xmin))
        I_DF_list.append(I_DF)
    I_DF_list = np.array(I_DF_list)

    #also compare with Kim07 model
    I_phi_Kim07_list = []
    Rp_rmin_ratios = np.array([10, 20, 50, 100])
    for Rp_rmin in Rp_rmin_ratios:
        I_phi_Kim07 = []
        for mach in mach_array:
            I_phi = Idf_phi_Kim2007(mach, Rp_rmin)
            I_phi_Kim07.append(I_phi)
        I_phi_Kim07_list.append(I_phi_Kim07)
    I_phi_Kim07_list = np.array(I_phi_Kim07_list)

    I_R_Kim07 = np.array([Idf_R_Kim2007(mach) for mach in mach_array])
        

    Ostriker99_colors = ['orange', 'darkorange', 'red', 'brown']
    Kim07_colors = ['cyan', 'deepskyblue', 'dodgerblue','royalblue']
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4),facecolor='w')
    for i, ratio in enumerate(Vt_rmin_ratios):
        ax.plot(mach_array, I_DF_list[i], color=Ostriker99_colors[i], label=r"Ostriker99, V t/r$_{\min}$" + f" = {ratio}")
    for i, ratio in enumerate(Rp_rmin_ratios):
        ax.plot(mach_array, I_phi_Kim07_list[i], color=Kim07_colors[i], linestyle='--', label=r"Kim07 I$_{\varphi}$, Rp/r$_{\min}$" + f" = {ratio}")
    ax.plot(mach_array, I_R_Kim07, color='k', linestyle=':', label=r"Kim07 I$_{\mathrm{R}}$")
    ax.set_xlabel(r"$\mathcal{M}$",fontsize=14)
    ax.set_ylabel(r"$I_{DF}$",fontsize=14)
    ax.tick_params(axis='both',direction='in')
    ax.legend()
    plt.tight_layout()
    filename = os.path.join(output_dir, "I_DF_Mach_new.png")
    plt.savefig(filename, dpi= 300)
    

    #F = I/Mach^2
    Dimensionless_I_DF = I_DF_list.copy()
    for i in range(len(Vt_rmin_ratios)):
        Dimensionless_I_DF[i] = Dimensionless_I_DF[i]/mach_array**2
    
    Dimensionless_I_phi_Kim07 = I_phi_Kim07_list.copy()
    for i in range(len(Rp_rmin_ratios)):
        Dimensionless_I_phi_Kim07[i] = Dimensionless_I_phi_Kim07[i]/mach_array**2
    Dimensionless_I_R_Kim07 = I_R_Kim07 / mach_array**2
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4),facecolor='w')
    for i, ratio in enumerate(Vt_rmin_ratios):
        ax.plot(mach_array, Dimensionless_I_DF[i], color=Ostriker99_colors[i], label=r"Ostriker99, V t/r$_{\min}$" + f" = {ratio}")
    for i, ratio in enumerate(Rp_rmin_ratios):
        ax.plot(mach_array, Dimensionless_I_phi_Kim07[i], color=Kim07_colors[i], linestyle='--', label=r"Kim07 I$_{\varphi}$, Rp/r$_{\min}$" + f" = {ratio}")
    ax.plot(mach_array, Dimensionless_I_R_Kim07, color='k', linestyle=':', label=r"Kim07 I$_{\mathrm{R}}$")
    
    ax.set_xlabel(r"$\mathcal{M}$",fontsize=14)
    ax.set_ylabel(r"$F / [4 \pi \rho_0 (G m_p)^2/C_s^2]$",fontsize=14)
    ax.tick_params(axis='both',direction='in')
    ax.legend()
    plt.tight_layout()
    filename = os.path.join(output_dir, "F_DF_Mach_new.png")
    plt.savefig(filename, dpi=300)


    #I_over_Mach (the factor for DF heating)
    fig, ax = plt.subplots(1, 1, figsize=(5, 4),facecolor='w')
    for i, ratio in enumerate(Vt_rmin_ratios):
        ax.plot(mach_array, I_DF_list[i]/mach_array, color=Ostriker99_colors[i], label=r"Ostriker99, V t/r$_{\min}$" + f" = {ratio}")
    for i, ratio in enumerate(Rp_rmin_ratios):
        ax.plot(mach_array, I_phi_Kim07_list[i]/mach_array, color=Kim07_colors[i], linestyle='--', label=r"Kim07 I$_{\varphi}$, Rp/r$_{\min}$" + f" = {ratio}")
    ax.plot(mach_array, I_R_Kim07/mach_array, color='k', linestyle=':', label=r"Kim07 I$_{\mathrm{R}}$")
    
    ax.set_xlabel(r"$\mathcal{M}$",fontsize=14)
    ax.set_ylabel(r"$I_{DF}/\mathcal{M}$",fontsize=14)
    ax.tick_params(axis='both',direction='in')
    ax.legend()
    plt.tight_layout()
    filename = os.path.join(output_dir, "I_over_Mach_new.png")
    plt.savefig(filename, dpi=300)

def get_nonlinear_correction(A, mach):
    #KK09 nonlinear correction factor
    if mach <= 1:
        nonlinear_corr = 1.0
    else:
        eta = A / (mach**2 - 1)
        if eta > 100:
            print(f"Nonlinear Corr Warning: eta = {eta} > 100, nonlinear correction may be inaccurate")
        if eta >= 2:
            nonlinear_corr = (eta/2)**(-0.45)
        else:
            nonlinear_corr = 1.0
    return nonlinear_corr

def compute_average_I(func, dist_params, dist_type, rmin_param=None, A=None, nonlinear_correction=False):
    """
    Compute the average of a function weighted by a velocity distribution (MB or TG),
    optionally applying nonlinear correction.

    Parameters
    ----------
    func : callable
        Function of (mach, rmin_param) or just (mach)
    dist_params : float or tuple
        Parameters for MB (sigma) or TG (mu, sigma)
    dist_type : str
        "maxwell-boltzmann" or "truncated-gaussian"
    rmin_param : float or None
        Physical scale param to pass into I_df functions (e.g. Vt/rmin or Rp/rmin)
    A : float or None
        Mass ratio param for nonlinear correction
    nonlinear_correction : bool
        Whether to apply nonlinear correction using get_nonlinear_correction

    Returns
    -------
    float
        Average I_df (or I_df / mach) over the velocity distribution
    """

    mach_min, mach_max = 0.01, 10.0

    if dist_type == "maxwell-boltzmann":
        sigma = dist_params
        def pdf(mach): return maxwell_boltzmann_pdf(mach, sigma)
    elif dist_type == "truncated-gaussian":
        mu, sigma = dist_params
        def pdf(mach): return truncated_gaussian_pdf(mach, mu, sigma)
    else:
        raise ValueError(f"Unknown dist_type: {dist_type}")

    def integrand(mach):
        try:
            I_val = func(mach) if rmin_param is None else func(mach, rmin_param)
            if nonlinear_correction and A is not None:
                I_val *= get_nonlinear_correction(A, mach)
        except Exception:
            I_val = 0.0
        return I_val * pdf(mach)

    numerator, _ = quad(integrand, mach_min, mach_max, limit=100)
    normalization, _ = quad(pdf, mach_min, mach_max, limit=100)

    return numerator / normalization



def plot_averages_MB(output_dir):
    """
    Calculate and plot the average values of I/mach as a function of sigma_Mach,
    including nonlinear correction for Ostriker99 (with different A values).
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- Setup ---
    sigma_mach_array = np.logspace(-1, np.log10(3), 50)  # From 0.1 to 3
    A_array = [0.0, 1.0, 2.0]
    linestyles = ['-', '--', '-.', ':']

    # Ostriker99 setup
    Vt_rmin_ratios = np.array([20, 40, 100, 200])
    Ostriker99_colors = ['orange', 'darkorange', 'red', 'brown']
    avg_I_over_mach_values = np.zeros((len(Vt_rmin_ratios), len(sigma_mach_array), len(A_array)))

    for i, Vt_rmin in enumerate(Vt_rmin_ratios):
        for k, A_number in enumerate(A_array):
            for j, sigma_mach in enumerate(sigma_mach_array):
                def I_over_mach(mach, Vt_rmin):
                    return Idf_Ostriker99_nosingularity_Vtrmin(mach, Vt_rmin) / mach

                avg_I_over_mach_values[i, j, k] = compute_average_I(
                    I_over_mach,
                    sigma_mach,
                    dist_type="maxwell-boltzmann",
                    rmin_param=Vt_rmin,
                    A=A_number,
                    nonlinear_correction=True
                )

    # Kim07 setup
    Rp_rmin_ratios = np.array([10, 20, 50, 100])
    Kim07_colors = ['cyan', 'deepskyblue', 'dodgerblue', 'royalblue']
    avg_I_phi_over_mach_values = np.zeros((len(Rp_rmin_ratios), len(sigma_mach_array)))
    for i, Rp_rmin in enumerate(Rp_rmin_ratios):
        for j, sigma_mach in enumerate(sigma_mach_array):
            def I_phi_over_mach(mach, Rp_rmin):
                return Idf_phi_Kim2007(mach, Rp_rmin) / mach
            avg_I_phi_over_mach_values[i, j] = compute_average_I(
                I_phi_over_mach,
                sigma_mach,
                dist_type="maxwell-boltzmann",
                rmin_param=Rp_rmin,
                nonlinear_correction=False
            )

    # Kim07 IR
    avg_I_R_over_mach_values = np.zeros(len(sigma_mach_array))
    for j, sigma_mach in enumerate(sigma_mach_array):
        def I_R_over_mach(mach):
            return Idf_R_Kim2007(mach) / mach
        avg_I_R_over_mach_values[j] = compute_average_I(
            I_R_over_mach,
            sigma_mach,
            dist_type="maxwell-boltzmann",
            nonlinear_correction=False
        )

    # --- Plot I/mach ---
    fig = plt.figure(figsize=(7.5, 5.5), facecolor='w')
    ax = plt.gca()

    # Ostriker99 with nonlinear correction (only A=0 labeled)
    # for i, ratio in enumerate(Vt_rmin_ratios):
    #     for k, A in enumerate(A_array):
    #         label = (
    #             rf"Ostriker99, V t/r$_{{\min}}$ = {ratio}" if A == 0 else None
    #         )
    #         ax.plot(
    #             sigma_mach_array,
    #             avg_I_over_mach_values[i, :, k],
    #             color=Ostriker99_colors[i],
    #             linestyle=linestyles[k],
    #             label=label
    #         )

    # Ostriker99: plot shaded region for A = 0 → A = 10
    for i, ratio in enumerate(Vt_rmin_ratios):
        upper_vals = avg_I_over_mach_values[i, :, 0]   # A = 0
        lower_vals = avg_I_over_mach_values[i, :, -1]  # A = 10

        ax.fill_between(
            sigma_mach_array,
            lower_vals,
            upper_vals,
            color=Ostriker99_colors[i],
            alpha=0.25,
            label=rf"Ostriker99+Kim09, V t/r$_{{\min}}$ = {ratio}"
        )


    # Kim07 I_phi
    for i, ratio in enumerate(Rp_rmin_ratios):
        ax.plot(
            sigma_mach_array,
            avg_I_phi_over_mach_values[i],
            color=Kim07_colors[i],
            linestyle='--',
            label=rf"Kim07 I$_\varphi$, Rp/r$_{{\min}}$ = {ratio}"
        )

    # Kim07 I_R
    ax.plot(
        sigma_mach_array,
        avg_I_R_over_mach_values,
        color='k',
        linestyle=':',
        label=r"Kim07 I$_{\mathrm{R}}$"
    )

    # --- Formatting ---
    ax.set_xlabel(r"$\sigma_{\mathcal{M}}$", fontsize=14)
    ax.set_ylabel(r"$\langle I_{DF}/\mathcal{M} \rangle$", fontsize=14)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=10, frameon=True, loc='best')
    plt.tight_layout()

    filename = os.path.join(output_dir, "avg_I_over_mach_vs_sigma_MB_nonlinear.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.close()


def plot_averages_TG(output_dir):

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # μ to loop over → subplot dimension
    mu_array = [0.8, 1.0, 1.3]
    mu_labels = [r"$\mu = {:.1f}$".format(mu) for mu in mu_array]

    # σ range (x-axis)
    sigma_array = np.linspace(0.2, 0.5, 40)

    # Vt/rmin ratios (color)
    Vt_rmin_ratios = np.array([20, 40, 100, 200])
    Ostriker99_colors = ['orange', 'darkorange', 'red', 'brown']

    # A values (linestyle)
    A_array = [0.0, 1.0, 2.0]
    linestyles = ['solid', 'dashed', 'dotted']

    # Init result array: shape = [n_mu, n_A, n_rmin, n_sigma]
    avg_I_over_mach_values = np.zeros((
        len(mu_array), len(A_array), len(Vt_rmin_ratios), len(sigma_array)
    ))

    for im, mu in enumerate(mu_array):
        for ia, A in enumerate(A_array):
            for ir, Vt_rmin in enumerate(Vt_rmin_ratios):
                for isigma, sigma in enumerate(sigma_array):

                    def I_over_mach(mach, rmin_param):
                        return Idf_Ostriker99_nosingularity_Vtrmin(mach, rmin_param) / mach

                    avg_I_over_mach_values[im, ia, ir, isigma] = compute_average_I(
                        I_over_mach,
                        dist_params=(mu, sigma),
                        dist_type="truncated-gaussian",
                        rmin_param=Vt_rmin,
                        A=A,
                        nonlinear_correction=True
                    )

    # --------- Plotting ---------
    fig, axes = plt.subplots(1, len(mu_array), figsize=(15, 5), sharey=True, facecolor='w')

    for im, mu in enumerate(mu_array):
        ax = axes[im]
        for ir, Vt_rmin in enumerate(Vt_rmin_ratios):
            # if ia == 0:  # Only label A=0 (linear case)
            #     label = fr"$Vt/r_{{\min}}$ = {Vt_rmin}"

            low_vals = avg_I_over_mach_values[im, -1, ir]  # A = max
            up_vals = avg_I_over_mach_values[im, 0, ir]  # A = 0

            ax.fill_between(
                sigma_array,
                low_vals,
                up_vals,
                color=Ostriker99_colors[ir],
                alpha=0.25,
                label=rf"Ostriker99+Kim09, V t/r$_{{\min}}$ = {Vt_rmin}"
            )

            # ax.plot(
            #     sigma_array,
            #     avg_I_over_mach_values[im, ia, ir],
            #     color=Ostriker99_colors[ir],
            #     linestyle=linestyles[ia],
            #     label=label
            # )


        

        ax.set_xlabel(r"$\sigma$ (Truncated Gaussian)", fontsize=12)
        ax.set_title(mu_labels[im], fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', direction='in')

    axes[0].set_ylabel(r"$\langle I_{\rm DF} / \mathcal{M} \rangle$", fontsize=13)
    axes[-1].legend(title="Linear case", loc='upper right')

    plt.tight_layout()
    outpath = os.path.join(output_dir, "avg_I_over_mach_vs_sigma_truncG_with_A.png")
    plt.savefig(outpath, dpi=300)
    print(f"Saved figure: {outpath}")
    plt.close()





if __name__ == '__main__':
    
    output_dir = '/home/zwu/21cm_project/unified_model/DF_results/'
    # fdfxplot()
    # fdfxinnplot()
    #fdfxplot_subsonic()
    #mach = 0.5
    #print(fdf_subsonic_from_xinn_to_xout(1 + mach, 1 - mach, mach))

    #plot_subsonic_schematic(0.8)
    #plot_subsonic_allmach()
    #plot_supersonic_schematic(mach=1.5, xmin=0.1)
    #plot_supersonic_allmach()
    
    # plot_I_Mach(output_dir)
    # plot_averages_MB(output_dir)
    plot_averages_TG(output_dir)