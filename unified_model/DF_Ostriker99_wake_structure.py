import numpy as np
from scipy.integrate import quad, dblquad
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator


from TNG_plots import maxwell_boltzmann_pdf

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
    
def test_func(mach, xmin):
    return 1.0

def compute_average_I(func, sigma_mach, param=None):
    """
    Compute the average of a function weighted by the Maxwell distribution
    func: function to average
    sigma_mach: standard deviation of the Mach number distribution
    param: parameter for the function (optional, can be None if the function doesn't need it)
    """

    mach_min = 0.01
    mach_max = 10.0
    #Maxwell distribution is already normalized
    def integrand_numerator(mach):
        if param is None:
            return func(mach) * maxwell_boltzmann_pdf(mach, sigma_mach)
        else:
            return func(mach, param) * maxwell_boltzmann_pdf(mach, sigma_mach)
    
    result, _ = quad(integrand_numerator, mach_min, mach_max, limit=100)
    normalization, _ = quad(lambda mach: maxwell_boltzmann_pdf(mach, sigma_mach), mach_min, mach_max)
    result = result / normalization  # Normalize the result
    
    # Return the average
    return result

def plot_averages(output_dir):
    """
    Calculate and plot the average values of I and I/mach as a function of sigma_mach
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sigma_mach_array = np.logspace(-1, np.log10(3), 50)  # From 0.1 to 3

    #average I and I/Mach in Ostriker99
    Vt_rmin_ratios = np.array([20, 40, 100, 200])
    Ostriker99_colors = ['orange', 'darkorange', 'red', 'brown']
    avg_I_values = np.zeros((len(Vt_rmin_ratios), len(sigma_mach_array)))
    avg_I_over_mach_values = np.zeros((len(Vt_rmin_ratios), len(sigma_mach_array)))
    
    for i, Vt_rmin in enumerate(Vt_rmin_ratios):
        for j, sigma_mach in enumerate(sigma_mach_array):
            avg_I_values[i, j] = compute_average_I(Idf_Ostriker99_nosingularity_Vtrmin, sigma_mach, Vt_rmin)
            # Calculate average I/mach
            def I_over_mach(mach, Vt_rmin):
                return Idf_Ostriker99_nosingularity_Vtrmin(mach, Vt_rmin) / mach
            avg_I_over_mach_values[i, j] = compute_average_I(I_over_mach, sigma_mach, Vt_rmin)
    
    #also compare with Kim07 model
    Rp_rmin_ratios = np.array([10, 20, 50, 100])
    Kim07_colors = ['cyan', 'deepskyblue', 'dodgerblue','royalblue']
    avg_I_phi_values = np.zeros((len(Rp_rmin_ratios), len(sigma_mach_array)))
    avg_I_phi_over_mach_values = np.zeros((len(Rp_rmin_ratios), len(sigma_mach_array)))
    for i, Rp_rmin in enumerate(Rp_rmin_ratios):
        for j, sigma_mach in enumerate(sigma_mach_array):
            avg_I_phi_values[i, j] = compute_average_I(Idf_phi_Kim2007, sigma_mach, Rp_rmin)
            # Calculate average I/mach
            def I_phi_over_mach(mach, Rp_rmin):
                return Idf_phi_Kim2007(mach, Rp_rmin) / mach
            avg_I_phi_over_mach_values[i, j] = compute_average_I(I_phi_over_mach, sigma_mach, Rp_rmin)

    #I_R
    avg_I_R_values = np.zeros(len(sigma_mach_array))
    avg_I_R_over_mach_values = np.zeros(len(sigma_mach_array))
    for j, sigma_mach in enumerate(sigma_mach_array):
        avg_I_R_values[j] = compute_average_I(Idf_R_Kim2007, sigma_mach, None)
        # Calculate average I/mach
        def I_R_over_mach(mach):
            return Idf_R_Kim2007(mach) / mach
        avg_I_R_over_mach_values[j] = compute_average_I(I_R_over_mach, sigma_mach, None)

    # Visualize the Maxwell-Boltzmann distribution for different sigma values
    fig = plt.figure(figsize=(10, 6), facecolor='w')
    mach_array = np.linspace(0, 10, 100)
    for sigma in [0.5, 1.0, 2.0, 3.0]:
        dist = [maxwell_boltzmann_pdf(m, sigma) for m in mach_array]
        plt.plot(mach_array, dist, label=f"σ = {sigma}")
    plt.xlabel("Mach Number", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.title("Maxwell-Boltzmann Distribution for Different σ Values", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "maxwell_distributions.png"), dpi=300)
    
    # Plot average I as a function of sigma_mach
    fig = plt.figure(facecolor='w')
    ax = plt.gca()
    for i, ratio in enumerate(Vt_rmin_ratios):
        plt.plot(sigma_mach_array, avg_I_values[i], color=Ostriker99_colors[i], 
                 label=r"Ostriker99, V t/r$_{\min}$ = "+f"{ratio}")
    for i, ratio in enumerate(Rp_rmin_ratios):
        plt.plot(sigma_mach_array, avg_I_phi_values[i], color=Kim07_colors[i], linestyle='--', 
                 label=r"Kim07 I$_{\varphi}$, Rp/r$_{\min}$ = "+f"{ratio}")
    plt.plot(sigma_mach_array, avg_I_R_values, color='k', linestyle=':', label=r"Kim07 I$_{\mathrm{R}}$")
    plt.xlabel(r"$\sigma_{\mathcal{M}}$", fontsize=14)
    plt.ylabel(r"$\langle I_{DF} \rangle$", fontsize=14)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_I_vs_sigma_new.png"), dpi=300)
    
    # Plot average I/mach as a function of sigma_mach
    fig = plt.figure()
    ax = plt.gca()
    for i, ratio in enumerate(Vt_rmin_ratios):
        plt.plot(sigma_mach_array, avg_I_over_mach_values[i], color=Ostriker99_colors[i], 
                 label=r"Ostriker99, V t/r$_{\min}$ = "+f"{ratio}")
    for i, ratio in enumerate(Rp_rmin_ratios):
        plt.plot(sigma_mach_array, avg_I_phi_over_mach_values[i], color=Kim07_colors[i], linestyle='--', 
                 label=r"Kim07 I$_{\varphi}$, Rp/r$_{\min}$ = "+f"{ratio}")
    plt.plot(sigma_mach_array, avg_I_R_over_mach_values, color='k', linestyle=':', label=r"Kim07 I$_{\mathrm{R}}$")
    plt.xlabel(r"$\sigma_{\mathcal{M}}$", fontsize=14)
    plt.ylabel(r"$\langle I_{DF}/\mathcal{M} \rangle$", fontsize=14)
    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', direction='in')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_I_over_mach_vs_sigma_new.png"), dpi=300)
    
    


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
    plot_averages(output_dir)