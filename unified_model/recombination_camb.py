import numpy as np
import matplotlib.pyplot as plt
import camb
#camb tutorial:  https://camb.readthedocs.io/en/latest/CAMBdemo.html
from physical_constants import H0, Ombh2, Omch2, ns, As, tau

# Set up CAMB parameters
pars = camb.CAMBparams()

# Set standard cosmological parameters (Planck 2018)
pars.set_cosmology(H0=H0, ombh2=Ombh2, omch2=Omch2, 
                    omk=0, tau=tau)

# Set up initial power spectrum parameters
pars.InitPower.set_params(As=As, ns=ns)

# We need to set some accuracy parameters for recombination
pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=2.0, lAccuracyBoost=2.0)

# Set redshift range - we want z from 1200 to 20
z_max = 1800
z_min = 10
nz = 2000  # number of redshift points

# Create redshift array (logarithmic spacing works well)
z_array = np.logspace(np.log10(z_min), np.log10(z_max), nz)

# Calculate results
results = camb.get_results(pars)

# Get the background evolution
background = results.get_background_redshift_evolution(z_array, 
                                                      vars=['x_e'])

# Extract free electron fraction
xe_array = background['x_e']
print("z:",z_array)
print("xe:",xe_array)

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(z_array, xe_array, 'b-', linewidth=2, label='Free electron fraction')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Redshift z', fontsize=12)
plt.ylabel('Free electron fraction $x_e$', fontsize=12)
plt.title('Evolution of Free Electron Fraction During Recombination', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Invert x-axis to show evolution from high to low redshift
plt.gca().invert_xaxis()

# Find approximate redshift of half ionization
z_half_idx = np.argmin(np.abs(xe_array - 0.5))
z_half = z_array[z_half_idx]


plt.tight_layout()
plt.savefig('recombination_plot.png', dpi=300)

# Print some key values
print(f"Redshift of 50% ionization: z ≈ {z_half:.1f}")
print(f"Redshift of 10% ionization: z ≈ {z_array[np.argmin(np.abs(xe_array - 0.1))]:.1f}")
print(f"Free electron fraction at z=10: {xe_array[0]:.6f}")

#save data to file
np.savetxt('xe_evolution.dat', np.column_stack([z_array, xe_array]), 
           header='Redshift  Free_electron_fraction', fmt='%.6e')
print("Data saved to 'xe_evolution.dat'")
