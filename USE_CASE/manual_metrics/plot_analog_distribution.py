"""
This script visualizes the distribution of the internal analog values of an
Arbiter PUF simulation. This value represents the final delay difference at the
end of an arbiter chain, just before the sign() function determines the output bit.

This provides a more direct view of how environmental factors and noise, as
modeled in the pypuf library, affect the PUF's behavior and reliability.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Correctly set up the path to find the pypuf package.
# The script is in the project's root directory, and the 'pypuf' package
# is located in the 'pypuf' subdirectory, which contains the actual package code.
pypuf_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pypuf')
sys.path.insert(0, pypuf_root)
from pypuf.simulation.delay import ArbiterPUF

# --- Simulation Parameters ---
n = 64
#seed = 123
noisiness = 0.1  # A non-zero noisiness to include measurement noise effect
num_challenges = 50000
challenges = np.random.choice([-1, 1], size=(num_challenges, n))

# Use a professional plot style
plt.style.use('seaborn-v0_8-whitegrid')

# --- PLOT 1: Analog Distribution vs. Temperature ---
fig1, ax1 = plt.subplots(figsize=(12, 7))
fixed_vdd_for_temp_plot = 1.34  # Nominal voltage
temperatures_to_plot = [10.0, 20.1, 50.0, 80.0]

for temp in temperatures_to_plot:
    # Create a PUF instance for each environmental condition
    puf = ArbiterPUF(n=n, noisiness=noisiness, temperature=temp, vdd=fixed_vdd_for_temp_plot)
    # Get the internal analog values (delay differences)
    analog_values = puf.val(challenges)
    sns.kdeplot(analog_values, ax=ax1, label=f'{temp}°C', fill=True, alpha=0.1)

ax1.set_title(f'KDE of Arbiter PUF Analog Output vs. Temperature (at Vdd = {fixed_vdd_for_temp_plot:.2f}V)', fontsize=16)
ax1.set_xlabel('Internal Analog Value (Total Delay Difference)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.axvline(0, color='k', linestyle='--', label='Decision Threshold (0.0)')
ax1.legend(title='Temperature')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

output_filename1 = 'analog_distribution_vs_temp.png'
plt.savefig(output_filename1)
print(f"Plot 1 saved to {output_filename1}")
plt.show()

# --- PLOT 2: Analog Distribution vs. Voltage (as per user request) ---
fig2, ax2 = plt.subplots(figsize=(12, 7))
fixed_temp_for_vdd_plot = 25.0
voltages_to_plot = [0.5, 0.8, 1.0, 1.8, 2.5, 5]

for vdd in voltages_to_plot:
    puf = ArbiterPUF(n=n, noisiness=noisiness, temperature=fixed_temp_for_vdd_plot, vdd=vdd)
    analog_values = puf.val(challenges)
    sns.kdeplot(analog_values, ax=ax2, label=f'{vdd:.1f} V', fill=True, alpha=0.1)

ax2.set_title(f'KDE of Arbiter PUF Analog Output vs. Voltage (at T = {fixed_temp_for_vdd_plot:.1f}°C)', fontsize=16)
ax2.set_xlabel('Internal Analog Value (Total Delay Difference)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.axvline(0, color='k', linestyle='--', label='Decision Threshold (0.0)')
ax2.legend(title='Voltage')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

output_filename2 = 'analog_distribution_vs_voltage.png'
plt.savefig(output_filename2)
print(f"Plot 2 saved to {output_filename2}")
plt.show()