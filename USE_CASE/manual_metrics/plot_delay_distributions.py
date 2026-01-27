"""
This script creates Kernel Density Estimate (KDE) plots to visualize the
distribution of individual component delays in a PUF under various
environmental conditions (temperature and voltage).

It uses a more detailed physical model where a base distribution of delays
is generated and then scaled according to the alpha-power law model
implemented in the pypuf library's PhysicalFactors class.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure pypuf can be imported by adjusting the path if necessary
try:
    from pypuf.simulation.temp_voltage import PhysicalFactors
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from pypuf.simulation.temp_voltage import PhysicalFactors

def get_scaled_delays(base_delays, temperature, vdd):
    """
    Applies environmental scaling to a base set of delays.
    """
    # Use the combined physical model from the library
    pf = PhysicalFactors(temperature=temperature, vdd=vdd)
    scaling_factor = pf.process(Tfactor=True, Vfactor=True)
    return base_delays * scaling_factor

# --- Simulation Parameters ---

# 1. Define a base distribution for component delays at nominal conditions
np.random.seed(42)
num_components = 10000
nominal_delay_mean_ns = 10.0  # 10 ns
nominal_delay_std_ns = 0.8    # 0.8 ns standard deviation
base_delays = np.random.normal(
    loc=nominal_delay_mean_ns,
    scale=nominal_delay_std_ns,
    size=num_components
)

# Use a professional plot style
plt.style.use('seaborn-v0_8-whitegrid')

# --- PLOT 1: Delay Distribution vs. Temperature ---

fig1, ax1 = plt.subplots(figsize=(12, 7))
fixed_vdd_for_temp_plot = 1.0
temperatures_to_plot = [10.0, 20.0, 50.0, 80.0]

for temp in temperatures_to_plot:
    scaled_delays = get_scaled_delays(base_delays, temp, fixed_vdd_for_temp_plot)
    sns.kdeplot(scaled_delays, ax=ax1, label=f'{temp}°C', fill=True, alpha=0.1)

ax1.set_title(f'KDE of Component Delays vs. Temperature (at Vdd = {fixed_vdd_for_temp_plot:.1f}V)', fontsize=16)
ax1.set_xlabel('Component Propagation Delay (ns)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.legend(title='Temperature')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

output_filename1 = 'delay_distribution_vs_temp.png'
plt.savefig(output_filename1)
print(f"Plot 1 saved to {output_filename1}")
plt.show()


# --- PLOT 2: Delay Distribution vs. Voltage ---

fig2, ax2 = plt.subplots(figsize=(12, 7))
fixed_temp_for_vdd_plot = 25.0
voltages_to_plot = [1.0, 1.8, 2.5]

for vdd in voltages_to_plot:
    scaled_delays = get_scaled_delays(base_delays, fixed_temp_for_vdd_plot, vdd)
    sns.kdeplot(scaled_delays, ax=ax2, label=f'{vdd:.1f} V', fill=True, alpha=0.1)

ax2.set_title(f'KDE of Component Delays vs. Voltage (at T = {fixed_temp_for_vdd_plot:.1f}°C)', fontsize=16)
ax2.set_xlabel('Component Propagation Delay (ns)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.legend(title='Voltage')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

output_filename2 = 'delay_distribution_vs_voltage.png'
plt.savefig(output_filename2)
print(f"Plot 2 saved to {output_filename2}")
plt.show()