
import numpy as np
import matplotlib.pyplot as plt
from pypuf.simulation import ArbiterPUF
from pypuf.io import ChallengeResponseSet
from pypuf.metrics import reliability_at_point
import os

# Ensure the script can find the pypuf module
# This adds the project root to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    # 1. Define PUF parameters
    n_bits = 64
    temp_fixed = 25.0
    vdd_nominal = 1.2
    num_challenges = 100000

    # 2. Generate challenges
    challenges = np.random.choice([-1, 1], size=(num_challenges, n_bits))

    # 3. Generate reference CRP set at nominal conditions
    print("Generating reference CRP set...")
    # By using a fixed seed, we ensure the "golden" PUF is always the same
    ref_puf = ArbiterPUF(n=n_bits, seed=1, temperature=temp_fixed, vdd=vdd_nominal, noisiness=0.05)
    ref_crps = ChallengeResponseSet(challenges=challenges, responses=ref_puf.eval(challenges))
    print("Reference set created.")

    # 4. Define voltage sweep range
    vdd_sweep = np.round(np.arange(0.8, 1.6, 0.05), 2)
    reliabilities = []

    # 5. Perform the sweep
    print("Performing voltage sweep...")
    for vdd in vdd_sweep:
        # We use the same seed to ensure we are evaluating the same PUF instance,
        # only affected by the environmental parameter change.
        puf_at_point = ArbiterPUF(n=n_bits, seed=1, temperature=temp_fixed, vdd=vdd, noisiness=0.05)
        
        # Use the new function to calculate reliability
        reliability = np.mean(reliability_at_point(puf_at_point, ref_crps))
        reliabilities.append(reliability)
        print(f"  Vdd={vdd:.2f} V, Reliability={reliability:.4f}")

    # 6. Plot the results
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.plot(vdd_sweep, reliabilities, marker='o', linestyle='-')
    plt.title(f'Reliability vs. Vdd (at T={temp_fixed}°C, n={n_bits})')
    plt.xlabel('Vdd (V)')
    plt.ylabel('Reliability')
    plt.grid(True)
    plt.ylim(0.98, 1.01)
    
    # Save the plot to a file to be displayed
    plot_filename = 'voltage_sweep_test.png'
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    plt.close()

if __name__ == "__main__":
    main()
