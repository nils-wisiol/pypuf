import sys
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from pypuf.simulation.bistable import XORBistableRingPUF
from pypuf.simulation.delay import XORArbiterPUF, FeedForwardArbiterPUF, XORFeedForwardArbiterPUF, ArbiterPUF, LightweightSecurePUF, PermutationPUF, InterposePUF

# Global parameters
n = 16
k_xor = 4
k_xorff = 3
num_uniqueness_crps = 100000
num_pufs = 20


def calculate_uniqueness(responses_matrix):
    """Calculate uniqueness (inter-device variation) from responses matrix."""
    k = responses_matrix.shape[0]
    n_bits = responses_matrix.shape[1]
    total_hd = 0
    num_pairs = 0
    for j in range(k-1):
        for i in range(j+1, k):
            hd = np.sum(responses_matrix[j] != responses_matrix[i]) / n_bits
            total_hd += hd
            num_pairs += 1
    if num_pairs > 0:
        return (2 / (k * (k - 1))) * total_hd * 100
    else:
        return 0


if __name__ == "__main__":
    print("Select uniqueness analysis mode:")
    print("1: Sweep Vdd from 0.5 to 3.0 at fixed temperatures (10, 50, 80)")
    print("2: Sweep temperature from 0 to 150°C at fixed Vdd (1.0, 1.8, 2.4)")
    mode = input("Enter 1 or 2: ").strip()

    os.makedirs('graphs', exist_ok=True)

    # Define all PUFs for this analysis
    all_pufs = [
        ("ArbiterPUF", lambda **kw: ArbiterPUF(n=n, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORArbiterPUF", lambda **kw: XORArbiterPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORBistableRingPUF", lambda **kw: (np.random.seed(kw.pop('seed', None)), XORBistableRingPUF(n=n, k=k_xor, weights=np.random.normal(0,1,(k_xor,n+1)), temperature=kw.pop('temperature', 25), vdd=kw.pop('vdd', 1.35)))[1]),
        ("LightweightSecurePUF", lambda **kw: LightweightSecurePUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("PermutationPUF", lambda **kw: PermutationPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("InterposePUF", lambda **kw: InterposePUF(n=n, k_down=k_xor, k_up=2, noisiness=kw.pop('noisiness', 0.05), **kw)),
    ]

    if mode == "1":
        # Mode 1: Sweep Vdd at fixed temperatures
        selected_temps = [10, 50, 80]
        vdd_fine = np.round(np.arange(0.5, 3.01, 0.01), 2)

        # Collect all data for all PUFs
        print("\nCollecting uniqueness data for all PUFs (Mode 1)...")
        for name, puf_factory in all_pufs:
            print(f"\nProcessing {name}...")
            all_data = []
            for temp in selected_temps:
                for vdd in vdd_fine:
                    responses_matrix = np.empty((num_pufs, num_uniqueness_crps), dtype=int)
    
    # 1. Generate ONE set of challenges to be used by ALL devices at this temp/vdd
                    shared_challenges = np.random.choice([-1, 1], size=(num_uniqueness_crps, n))
                    for idx in range(num_pufs):
                        puf = puf_factory(temperature=temp, vdd=vdd, seed=idx)
                        responses_matrix[idx] = puf.eval(shared_challenges)
                    uniqueness_val = calculate_uniqueness(responses_matrix)
                    all_data.append({
                        'Temperature': temp,
                        'Vdd': vdd,
                        'Uniqueness': uniqueness_val
                    })

            # Print table for this PUF
            print(f"\n{name} Uniqueness Table:")
            print("Temperature | Vdd | Uniqueness (%)")
            for d in all_data:
                print(f"{d['Temperature']:>11} | {d['Vdd']:<4} | {d['Uniqueness']:.3f}")

            # Plot by temperature
            uniqueness_curves = []
            for temp in selected_temps:
                temp_data = [d for d in all_data if d['Temperature'] == temp]
                vdds_plot = [d['Vdd'] for d in temp_data]
                uniquenesses = [d['Uniqueness'] for d in temp_data]
                uniqueness_curves.append((temp, vdds_plot, uniquenesses))

            plt.figure(figsize=(10, 6))
            for temp, vdds_plot, uniquenesses in uniqueness_curves:
                plt.plot(vdds_plot, uniquenesses, marker='o', label=f'T={temp}°C')
            plt.title(f'{name} Uniqueness vs Vdd (various T)')
            plt.xlabel('Vdd (V)')
            plt.ylabel('Uniqueness (%)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('graphs', f'{name}_uniqueness_vs_vdd_multiT.png'), dpi=150)
            plt.close()

            # Save CSV for this PUF
            csv_filename = os.path.join('graphs', f'{name}_uniqueness_mode1.csv')
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['Temperature', 'Vdd', 'Uniqueness']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_data:
                    writer.writerow(row)
            print(f"Data saved to {csv_filename}")

    elif mode == "2":
        # Mode 2: Sweep temperature at fixed Vdd
        sweep_vdds = [1.0, 1.8, 2.4]
        sweep_temps = [round(x, 1) for x in np.arange(0, 150, 5)]

        # Collect all data for all PUFs
        print("\nCollecting uniqueness data for all PUFs (Mode 2)...")
        for name, puf_factory in all_pufs:
            print(f"\nProcessing {name}...")
            all_data = []
            for vdd in sweep_vdds:
                for temp in sweep_temps:
                    responses_matrix = np.empty((num_pufs, num_uniqueness_crps), dtype=int)
                    
                    # 1. Generate ONE set of challenges to be used by ALL devices at this temp/vdd
                    shared_challenges = np.random.choice([-1, 1], size=(num_uniqueness_crps, n))
                    
                    for idx in range(num_pufs):
                        puf = puf_factory(temperature=temp, vdd=vdd, seed=idx)
                        # 2. Use the SHARED challenges for every device instance
                        responses_matrix[idx] = puf.eval(shared_challenges)
                    uniqueness_val = calculate_uniqueness(responses_matrix)
                    all_data.append({
                        'Temperature': temp,
                        'Vdd': vdd,
                        'Uniqueness': uniqueness_val
                    })

            # Print table for this PUF
            print(f"\n{name} Uniqueness Table:")
            print("Vdd | Temperature | Uniqueness (%)")
            for d in all_data:
                print(f"{d['Vdd']:<4} | {d['Temperature']:>11} | {d['Uniqueness']:.3f}")

            # Plot by Vdd
            uniqueness_curves = []
            for vdd in sweep_vdds:
                vdd_data = [d for d in all_data if d['Vdd'] == vdd]
                temps_plot = [d['Temperature'] for d in vdd_data]
                uniquenesses = [d['Uniqueness'] for d in vdd_data]
                uniqueness_curves.append((vdd, temps_plot, uniquenesses))

            plt.figure(figsize=(10, 6))
            for vdd, temps_plot, uniquenesses in uniqueness_curves:
                plt.plot(temps_plot, uniquenesses, marker='o', label=f'Vdd={vdd}V')
            plt.title(f'{name} Uniqueness vs Temperature (various Vdd)')
            plt.xlabel('Temperature (°C)')
            plt.ylabel('Uniqueness (%)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('graphs', f'{name}_uniqueness_vs_temp_multiVdd.png'), dpi=150)
            plt.close()

            # Save CSV for this PUF
            csv_filename = os.path.join('graphs', f'{name}_uniqueness_mode2.csv')
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['Temperature', 'Vdd', 'Uniqueness']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_data:
                    writer.writerow(row)
            print(f"Data saved to {csv_filename}")

    else:
        print("Invalid mode selected. Exiting.")
