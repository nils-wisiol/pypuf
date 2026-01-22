
# Helper to plot PUF performance
def plot_puf_performance(results, name):
    # Ensure 'graphs_reliability' directory exists
    os.makedirs('graphs', exist_ok=True)
    fig, ax = plt.subplots()
    for vdd in vdds:
        temps = [r['temperature'] for r in results if r['vdd'] == vdd]
        biases = [r['bias'] for r in results if r['vdd'] == vdd]
        ax.plot(temps, biases, marker='o', label=f'Vdd={vdd}')
    ax.set_title(f'{name} Bias vs Temperature')
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Bias (mean response)')
    ax.legend()
    plt.savefig(os.path.join('graphs', f'{name}_bias_vs_temp.png'))
    plt.close()

    # Reliability vs Temperature
    fig, ax = plt.subplots()
    for vdd in vdds:
        temps = [r['temperature'] for r in results if r['vdd'] == vdd]
        reliabilities = [r['reliability'] for r in results if r['vdd'] == vdd]
        ax.plot(temps, reliabilities, marker='o', label=f'Vdd={vdd}')
    ax.set_title(f'{name} Reliability vs Temperature')
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Reliability')
    ax.legend()
    plt.savefig(os.path.join('graphs_reliability', f'{name}_reliability_vs_temp.png'))
    plt.close()

    # Reliability vs Vdd
    fig, ax = plt.subplots()
    for temp in temperatures:
        vdd_vals = [r['vdd'] for r in results if r['temperature'] == temp]
        reliabilities = [r['reliability'] for r in results if r['temperature'] == temp]
        ax.plot(vdd_vals, reliabilities, marker='o', label=f'T={temp}')
    ax.set_title(f'{name} Reliability vs Vdd')
    ax.set_xlabel('Vdd (V)')
    ax.set_ylabel('Reliability')
    ax.legend()
    plt.savefig(os.path.join('graphs', f'{name}_reliability_vs_vdd.png'))
    plt.close()

    # Table
    print(f'\n{name} Performance Table:')
    print('Temperature | Vdd | Bias | Std | Reliability')
    for r in results:
        print(f"{r['temperature']:>11} | {r['vdd']:<3} | {r['bias']:.3f} | {r['std']:.3f} | {r['reliability']:.3f}")
# Helper to evaluate a PUF under all conditions
def evaluate_puf(puf_factory, name, **kwargs):
    results = []
    default_temp = 25
    default_vdd = 1.35
    # Reference responses at default conditions for reliability
    ref_puf = puf_factory(temperature=default_temp, vdd=default_vdd, **kwargs)
    ref_responses = ref_puf.eval(challenges)
    for temp in temperatures:
        for vdd in vdds:
            # Only allow one parameter to differ from default at a time
            if (temp != default_temp and vdd != default_vdd):
                continue  # skip combined case
            puf = puf_factory(temperature=temp, vdd=vdd, **kwargs)
            responses = puf.eval(challenges)
            bias = np.mean(responses)
            std = np.std(responses)
            # Reliability: use pypuf.metrics.reliability function
            reliability = reliability(responses, ref_responses)
            results.append({
                'temperature': temp,
                'vdd': vdd,
                'bias': bias,
                'std': std,
                'reliability': reliability
            })
    return results
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from pypuf.simulation.bistable import XORBistableRingPUF
from pypuf.simulation.delay import XORArbiterPUF, FeedForwardArbiterPUF, XORFeedForwardArbiterPUF, ArbiterPUF, LightweightSecurePUF, PermutationPUF, InterposePUF

# Global parameters (must be before any function definitions)
n = 16  # Increased to 16 for PermutationPUF compatibility (requires n >= 16)
k_xor = 4
k_xorff = 3
num_challenges = 10000
challenges = np.random.choice([-1, 1], size=(num_challenges, n))
temperatures = np.arange(0, 151, 5)  # 0, 5, 10, ..., 150 (valid range)
vdds = np.round(np.arange(0.5, 3.01, 0.01), 2)  # 0.5, 0.51, ..., 3.0

# --- New: Multi-curve reliability plots for selected constant temperatures and vdds ---

def plot_reliability_vs_vdd_for_temps_from_results(results, puf_name, temps, vdd_range):
    os.makedirs('graphs_reliability', exist_ok=True)
    fig, ax = plt.subplots()
    for temp in temps:
        vdds = []
        reliabilities = []
        for vdd in vdd_range:
            for r in results:
                if abs(r['temperature'] - temp) < 1e-6 and abs(r['vdd'] - vdd) < 1e-6:
                    vdds.append(vdd)
                    reliabilities.append(r['reliability'])
                    break
        if vdds:
            ax.plot(vdds, reliabilities, marker='o', label=f'T={temp}')
    ax.set_title(f'{puf_name} Reliability vs Vdd (various T)')
    ax.set_xlabel('Vdd (V)')
    ax.set_ylabel('Reliability')
    ax.legend()
    plt.savefig(os.path.join('graphs_reliability', f'{puf_name}_reliability_vs_vdd_multiT.png'))
    plt.close()


def plot_reliability_vs_temp_for_vdds_from_results(results, puf_name, vdds, temp_range):
    os.makedirs('graphs_reliability', exist_ok=True)
    fig, ax = plt.subplots()
    for vdd in vdds:
        temps = []
        reliabilities = []
        for temp in temp_range:
            for r in results:
                if abs(r['vdd'] - vdd) < 1e-6 and abs(r['temperature'] - temp) < 1e-6:
                    temps.append(temp)
                    reliabilities.append(r['reliability'])
                    break
        if temps:
            ax.plot(temps, reliabilities, marker='o', label=f'Vdd={vdd}')
    ax.set_title(f'{puf_name} Reliability vs Temperature (various Vdd)')
    ax.set_xlabel('Temperature (C)')
    ax.set_ylabel('Reliability')
    ax.legend()
    plt.savefig(os.path.join('graphs_reliability', f'{puf_name}_reliability_vs_temp_multiVdd.png'))
    plt.close()
# --- Example usage: plot for selected T and Vdd values ---

if __name__ == "__main__":
    # Allow the user to choose which PUF(s) to analyze and which analysis to run
    # Helper factory for XORBistableRingPUF: extract `seed` from kw before calling
    def xor_bistable_factory(**kw):
        seed = kw.pop('seed', 0)
        weights = np.random.default_rng(seed).normal(size=(k_xor, n + 1))
        return XORBistableRingPUF(n=n, k=k_xor, weights=weights, **kw)

    available_pufs = [
        ("ArbiterPUF", lambda **kw: ArbiterPUF(n=n, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORArbiterPUF", lambda **kw: XORArbiterPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORBistableRingPUF", xor_bistable_factory),
        ("FeedForwardArbiterPUF", lambda **kw: FeedForwardArbiterPUF(n=n, ff=[(3, 7)], noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORFeedForwardArbiterPUF", lambda **kw: XORFeedForwardArbiterPUF(n=n, k=k_xorff, ff=[[(3,7)]], noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("LightweightSecurePUF", lambda **kw: LightweightSecurePUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("PermutationPUF", lambda **kw: PermutationPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("InterposePUF", lambda **kw: InterposePUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
    ]

    print("Available PUFs:")
    for i, (name, _) in enumerate(available_pufs, start=1):
        print(f"{i}: {name}")
    print("A: All PUFs")
    sel = input("Select PUF by number (comma-separated) or 'A' for all: ").strip()
    if sel.lower() == 'a':
        selected = [name for name, _ in available_pufs]
    else:
        indices = [s.strip() for s in sel.split(',') if s.strip()]
        selected = []
        for s in indices:
            try:
                idx = int(s) - 1
                if 0 <= idx < len(available_pufs):
                    selected.append(available_pufs[idx][0])
            except Exception:
                pass

    print("Select analysis to run:")
    print("1: Uniformity vs Vdd/Temperature sweeps (mode 1)")
    print("2: Uniformity sweep over temperature (mode 2)")
    print("3: Bias & Reliability analysis (evaluate_puf)")
    mode = input("Enter 1, 2 or 3: ").strip()

    # Helper: filter factories according to user's selection
    def factories_for_selection(names):
        return [(name, fac) for (name, fac) in available_pufs if name in names]

    if mode == "1":
        selected_temps = [10, 50, 80]
        vdd_fine = np.round(np.arange(0.5, 3.01, 0.01), 2)
        all_pufs = factories_for_selection(selected)
        import csv
        num_uniqueness_crps = 100000
        num_pufs = 20
        uniqueness_challenges = np.random.choice([-1, 1], size=(num_uniqueness_crps, n))
        for name, puf_factory in all_pufs:
            print(f"\n{name} Bias Table:")
            print("Temperature | Vdd | Uniformity (%)")
            uniformity_curves = []
            all_data = []
            for temp in selected_temps:
                uniformities = []
                vdds_plot = []
                for vdd in vdd_fine:
                    responses_matrix = np.empty((num_pufs, num_uniqueness_crps), dtype=int)
                    for idx in range(num_pufs):
                        puf = puf_factory(temperature=temp, vdd=vdd, seed=idx)
                        responses_matrix[idx] = puf.eval(uniqueness_challenges)
                    # Uniformity: percentage of 1s averaged over all instances
                    uniformity_vals = np.mean((responses_matrix == 1).sum(axis=1) / num_uniqueness_crps) * 100
                    print(f"{temp:>11} | {vdd:<4} | {uniformity_vals:.3f}")
                    vdds_plot.append(vdd)
                    uniformities.append(uniformity_vals)
                    all_data.append({
                        'PUF': name,
                        'Temperature': temp,
                        'Vdd': vdd,
                        'Uniformity': uniformity_vals
                    })
                uniformity_curves.append((temp, vdds_plot, uniformities))
            plt.figure()
            for temp, vdds_plot, uniformities in uniformity_curves:
                plt.plot(vdds_plot, uniformities, label=f'T={temp}')
            plt.title(f'{name} Uniformity vs Vdd (various T)')
            plt.xlabel('Vdd (V)')
            plt.ylabel('Uniformity (%)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('graphs', f'{name}_uniformity_vs_vdd_multiT.png'))
            plt.close()
            csv_filename = os.path.join('graphs', f'{name}_uniformity_data.csv')
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['PUF', 'Temperature', 'Vdd', 'Uniformity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_data:
                    writer.writerow(row)
    elif mode == "2":
        sweep_vdds = [1.0, 1.8, 2.4]
        sweep_temps = [round(x, 1) for x in np.arange(0, 151, 5)]
        all_pufs = factories_for_selection(selected)
        import csv
        num_uniqueness_crps = 100000
        num_pufs = 20
        uniqueness_challenges = np.random.choice([-1, 1], size=(num_uniqueness_crps, n))
        for name, puf_factory in all_pufs:
            print(f"\n{name} Bias Table:")
            print("Vdd | Temperature | Uniformity (%)")
            uniformity_curves = []
            all_data = []
            for vdd in sweep_vdds:
                uniformities = []
                temps_plot = []
                for temp in sweep_temps:
                    responses_matrix = np.empty((num_pufs, num_uniqueness_crps), dtype=int)
                    for idx in range(num_pufs):
                        puf = puf_factory(temperature=temp, vdd=vdd, seed=idx)
                        responses_matrix[idx] = puf.eval(uniqueness_challenges)
                    uniformity_vals = np.mean((responses_matrix == 1).sum(axis=1) / num_uniqueness_crps) * 100
                    print(f"{vdd:<4} | {temp:>11} | {uniformity_vals:.3f}")
                    temps_plot.append(temp)
                    uniformities.append(uniformity_vals)
                    all_data.append({
                        'PUF': name,
                        'Temperature': temp,
                        'Vdd': vdd,
                        'Uniformity': uniformity_vals
                    })
                uniformity_curves.append((vdd, temps_plot, uniformities))
            plt.figure()
            for vdd, temps_plot, uniformities in uniformity_curves:
                plt.plot(temps_plot, uniformities, label=f'Vdd={vdd}')
            plt.title(f'{name} Uniformity vs Temperature (various Vdd)')
            plt.xlabel('Temperature (C)')
            plt.ylabel('Uniformity (%)')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('graphs', f'{name}_uniformity_vs_temp_multiVdd_sweep.png'))
            plt.close()
            csv_filename = os.path.join('graphs', f'{name}_uniformity_data_sweep.csv')
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['PUF', 'Temperature', 'Vdd', 'Uniformity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_data:
                    writer.writerow(row)
    elif mode == "3":
        # Run bias & reliability evaluation for selected PUFs
        all_pufs = factories_for_selection(selected)
        for name, puf_factory in all_pufs:
            print(f"Running bias/reliability evaluation for {name} ...")
            results = evaluate_puf(puf_factory, name)
            plot_puf_performance(results, name)
        print("Done.")
    else:
        print("Invalid mode selected. Exiting.")
