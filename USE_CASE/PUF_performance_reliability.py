
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
n = 16
k_xor = 4
k_xorff = 3
num_challenges = 10000
challenges = np.random.choice([-1, 1], size=(num_challenges, n))
temperatures = np.arange(0, 101, 5)  # 0, 5, 10, ..., 100 (more steps)
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
    print("Select reliability analysis mode:")
    print("1: Sweep Vdd from 0.5 to 3.0 at fixed temperatures (old mode)")
    print("2: Sweep temperature from 10.1 to 80.1 (step 5) at fixed Vdd 1, 1.8, 2.5 (new mode)")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        # Old mode: sweep Vdd at fixed temperatures
        selected_temps = [0, 5, 10, 15, 20, 25, 30, 35, 40]
        vdd_fine = np.round(np.arange(0.5, 3.01, 0.01), 2)
        all_pufs = [
            ("ArbiterPUF", lambda **kw: ArbiterPUF(n=n, noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("XORArbiterPUF", lambda **kw: XORArbiterPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("FeedForwardArbiterPUF", lambda **kw: FeedForwardArbiterPUF(n=n, ff=[(2,5),(4,7)], noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("XORFeedForwardArbiterPUF", lambda **kw: XORFeedForwardArbiterPUF(n=n, k=k_xorff, ff=[[(2,5)],[(4,7)],[(1,6)]], noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("XORBistableRingPUF", lambda **kw: XORBistableRingPUF(n=n, k=k_xor, weights=np.random.normal(0,1,(k_xor,n+1)), **kw)),
            ("LightweightSecurePUF", lambda **kw: LightweightSecurePUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("PermutationPUF", lambda **kw: PermutationPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("InterposePUF", lambda **kw: InterposePUF(n=n, k_down=k_xor, k_up=2, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ]
        import csv
        num_reliability_crps = 100000
        reliability_challenges = np.random.choice([-1, 1], size=(num_reliability_crps, n))
        for name, puf_factory in all_pufs:
            print(f"\n{name} Bias Table:")
            print("Temperature | Vdd | Bias")
            reliability_curves = []
            all_data = []
            for temp in selected_temps:
                reliabilities = []
                vdds_plot = []
                ref_puf = puf_factory(temperature=temp, vdd=1.35)
                ref_responses = ref_puf.eval(reliability_challenges)
                for vdd in vdd_fine:
                    puf = puf_factory(temperature=temp, vdd=vdd)
                    responses = puf.eval(reliability_challenges)
                    bias = np.mean(responses)
                    std = np.std(responses)
                    reliability_val = np.mean(responses == ref_responses)
                    print(f"{temp:>11} | {vdd:<4} | {bias:.3f}")
                    vdds_plot.append(vdd)
                    reliabilities.append(reliability_val)
                    all_data.append({
                        'PUF': name,
                        'Temperature': temp,
                        'Vdd': vdd,
                        'Bias': bias,
                        'Std': std,
                        'Reliability': reliability_val
                    })
                reliability_curves.append((temp, vdds_plot, reliabilities))
            plt.figure()
            for temp, vdds_plot, reliabilities in reliability_curves:
                plt.plot(vdds_plot, reliabilities, label=f'T={temp}')
            plt.title(f'{name} Reliability vs Vdd (various T)')
            plt.xlabel('Vdd (V)')
            plt.ylabel('Reliability')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('graphs', f'{name}_reliability_vs_vdd_multiT.png'))
            plt.close()
            csv_filename = os.path.join('graphs', f'{name}_performance_data.csv')
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['PUF', 'Temperature', 'Vdd', 'Bias', 'Std', 'Reliability']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_data:
                    writer.writerow(row)
    elif mode == "2":
        # New mode: sweep temperature at fixed Vdd
        sweep_vdds = [1.0, 1.35, 1.8, 2.5]
        sweep_temps = [round(x, 1) for x in np.arange(0, 100, 5)]
        all_pufs = [
            ("ArbiterPUF", lambda **kw: ArbiterPUF(n=n, noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("XORArbiterPUF", lambda **kw: XORArbiterPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("FeedForwardArbiterPUF", lambda **kw: FeedForwardArbiterPUF(n=n, ff=[(2,5),(4,7)], noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("XORFeedForwardArbiterPUF", lambda **kw: XORFeedForwardArbiterPUF(n=n, k=k_xorff, ff=[[(2,5)],[(4,7)],[(1,6)]], noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("XORBistableRingPUF", lambda **kw: XORBistableRingPUF(n=n, k=k_xor, weights=np.random.normal(0,1,(k_xor,n+1)), **kw)),
            ("LightweightSecurePUF", lambda **kw: LightweightSecurePUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("PermutationPUF", lambda **kw: PermutationPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
            ("InterposePUF", lambda **kw: InterposePUF(n=n, k_down=k_xor, k_up=2, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ]
        import csv
        num_reliability_crps = 100000
        reliability_challenges = np.random.choice([-1, 1], size=(num_reliability_crps, n))
        for name, puf_factory in all_pufs:
            print(f"\n{name} Bias Table:")
            print("Vdd | Temperature | Bias")
            reliability_curves = []
            all_data = []
            for vdd in sweep_vdds:
                reliabilities = []
                temps_plot = []
                ref_puf = puf_factory(temperature=25, vdd=vdd)
                ref_responses = ref_puf.eval(reliability_challenges)
                for temp in sweep_temps:
                    puf = puf_factory(temperature=temp, vdd=vdd)
                    responses = puf.eval(reliability_challenges)
                    bias = np.mean(responses)
                    std = np.std(responses)
                    reliability_val = np.mean(responses == ref_responses)
                    print(f"{vdd:<4} | {temp:>11} | {bias:.3f}")
                    temps_plot.append(temp)
                    reliabilities.append(reliability_val)
                    all_data.append({
                        'PUF': name,
                        'Temperature': temp,
                        'Vdd': vdd,
                        'Bias': bias,
                        'Std': std,
                        'Reliability': reliability_val
                    })
                reliability_curves.append((vdd, temps_plot, reliabilities))
            plt.figure()
            for vdd, temps_plot, reliabilities in reliability_curves:
                plt.plot(temps_plot, reliabilities, label=f'Vdd={vdd}')
            plt.title(f'{name} Reliability vs Temperature (various Vdd)')
            plt.xlabel('Temperature (C)')
            plt.ylabel('Reliability')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join('graphs', f'{name}_reliability_vs_temp_multiVdd_sweep.png'))
            plt.close()
            csv_filename = os.path.join('graphs', f'{name}_performance_data_sweep.csv')
            with open(csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['PUF', 'Temperature', 'Vdd', 'Bias', 'Std', 'Reliability']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in all_data:
                    writer.writerow(row)
    else:
        print("Invalid mode selected. Exiting.")

    # Only these temperatures:
    selected_temps = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    vdd_fine = np.round(np.arange(0.5, 3.01, 0.01), 2)
    all_pufs = [
        ("ArbiterPUF", lambda **kw: ArbiterPUF(n=n, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORArbiterPUF", lambda **kw: XORArbiterPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("FeedForwardArbiterPUF", lambda **kw: FeedForwardArbiterPUF(n=n, ff=[(2,5),(4,7)], noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORFeedForwardArbiterPUF", lambda **kw: XORFeedForwardArbiterPUF(n=n, k=k_xorff, ff=[[(2,5)],[(4,7)],[(1,6)]], noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORBistableRingPUF", lambda **kw: XORBistableRingPUF(n=n, k=k_xor, weights=np.random.normal(0,1,(k_xor,n+1)), **kw)),
        ("LightweightSecurePUF", lambda **kw: LightweightSecurePUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("PermutationPUF", lambda **kw: PermutationPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("InterposePUF", lambda **kw: InterposePUF(n=n, k_down=k_xor, k_up=2, noisiness=kw.pop('noisiness', 0.05), **kw)),
    ]

    import csv
    # Use more CRPs for reliability calculation
    num_reliability_crps = 100000
    reliability_challenges = np.random.choice([-1, 1], size=(num_reliability_crps, n))
    for name, puf_factory in all_pufs:
        print(f"\n{name} Bias Table:")
        print("Temperature | Vdd | Bias")
        reliability_curves = []
        all_data = []
        for temp in selected_temps:
            reliabilities = []
            vdds_plot = []
            # Reference responses at default conditions
            ref_puf = puf_factory(temperature=temp, vdd=1.35)
            ref_responses = ref_puf.eval(reliability_challenges)
            for vdd in vdd_fine:
                puf = puf_factory(temperature=temp, vdd=vdd)
                responses = puf.eval(reliability_challenges)
                bias = np.mean(responses)
                std = np.std(responses)
                reliability_val = np.mean(responses == ref_responses)
                print(f"{temp:>11} | {vdd:<4} | {bias:.3f}")
                vdds_plot.append(vdd)
                reliabilities.append(reliability_val)
                all_data.append({
                    'PUF': name,
                    'Temperature': temp,
                    'Vdd': vdd,
                    'Bias': bias,
                    'Std': std,
                    'Reliability': reliability_val
                })
            reliability_curves.append((temp, vdds_plot, reliabilities))
        # Plot all curves for this PUF
        plt.figure()
        for temp, vdds_plot, reliabilities in reliability_curves:
            plt.plot(vdds_plot, reliabilities, label=f'T={temp}')
        plt.title(f'{name} Reliability vs Vdd (various T)')
        plt.xlabel('Vdd (V)')
        plt.ylabel('Reliability')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join('graphs', f'{name}_reliability_vs_vdd_multiT.png'))
        plt.close()
        # Save all data to CSV
        csv_filename = os.path.join('graphs', f'{name}_performance_data.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['PUF', 'Temperature', 'Vdd', 'Bias', 'Std', 'Reliability']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_data:
                writer.writerow(row)
