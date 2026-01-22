
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
    default_temp = 21
    default_vdd = 1.34
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
n = 8
k_xor = 4
k_xorff = 3
num_challenges = 100000
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
    # refactor: reusable sweep functions and interactive PUF selection
    def run_vdd_sweep(puf_factory, puf_name, sweep_ns, temp_fixed=25, num_reliability_crps=10000, noisiness=0.05):
        vdd_fine = np.round(np.arange(0.5, 3.001, 0.01), 3)
        plt.figure(figsize=(16,10))
        for n_bits in sweep_ns:
            print(f"  [{puf_name}] Challenge bits: {n_bits} (Vdd sweep)")
            reliability_challenges = np.random.choice([-1, 1], size=(num_reliability_crps, n_bits))
            reliabilities = []
            ref_puf = puf_factory(n=n_bits, temperature=temp_fixed, vdd=1.35, noisiness=noisiness)
            ref_responses = ref_puf.eval(reliability_challenges)
            for vdd in vdd_fine:
                puf = puf_factory(n=n_bits, temperature=temp_fixed, vdd=vdd, noisiness=noisiness)
                responses = puf.eval(reliability_challenges)
                reliabilities.append(np.mean(responses == ref_responses))
            plt.plot(vdd_fine, reliabilities, label=f'n={n_bits}')
        plt.title(f'{puf_name} Reliability vs Vdd at {temp_fixed}°C (noisiness={noisiness})', fontsize=20)
        plt.xlabel('Vdd (V)', fontsize=16)
        plt.ylabel('Reliability', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_temp_sweep(puf_factory, puf_name, sweep_ns, vdd_fixed=1.3, num_reliability_crps=10000, noisiness=0.05):
        temp_sweep = np.round(np.arange(0, 85.001, 0.1), 3)
        plt.figure(figsize=(16,10))
        for n_bits in sweep_ns:
            print(f"  [{puf_name}] Challenge bits: {n_bits} (Temperature sweep)")
            reliability_challenges = np.random.choice([-1, 1], size=(num_reliability_crps, n_bits))
            reliabilities = []
            ref_puf = puf_factory(n=n_bits, temperature=25, vdd=vdd_fixed, noisiness=noisiness)
            ref_responses = ref_puf.eval(reliability_challenges)
            for temp in temp_sweep:
                puf = puf_factory(n=n_bits, temperature=temp, vdd=vdd_fixed, noisiness=noisiness)
                responses = puf.eval(reliability_challenges)
                reliabilities.append(np.mean(responses == ref_responses))
            plt.plot(temp_sweep, reliabilities, label=f'n={n_bits}')
        plt.title(f'{puf_name} Reliability vs Temperature at Vdd={vdd_fixed}V (noisiness={noisiness})', fontsize=20)
        plt.xlabel('Temperature (°C)', fontsize=16)
        plt.ylabel('Reliability', fontsize=16)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # available PUF factories
    def bistable_factory(**kw):
        n = kw.pop('n', 8)
        temperature = kw.pop('temperature', 20)
        vdd = kw.pop('vdd', 1.35)
        T_factor = kw.pop('T_factor', None)
        V_factor = kw.pop('V_factor', None)
        weights = np.random.normal(0, 1, (4, n + 1))
        return XORBistableRingPUF(n=n, k=4, weights=weights, temperature=temperature, vdd=vdd, T_factor=T_factor, V_factor=V_factor)
    
    all_pufs = [
        ("ArbiterPUF", lambda **kw: ArbiterPUF(noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORArbiterPUF", lambda **kw: XORArbiterPUF(k=4, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("FeedForwardArbiterPUF", lambda **kw: FeedForwardArbiterPUF(ff=[(2,5),(4,7)], noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORFeedForwardArbiterPUF", lambda **kw: XORFeedForwardArbiterPUF(k=3, ff=[[(2,5)],[(4,7)],[(1,6)]], noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORBistableRingPUF", bistable_factory),
        ("LightweightSecurePUF", lambda **kw: LightweightSecurePUF(k=4, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("PermutationPUF", lambda **kw: PermutationPUF(k=4, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("InterposePUF", lambda **kw: InterposePUF(k_down=4, k_up=2, noisiness=kw.pop('noisiness', 0.05), **kw)),
    ]

    # select PUF
    print("Select PUF to evaluate (enter number):")
    for i, (name, _) in enumerate(all_pufs, start=1):
        print(f"  {i}: {name}")
    print(f"  {len(all_pufs)+1}: All of the above")
    sel = input("Choice: ").strip()
    try:
        sel_i = int(sel)
    except ValueError:
        print("Invalid input; defaulting to ArbiterPUF")
        sel_i = 1

    if sel_i == len(all_pufs) + 1:
        chosen = all_pufs
    else:
        chosen = [all_pufs[max(0, sel_i-1)]]

    # sweep settings
    sweep_ns = [32, 64, 128, 256]
    num_reliability_crps = 10000

    for puf_name, factory in chosen:
        # run both sweeps
        run_vdd_sweep(factory, puf_name, sweep_ns, temp_fixed=25, num_reliability_crps=num_reliability_crps, noisiness=0.05)
        run_temp_sweep(factory, puf_name, sweep_ns, vdd_fixed=1.3, num_reliability_crps=num_reliability_crps, noisiness=0.05)
