"""
Comprehensive metrics plotting for all PUF types across temperature and voltage sweeps.
Generates plots for: reliability, uniqueness, bias, similarity, influence, total_influence
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pypuf.simulation.bistable import XORBistableRingPUF
from pypuf.simulation.delay import (XORArbiterPUF, FeedForwardArbiterPUF, 
                                     XORFeedForwardArbiterPUF, ArbiterPUF, 
                                     LightweightSecurePUF, PermutationPUF, InterposePUF)
from pypuf.metrics import reliability, uniqueness, bias, similarity
from pypuf.metrics.fourier import total_influence

# Global parameters
n = 64
k_xor = 4
k_xorff = 3

def create_puf_factories():
    """Returns list of (name, factory_func) tuples for all PUF types"""
    return [
        ("ArbiterPUF", lambda **kw: ArbiterPUF(n=n, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORArbiterPUF", lambda **kw: XORArbiterPUF(n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORBistableRingPUF", lambda **kw: (np.random.seed(kw.pop('seed', None)), 
                                              XORBistableRingPUF(n=n, k=k_xor, 
                                                                weights=np.random.normal(0,1,(k_xor,n+1)), 
                                                                temperature=kw.pop('temperature', 25), 
                                                                vdd=kw.pop('vdd', 1.35)))[1]),
        ("FeedForwardArbiterPUF", lambda **kw: FeedForwardArbiterPUF(n=n, ff=[(2,5),(4,7)], 
                                                                     noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("XORFeedForwardArbiterPUF", lambda **kw: XORFeedForwardArbiterPUF(n=n, k=k_xorff, 
                                                                           ff=[[(2,5)],[(4,7)],[(1,6)]], 
                                                                           noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("LightweightSecurePUF", lambda **kw: LightweightSecurePUF(n=n, k=k_xor, 
                                                                   noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("PermutationPUF", lambda **kw: PermutationPUF(n=n, k=k_xor, 
                                                       noisiness=kw.pop('noisiness', 0.05), **kw)),
        ("InterposePUF", lambda **kw: InterposePUF(n=n, k_down=k_xor, k_up=2, 
                                                   noisiness=kw.pop('noisiness', 0.05), **kw)),
    ]

class MetricsCalculator:
    """Handles calculation of metrics for PUF instances"""
    
    @staticmethod
    def calc_reliability(puf, N_challenges=100000, r=5):
        """Calculate reliability using pypuf.metrics.reliability
        
        Measures how consistently the PUF returns same response for repeated queries
        """
        try:
            # Use pypuf's built-in reliability function
            # This queries the PUF r times per challenge and measures consistency
            rel = reliability(puf, seed=42, N=N_challenges, r=r)
            # Return mean reliability across all bits and challenges
            return np.mean(rel)
        except Exception as e:
            print(f"    Reliability calc error: {e}")
            return np.nan
    
    @staticmethod
    def calc_uniqueness(puf_factory, temperature, vdd, N=100000, num_instances=5):
        """Calculate uniqueness"""
        try:
            instances = [puf_factory(temperature=temperature, vdd=vdd, seed=i) 
                        for i in range(num_instances)]
            uniq = uniqueness(instances, seed=42, N=N)
            return np.mean(uniq)
        except Exception as e:
            return np.nan
    
    @staticmethod
    def calc_bias(puf, N=100000):
        """Calculate bias"""
        try:
            b = bias(puf, seed=42, N=N)
            return np.abs(np.mean(b))
        except Exception as e:
            return np.nan
    
    @staticmethod
    def calc_similarity(puf_factory, temperature, vdd, N=100000):
        """Calculate similarity between two instances"""
        try:
            puf1 = puf_factory(temperature=temperature, vdd=vdd, seed=1)
            puf2 = puf_factory(temperature=temperature, vdd=vdd, seed=2)
            sim = similarity(puf1, puf2, seed=42, N=N)
            return np.mean(sim)
        except Exception as e:
            return np.nan
    
    @staticmethod
    def calc_total_influence(puf, N=100000):
        """Calculate total influence"""
        try:
            total_inf = total_influence(puf, seed=42, N=N)
            return total_inf
        except Exception as e:
            return np.nan

def plot_metrics_vs_temperature(puf_name, puf_factory, vdds_to_plot, temp_range, output_dir):
    """
    Plot all metrics vs temperature for multiple constant voltages
    
    Args:
        puf_name: Name of PUF type
        puf_factory: Factory function to create PUF instances
        vdds_to_plot: List of voltage values to plot
        temp_range: Range of temperatures to sweep
        output_dir: Directory to save plots
    """
    print(f"\nCalculating metrics for {puf_name} vs Temperature...")
    
    metrics_data = {
        'reliability': {vdd: [] for vdd in vdds_to_plot},
        'uniqueness': {vdd: [] for vdd in vdds_to_plot},
        'bias': {vdd: [] for vdd in vdds_to_plot},
        'similarity': {vdd: [] for vdd in vdds_to_plot},
        'total_influence': {vdd: [] for vdd in vdds_to_plot},
    }
    
    calculator = MetricsCalculator()
    
    for vdd in vdds_to_plot:
        print(f"  Vdd={vdd}V: ", end='', flush=True)
        for temp in temp_range:
            puf = puf_factory(temperature=temp, vdd=vdd, seed=1)
            
            rel = calculator.calc_reliability(puf, N_challenges=100000, r=5)
            uniq = calculator.calc_uniqueness(puf_factory, temp, vdd, N=100000, num_instances=5)
            b = calculator.calc_bias(puf, N=100000)
            sim = calculator.calc_similarity(puf_factory, temp, vdd, N=100000)
            total_inf = calculator.calc_total_influence(puf, N=100000)
            
            metrics_data['reliability'][vdd].append(rel)
            metrics_data['uniqueness'][vdd].append(uniq)
            metrics_data['bias'][vdd].append(b)
            metrics_data['similarity'][vdd].append(sim)
            metrics_data['total_influence'][vdd].append(total_inf)
            
            print(".", end='', flush=True)
        print(" Done")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{puf_name}: Metrics vs Temperature (Various Vdd)', fontsize=16, fontweight='bold')
    
    # Plot each metric
    ax = axes[0, 0]
    for vdd in vdds_to_plot:
        ax.plot(temp_range, metrics_data['reliability'][vdd], marker='o', label=f'Vdd={vdd}V')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Reliability')
    ax.set_title('Reliability vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for vdd in vdds_to_plot:
        ax.plot(temp_range, metrics_data['uniqueness'][vdd], marker='s', label=f'Vdd={vdd}V')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Uniqueness')
    ax.set_title('Uniqueness vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    for vdd in vdds_to_plot:
        ax.plot(temp_range, metrics_data['bias'][vdd], marker='^', label=f'Vdd={vdd}V')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Absolute Bias')
    ax.set_title('Bias vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for vdd in vdds_to_plot:
        ax.plot(temp_range, metrics_data['similarity'][vdd], marker='d', label=f'Vdd={vdd}V')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for vdd in vdds_to_plot:
        ax.plot(temp_range, metrics_data['total_influence'][vdd], marker='v', label=f'Vdd={vdd}V')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Total Influence')
    ax.set_title('Total Influence vs Temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"Summary Statistics:\n\n"
    for metric_name, metric_data in metrics_data.items():
        all_values = [v for values in metric_data.values() for v in values if not np.isnan(v)]
        if all_values:
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            summary_text += f"{metric_name}:\n  Mean: {mean_val:.4f}\n  Std: {std_val:.4f}\n\n"
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center', 
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{puf_name}_metrics_vs_temperature.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    return metrics_data

def plot_metrics_vs_voltage(puf_name, puf_factory, temps_to_plot, vdd_range, output_dir):
    """
    Plot all metrics vs voltage for multiple constant temperatures
    
    Args:
        puf_name: Name of PUF type
        puf_factory: Factory function to create PUF instances
        temps_to_plot: List of temperature values to plot
        vdd_range: Range of voltages to sweep
        output_dir: Directory to save plots
    """
    print(f"\nCalculating metrics for {puf_name} vs Voltage...")
    
    metrics_data = {
        'reliability': {temp: [] for temp in temps_to_plot},
        'uniqueness': {temp: [] for temp in temps_to_plot},
        'bias': {temp: [] for temp in temps_to_plot},
        'similarity': {temp: [] for temp in temps_to_plot},
        'total_influence': {temp: [] for temp in temps_to_plot},
    }
    
    calculator = MetricsCalculator()
    
    for temp in temps_to_plot:
        print(f"  T={temp}°C: ", end='', flush=True)
        for vdd in vdd_range:
            puf = puf_factory(temperature=temp, vdd=vdd, seed=1)
            
            rel = calculator.calc_reliability(puf, N_challenges=100000, r=5)
            uniq = calculator.calc_uniqueness(puf_factory, temp, vdd, N=100000, num_instances=5)
            b = calculator.calc_bias(puf, N=100000)
            sim = calculator.calc_similarity(puf_factory, temp, vdd, N=100000)
            total_inf = calculator.calc_total_influence(puf, N=100000)
            
            metrics_data['reliability'][temp].append(rel)
            metrics_data['uniqueness'][temp].append(uniq)
            metrics_data['bias'][temp].append(b)
            metrics_data['similarity'][temp].append(sim)
            metrics_data['total_influence'][temp].append(total_inf)
            
            print(".", end='', flush=True)
        print(" Done")
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{puf_name}: Metrics vs Voltage (Various Temperature)', fontsize=16, fontweight='bold')
    
    # Plot each metric
    ax = axes[0, 0]
    for temp in temps_to_plot:
        ax.plot(vdd_range, metrics_data['reliability'][temp], marker='o', label=f'T={temp}°C')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Reliability')
    ax.set_title('Reliability vs Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    for temp in temps_to_plot:
        ax.plot(vdd_range, metrics_data['uniqueness'][temp], marker='s', label=f'T={temp}°C')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Uniqueness')
    ax.set_title('Uniqueness vs Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    for temp in temps_to_plot:
        ax.plot(vdd_range, metrics_data['bias'][temp], marker='^', label=f'T={temp}°C')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Absolute Bias')
    ax.set_title('Bias vs Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    for temp in temps_to_plot:
        ax.plot(vdd_range, metrics_data['similarity'][temp], marker='d', label=f'T={temp}°C')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Similarity')
    ax.set_title('Similarity vs Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    for temp in temps_to_plot:
        ax.plot(vdd_range, metrics_data['total_influence'][temp], marker='v', label=f'T={temp}°C')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Total Influence')
    ax.set_title('Total Influence vs Voltage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"Summary Statistics:\n\n"
    for metric_name, metric_data in metrics_data.items():
        all_values = [v for values in metric_data.values() for v in values if not np.isnan(v)]
        if all_values:
            mean_val = np.mean(all_values)
            std_val = np.std(all_values)
            summary_text += f"{metric_name}:\n  Mean: {mean_val:.4f}\n  Std: {std_val:.4f}\n\n"
    ax.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center', 
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{puf_name}_metrics_vs_voltage.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    return metrics_data

def run_comprehensive_sweep():
    """Run comprehensive metrics sweeps and generate plots"""
    
    output_dir = 'graphs_metrics'
    os.makedirs(output_dir, exist_ok=True)
    
    factories = create_puf_factories()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE METRICS SWEEP AND PLOTTING")
    print("="*80)
    
    # User configuration
    print("\nConfiguration Options:")
    print("  1: Full sweep (all temperatures and voltages)")
    print("  2: Quick sweep (fewer points)")
    print("  3: Custom sweep")
    
    config_choice = input("Choose configuration (default: 1): ").strip() or '1'
    
    if config_choice == '1':  # Full sweep
        temp_range = np.arange(20, 81, 10)  # 20-80°C in 10°C steps
        vdd_range = np.arange(1.0, 2.61, 0.2)  # 1.0-2.6V in 0.2V steps
        temps_to_plot = [20, 50, 80]
        vdds_to_plot = [1.0, 1.5, 2.5]
        
    elif config_choice == '2':  # Quick sweep
        temp_range = np.arange(20, 81, 20)  # 20-80°C in 20°C steps
        vdd_range = np.arange(1.0, 2.61, 0.4)  # 1.0-2.6V in 0.4V steps
        temps_to_plot = [20, 50, 80]
        vdds_to_plot = [1.0, 1.8]
        
    else:  # Custom
        try:
            t_start = float(input("Temperature start (°C, default 20): ") or "20")
            t_end = float(input("Temperature end (°C, default 80): ") or "80")
            t_step = float(input("Temperature step (°C, default 20): ") or "20")
            temp_range = np.arange(t_start, t_end + 1, t_step)
            
            v_start = float(input("Voltage start (V, default 1.0): ") or "1.0")
            v_end = float(input("Voltage end (V, default 2.6): ") or "2.6")
            v_step = float(input("Voltage step (V, default 0.4): ") or "0.4")
            vdd_range = np.arange(v_start, v_end + 0.01, v_step)
            
            temps_to_plot = list(np.linspace(t_start, t_end, 3).astype(int))
            vdds_to_plot = list(np.linspace(v_start, v_end, 3))
            
        except ValueError:
            print("Invalid input, using default full sweep")
            temp_range = np.arange(0, 151, 1)
            vdd_range = np.arange(0.5, 3.01, 0.01)
            temps_to_plot = [25, 75, 125]
            vdds_to_plot = [1.0, 1.35, 2.5]
    
    print(f"\nSweep Configuration:")
    print(f"  Temperature: {temp_range[0]:.0f}-{temp_range[-1]:.0f}°C ({len(temp_range)} points)")
    print(f"  Voltage: {vdd_range[0]:.2f}-{vdd_range[-1]:.2f}V ({len(vdd_range)} points)")
    print(f"  Total points per PUF: {len(temp_range) * len(vdd_range)}")
    print(f"  Plotting temperatures: {temps_to_plot}")
    print(f"  Plotting voltages: {[f'{v:.2f}' for v in vdds_to_plot]}")
    
    # Select PUF(s)
    print("\nSelect PUF(s) to analyze:")
    print("  0: All PUFs")
    for i, (name, _) in enumerate(factories, start=1):
        print(f"  {i}: {name}")
    
    puf_choice = input("Choice (default: 0 for all): ").strip() or '0'
    
    if puf_choice == '0':
        selected_pufs = factories
    else:
        try:
            idx = int(puf_choice) - 1
            selected_pufs = [factories[idx]]
        except (ValueError, IndexError):
            selected_pufs = factories
    
    # Run sweeps
    print("\n" + "="*80)
    print("SWEEPING AND PLOTTING METRICS")
    print("="*80)
    
    for puf_name, puf_factory in selected_pufs:
        print(f"\n{'='*80}")
        print(f"Processing: {puf_name}")
        print(f"{'='*80}")
        
        try:
            # Temperature sweep
            plot_metrics_vs_temperature(puf_name, puf_factory, vdds_to_plot, temp_range, output_dir)
            
            # Voltage sweep
            plot_metrics_vs_voltage(puf_name, puf_factory, temps_to_plot, vdd_range, output_dir)
            
            print(f"✓ {puf_name} completed successfully")
            
        except Exception as e:
            print(f"✗ Error processing {puf_name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("SWEEP COMPLETE!")
    print(f"All plots saved to: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_comprehensive_sweep()
