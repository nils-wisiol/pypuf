"""
Comprehensive metrics evaluation for all PUF types with temperature and voltage support.
Tests: reliability, uniqueness, bias, similarity, correlation, influence, total_influence
"""
import sys
import os
import numpy as np
from pypuf.simulation.bistable import XORBistableRingPUF
from pypuf.simulation.delay import (XORArbiterPUF, FeedForwardArbiterPUF, 
                                     XORFeedForwardArbiterPUF, ArbiterPUF, 
                                     LightweightSecurePUF, PermutationPUF, InterposePUF)
from pypuf.metrics import reliability, uniqueness, bias, similarity, correlation_data
from pypuf.metrics.fourier import influence, total_influence
from pypuf.io import random_inputs, ChallengeResponseSet

# Global parameters
n = 16
k_xor = 4
k_xorff = 3

# Define all PUF types
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

def test_reliability(puf, temperature=None, vdd=None, N=1000, r=5):
    """Test reliability metric"""
    try:
        puf.temperature = temperature if temperature is not None else puf.temperature
        puf.vdd = vdd if vdd is not None else puf.vdd
        rel = reliability(puf, seed=42, N=N, r=r)
        return np.mean(rel)
    except Exception as e:
        return f"Error: {str(e)}"

def test_uniqueness(puf_factory, temperature=None, vdd=None, N=1000, num_instances=5):
    """Test uniqueness metric with multiple instances"""
    try:
        instances = [puf_factory(temperature=temperature if temperature is not None else 25, 
                                vdd=vdd if vdd is not None else 1.35, 
                                seed=i) for i in range(num_instances)]
        uniq = uniqueness(instances, seed=42, N=N)
        return np.mean(uniq)
    except Exception as e:
        return f"Error: {str(e)}"

def test_bias(puf, temperature=None, vdd=None, N=1000):
    """Test bias metric"""
    try:
        puf.temperature = temperature if temperature is not None else puf.temperature
        puf.vdd = vdd if vdd is not None else puf.vdd
        b = bias(puf, seed=42, N=N)
        return np.mean(b)
    except Exception as e:
        return f"Error: {str(e)}"

def test_similarity(puf_factory, temperature=None, vdd=None, N=1000):
    """Test similarity metric between two instances"""
    try:
        puf1 = puf_factory(temperature=temperature if temperature is not None else 25, 
                          vdd=vdd if vdd is not None else 1.35, 
                          seed=1)
        puf2 = puf_factory(temperature=temperature if temperature is not None else 25, 
                          vdd=vdd if vdd is not None else 1.35, 
                          seed=2)
        sim = similarity(puf1, puf2, seed=42, N=N)
        return np.mean(sim)
    except Exception as e:
        return f"Error: {str(e)}"

def test_influence(puf, temperature=None, vdd=None, bit_index=0, N=1000):
    """Test influence metric (sensitivity to bit changes)"""
    try:
        puf.temperature = temperature if temperature is not None else puf.temperature
        puf.vdd = vdd if vdd is not None else puf.vdd
        inf = influence(puf, i=bit_index, seed=42, N=N)
        return inf
    except Exception as e:
        return f"Error: {str(e)}"

def test_total_influence(puf, temperature=None, vdd=None, N=1000):
    """Test total influence metric (sum of all bit influences)"""
    try:
        puf.temperature = temperature if temperature is not None else puf.temperature
        puf.vdd = vdd if vdd is not None else puf.vdd
        total_inf = total_influence(puf, seed=42, N=N)
        return total_inf
    except Exception as e:
        return f"Error: {str(e)}"

def run_comprehensive_test():
    """Run all metrics tests for all PUF types"""
    factories = create_puf_factories()
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PUF METRICS EVALUATION")
    print("="*80)
    
    # Select metric to test
    metrics_menu = {
        '1': ('Reliability', 'test_reliability'),
        '2': ('Uniqueness', 'test_uniqueness'),
        '3': ('Bias', 'test_bias'),
        '4': ('Similarity', 'test_similarity'),
        '5': ('Influence', 'test_influence'),
        '6': ('Total Influence', 'test_total_influence'),
        '7': ('All Metrics', 'all'),
    }
    
    print("\nSelect metric(s) to test:")
    for key, (name, _) in metrics_menu.items():
        print(f"  {key}: {name}")
    
    metric_choice = input("Choice (default: 7 for all): ").strip() or '7'
    
    # Select environment conditions
    print("\nSelect environment conditions:")
    print("  1: Default (T=25°C, Vdd=1.35V)")
    print("  2: Custom temperature")
    print("  3: Custom voltage")
    print("  4: Custom both")
    
    env_choice = input("Choice (default: 1): ").strip() or '1'
    
    temperature = None
    vdd = None
    
    if env_choice in ['2', '4']:
        try:
            temperature = float(input("Enter temperature (0-150°C, default 25): ") or "25")
        except ValueError:
            temperature = 25
    
    if env_choice in ['3', '4']:
        try:
            vdd = float(input("Enter Vdd (0.5-3.0V, default 1.35): ") or "1.35")
        except ValueError:
            vdd = 1.35
    
    # Select sample size
    try:
        N = int(input("Number of challenges (default 1000): ") or "1000")
    except ValueError:
        N = 1000
    
    # Run tests
    if metric_choice == '7':  # All metrics
        print("\n" + "="*80)
        print(f"Running ALL metrics (T={temperature}°C, Vdd={vdd}V, N={N})")
        print("="*80)
        
        for puf_name, factory in factories:
            print(f"\n{puf_name}:")
            print("-" * 40)
            
            try:
                # Reliability
                puf = factory(temperature=temperature, vdd=vdd, seed=1)
                rel = test_reliability(puf, temperature, vdd, N=N, r=3)
                print(f"  Reliability:      {rel:.4f}" if isinstance(rel, (int, float)) else f"  Reliability:      {rel}")
                
                # Uniqueness
                uniq = test_uniqueness(factory, temperature, vdd, N=N, num_instances=5)
                print(f"  Uniqueness:       {uniq:.4f}" if isinstance(uniq, (int, float)) else f"  Uniqueness:       {uniq}")
                
                # Bias
                puf = factory(temperature=temperature, vdd=vdd, seed=1)
                b = test_bias(puf, temperature, vdd, N=N)
                print(f"  Bias:             {b:.4f}" if isinstance(b, (int, float)) else f"  Bias:             {b}")
                
                # Similarity
                sim = test_similarity(factory, temperature, vdd, N=N)
                print(f"  Similarity:       {sim:.4f}" if isinstance(sim, (int, float)) else f"  Similarity:       {sim}")
                
                # Influence (first bit only for speed)
                puf = factory(temperature=temperature, vdd=vdd, seed=1)
                inf = test_influence(puf, temperature, vdd, bit_index=0, N=500)
                print(f"  Influence(bit 0): {inf:.4f}" if isinstance(inf, (int, float)) else f"  Influence(bit 0): {inf}")
                
                # Total Influence
                puf = factory(temperature=temperature, vdd=vdd, seed=1)
                total_inf = test_total_influence(puf, temperature, vdd, N=500)
                print(f"  Total Influence:  {total_inf:.4f}" if isinstance(total_inf, (int, float)) else f"  Total Influence:  {total_inf}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
    
    elif metric_choice in metrics_menu:
        metric_name, test_func = metrics_menu[metric_choice][0], metrics_menu[metric_choice][1]
        print("\n" + "="*80)
        print(f"Running {metric_name} (T={temperature}°C, Vdd={vdd}V, N={N})")
        print("="*80)
        
        for puf_name, factory in factories:
            print(f"\n{puf_name}:")
            try:
                if test_func == 'test_reliability':
                    puf = factory(temperature=temperature, vdd=vdd, seed=1)
                    result = test_reliability(puf, temperature, vdd, N=N, r=3)
                elif test_func == 'test_uniqueness':
                    result = test_uniqueness(factory, temperature, vdd, N=N, num_instances=5)
                elif test_func == 'test_bias':
                    puf = factory(temperature=temperature, vdd=vdd, seed=1)
                    result = test_bias(puf, temperature, vdd, N=N)
                elif test_func == 'test_similarity':
                    result = test_similarity(factory, temperature, vdd, N=N)
                elif test_func == 'test_influence':
                    puf = factory(temperature=temperature, vdd=vdd, seed=1)
                    result = test_influence(puf, temperature, vdd, bit_index=0, N=500)
                elif test_func == 'test_total_influence':
                    puf = factory(temperature=temperature, vdd=vdd, seed=1)
                    result = test_total_influence(puf, temperature, vdd, N=500)
                
                if isinstance(result, (int, float)):
                    print(f"  Result: {result:.6f}")
                else:
                    print(f"  Result: {result}")
                    
            except Exception as e:
                print(f"  ERROR: {str(e)}")
    
    print("\n" + "="*80)
    print("Test completed!")
    print("="*80 + "\n")

if __name__ == "__main__":
    os.makedirs('graphs', exist_ok=True)
    run_comprehensive_test()
