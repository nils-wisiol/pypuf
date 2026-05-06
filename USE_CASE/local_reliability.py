"""
Multi-constant reliability trend extraction.

For each PUF:
    - Sweep temperature for multiple fixed Vdd values
    - Sweep voltage for multiple fixed temperature values
    - Fit linear trend
    - Store max, min, median, slope

Outputs CSV file with full statistics.
"""

import numpy as np
import csv
from pypuf.simulation.bistable import XORBistableRingPUF
from pypuf.simulation.delay import (
    XORArbiterPUF, FeedForwardArbiterPUF,
    XORFeedForwardArbiterPUF, ArbiterPUF,
    LightweightSecurePUF, PermutationPUF, InterposePUF
)
from pypuf.metrics import reliability


# ===============================
# Global Parameters
# ===============================
n = 64
k_xor = 4
k_xorff = 3
N_CHALLENGES = 100000
R_REPEATS = 5


# ===============================
# PUF Factories
# ===============================
def create_puf_factories():
    return [
        ("ArbiterPUF", lambda **kw: ArbiterPUF(
            n=n, noisiness=kw.pop('noisiness', 0.05), **kw)),

        ("XORArbiterPUF", lambda **kw: XORArbiterPUF(
            n=n, k=k_xor, noisiness=kw.pop('noisiness', 0.05), **kw)),

        ("XORBistableRingPUF", lambda **kw: (
            np.random.seed(kw.pop('seed', None)),
            XORBistableRingPUF(
                n=n,
                k=k_xor,
                weights=np.random.normal(0, 1, (k_xor, n + 1)),
                temperature=kw.pop('temperature', 25),
                vdd=kw.pop('vdd', 1.35)
            )
        )[1]),

        ("FeedForwardArbiterPUF", lambda **kw: FeedForwardArbiterPUF(
            n=n, ff=[(2, 5), (4, 7)],
            noisiness=kw.pop('noisiness', 0.05), **kw)),

        ("XORFeedForwardArbiterPUF", lambda **kw: XORFeedForwardArbiterPUF(
            n=n, k=k_xorff,
            ff=[[(2, 5)], [(4, 7)], [(1, 6)]],
            noisiness=kw.pop('noisiness', 0.05), **kw)),

        ("LightweightSecurePUF", lambda **kw: LightweightSecurePUF(
            n=n, k=k_xor,
            noisiness=kw.pop('noisiness', 0.05), **kw)),

        ("PermutationPUF", lambda **kw: PermutationPUF(
            n=n, k=k_xor,
            noisiness=kw.pop('noisiness', 0.05), **kw)),

        ("InterposePUF", lambda **kw: InterposePUF(
            n=n, k_down=k_xor, k_up=2,
            noisiness=kw.pop('noisiness', 0.05), **kw)),
    ]


# ===============================
# Reliability Calculation
# ===============================
def calc_reliability(puf):
    rel = reliability(puf, seed=42, N=N_CHALLENGES, r=R_REPEATS)
    return np.mean(rel)


# ===============================
# Linear Trend + Statistics
# ===============================
def analyze_trend(x, y):
    slope, intercept = np.polyfit(x, y, 1)

    return (
        float(np.max(y)),
        float(np.min(y)),
        float(np.median(y)),
        float(slope)
    )


# ===============================
# Main Execution
# ===============================
def run_multi_constant_sweep():

    output_file = "multi_constant_reliability_analysis.csv"

    # Full sweep ranges
    temp_range = np.arange(20, 81, 10)
    vdd_range = np.arange(1.0, 2.61, 0.2)

    # Multiple constant anchor points
    constant_vdds = [1.0, 1.5, 2.0, 2.5]
    constant_temps = [20, 40, 60, 80]

    factories = create_puf_factories()

    results = []

    print("\nStarting Multi-Constant Reliability Analysis...\n")

    for puf_name, puf_factory in factories:
        print(f"Processing {puf_name}")

        # ==========================================
        # 1) Temperature sweeps at multiple Vdd
        # ==========================================
        for vdd_fixed in constant_vdds:

            reliability_values = []

            for temp in temp_range:
                puf = puf_factory(temperature=temp, vdd=vdd_fixed, seed=1)
                reliability_values.append(calc_reliability(puf))

            max_v, min_v, median_v, slope_v = analyze_trend(
                temp_range, reliability_values)

            results.append([
                puf_name,
                "Temperature Sweep",
                f"Vdd={vdd_fixed}",
                max_v,
                min_v,
                median_v,
                slope_v
            ])

        # ==========================================
        # 2) Voltage sweeps at multiple Temperature
        # ==========================================
        for temp_fixed in constant_temps:

            reliability_values = []

            for vdd in vdd_range:
                puf = puf_factory(temperature=temp_fixed, vdd=vdd, seed=1)
                reliability_values.append(calc_reliability(puf))

            max_v, min_v, median_v, slope_v = analyze_trend(
                vdd_range, reliability_values)

            results.append([
                puf_name,
                "Voltage Sweep",
                f"T={temp_fixed}",
                max_v,
                min_v,
                median_v,
                slope_v
            ])

    # ==========================================
    # Save CSV
    # ==========================================
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "PUF Type",
            "Sweep Type",
            "Constant Parameter",
            "Max Reliability",
            "Min Reliability",
            "Median Reliability",
            "Slope"
        ])
        writer.writerows(results)

    print("\nAnalysis Complete.")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    run_multi_constant_sweep()