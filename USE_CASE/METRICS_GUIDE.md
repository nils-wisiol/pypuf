# PUF Metrics Evaluation Guide

## Overview
This file (`test_metrics_all_pufs.py`) provides comprehensive testing of pypuf metrics functions with temperature and voltage support for all PUF types.

## Supported Metrics

### 1. **Reliability**
- **Definition**: Probability that a PUF generates the same response when evaluated multiple times on the same challenge with noise
- **Ideal Value**: Close to 1.0 (100% reproducible)
- **Sensitivity**: Affected by noise level and temperature/voltage variations
- **Usage**: `reliability(puf, seed, N, r)` - queries each challenge `r` times

### 2. **Uniqueness**
- **Definition**: Inter-device variation - how different PUF instances respond to the same challenges
- **Ideal Value**: Around 0.5 (equal probability of same/different responses between devices)
- **Sensitivity**: Should be unaffected by environment if devices are different
- **Usage**: `uniqueness(instances, seed, N)` - requires multiple PUF instances

### 3. **Bias**
- **Definition**: Probability of getting a `+1` vs `-1` response
- **Ideal Value**: Close to 0 (equal probability of both)
- **Sensitivity**: Can shift with temperature/voltage
- **Usage**: `bias(puf, seed, N)` - simpler, single instance

### 4. **Similarity**
- **Definition**: How similar two different PUF instances are in their response behavior
- **Ideal Value**: Close to 0.5 (no more similar than random chance)
- **Sensitivity**: Should stay relatively constant across environments
- **Usage**: `similarity(puf1, puf2, seed, N)` - compares two instances

### 5. **Influence (Fourier Analysis)**
- **Definition**: How much changing a single input bit affects the output
- **Ideal Value**: Around 0.5 per bit (equal effect from all bits)
- **Sensitivity**: Structural property - shouldn't change with temperature/voltage
- **Usage**: `influence(puf, bit_index, seed, N)` - tests one bit at a time

### 6. **Total Influence**
- **Definition**: Sum of all bit influences (average sensitivity)
- **Ideal Value**: Around n/2 where n is challenge length
- **Sensitivity**: Structural property - independent of environment
- **Usage**: `total_influence(puf, seed, N)` - faster than individual influences

## Temperature and Voltage Parameters

All metrics support optional environment parameters:
```python
puf.temperature = 50  # 0-150°C
puf.vdd = 2.0         # 0.5-3.0V
```

This allows testing how metrics change across environmental conditions.

## PUF Types Supported

1. **ArbiterPUF** - Single arbiter chain
2. **XORArbiterPUF** - Multiple arbiter chains with XOR
3. **XORBistableRingPUF** - Bistable ring PUF variant
4. **FeedForwardArbiterPUF** - Arbiter with feedback connections
5. **XORFeedForwardArbiterPUF** - XOR of feedback-based arbiters
6. **LightweightSecurePUF** - Enhanced security variant
7. **PermutationPUF** - With permutation transform
8. **InterposePUF** - Two-stage interposed architecture

## Running the Script

```bash
python test_metrics_all_pufs.py
```

The script provides an interactive menu to:
1. Select specific metric(s) to test
2. Set environment conditions (temperature, voltage)
3. Configure sample size (number of challenges)

## Example Output

```
COMPREHENSIVE PUF METRICS EVALUATION
==============================

Running ALL metrics (T=25°C, Vdd=1.35V, N=1000)

ArbiterPUF:
  Reliability:      0.9542
  Uniqueness:       0.4923
  Bias:             -0.0245
  Similarity:       0.5187
  Influence(bit 0): 0.0420
  Total Influence:  8.3245
```

## What These Results Mean

- **High Reliability (>0.9)** + **Stable with Env Changes** → Environment scaling working correctly
- **Uniqueness ~0.5** → Good inter-device variation
- **Bias ~0** → No systematic output bias
- **Similarity ~0.5** → Different devices as expected
- **Influences Vary** → Some bits matter more (expected for Arbiter PUF)
- **Total Influence** → Measure of circuit complexity

## Troubleshooting

- **"Error: seed not supported"** → Some PUF types need seed passed at creation time
- **"Temperature should be between 0°C and 150°C"** → Check temperature input range
- **Metrics don't change with Vdd** → May not have implemented V_factor=True in PUF
- **Reliability = 1.0 always** → No noisiness; set noisiness > 0

## Next Steps

- Test different noisiness levels
- Compare metrics across temperature ranges
- Analyze which PUF types are most stable
- Validate environment scaling implementation
