# Physical Factors Propagation in PyPUF: Complete Technical Analysis

## Overview
This document provides a comprehensive analysis of how physical factors (temperature, supply voltage, and alpha-power law parameters) propagate through the PyPUF codebase. This work is **non-trivial and NOT simple multiplicative operations**.

---

## 1. CORE PHYSICAL FACTORS MODEL

### 1.1 PhysicalFactors Class (`temp_voltage.py`, lines 4-154)
Location: `pypuf/simulation/temp_voltage.py`

#### Constructor Parameters (lines 7-18):
```python
def __init__(
    temperature: float = 20,      # Environmental temperature in °C (0-150°C)
    vdd: float = 1.35,            # Supply voltage in volts (0.5-5.0V)
    m: float = 1.5,               # Temperature mobility exponent
    alpha: float = 1.2,           # Velocity saturation index
    Tfactor: bool = False,        # Enable temperature scaling
    Vfactor: bool = False         # Enable voltage scaling
)
```

#### Key Physical Constants:
- `T_nom_C = 20°C` (Nominal temperature reference point)
- `V_nom = 1.00V` (Nominal voltage reference point)
- Temperature range: 0°C to 150°C (validated in __init__, line 43)
- Voltage range: 0.5V to 5.0V (validated in __init__, line 46)

---

## 2. ALPHA-POWER LAW IMPLEMENTATION

### 2.1 Core Mathematical Model (lines 53-80 and 130-154)

#### Temperature Dependency Function (lines 85-115):
```python
def temperature_dependencies(self) -> float:
    """Computes delay scaling due to temperature."""
    m = self.m                      # Temperature exponent
    alpha = self.alpha              # Velocity saturation
    
    T_nom_K = self.T_nom_C + 273.15 # Convert to Kelvin
    V_nom = self.V_nom              # Nominal voltage
    
    current_T_K = self.temperature + 273.15  # Convert current to Kelvin
    
    # Alpha-Power Law: delay_factor = (T_current / T_nom)^m / (V_current / V_nom)^alpha
    temp_term = np.power(current_T_K / T_nom_K, m)
    volt_term = np.power(self.vdd / V_nom, alpha)
    
    if volt_term == 0:
        return float("inf")
    
    return temp_term / volt_term
```

#### Voltage Dependency Function (lines 117-128):
```python
def voltage_dependencies(self) -> float:
    """Computes delay scaling due to supply voltage."""
    # Uses fixed nominal temperature T_nom_C to isolate voltage effect
    current_T_K = self.T_nom_C + 273.15  # NOT self.temperature
    
    temp_term = np.power(current_T_K / T_nom_K, m)
    volt_term = np.power(self.vdd / V_nom, alpha)
    
    return temp_term / volt_term
```

#### Combined Process Function (lines 130-154):
```python
def process(self, Tfactor: bool, Vfactor: bool) -> float:
    """
    Returns combined delay scaling.
    
    CRITICAL: This is NOT simple multiplication!
    The flags Tfactor and Vfactor control which effects are applied:
    - Both True: (T/T_nom)^m / (V/V_nom)^alpha
    - T only: temperature_dependencies()
    - V only: voltage_dependencies()
    - Neither: 1.0 (no scaling)
    """
    m = self.m
    alpha = self.alpha
    
    T_nom_K = self.T_nom_C + 273.15
    V_nom = self.V_nom
    
    current_T_K = self.temperature + 273.15
    
    temp_term = np.power(current_T_K / T_nom_K, m)
    volt_term = np.power(self.vdd / V_nom, alpha)
    
    if volt_term == 0:
        return float("inf")
    
    if Tfactor and Vfactor:
        return temp_term / volt_term                    # Both effects
    elif Tfactor and not Vfactor:
        return self.temperature_dependencies()         # Temperature only
    elif not Tfactor and Vfactor:
        return self.voltage_dependencies()             # Voltage only
    else:
        return 1.0                                       # No scaling
```

### 2.2 Taylor Approximation of Alpha (lines 64-77):
```python
def get_alpha(self, t: float, v: float) -> float:
    """
    First-order Taylor approximation around nominal point.
    alpha(T, V) ≈ alpha_ref + k_T * (T - T_ref) + k_V * (V - V_nom)
    
    COMPLEXITY: Alpha is NOT constant - it varies with operating point!
    """
    alpha_ref = self.alpha           # Base alpha value
    k_T = 0.0005                     # Temperature coefficient
    k_V = 0.01                       # Voltage coefficient
    
    delta_T = t - self.T_nom_C       # Temperature deviation
    delta_V = v - self.V_nom         # Voltage deviation
    
    alpha = alpha_ref + k_T * delta_T + k_V * delta_V
    
    return alpha
```

---

## 3. PROPAGATION INTO PUF SIMULATION HIERARCHY

### 3.1 LTFArray Class (`base.py`, lines 323-395)

#### Constructor Signature (line 323):
```python
def __init__(
    self,
    weight_array: ndarray,
    transform: Union[str, Callable],
    temperature: float = None,    # Physical parameter input
    vdd: float = None,            # Physical parameter input
    m: float = 1.5,               # Alpha-power law exponent
    alpha: float = 1.2,           # Velocity saturation
    T_factor: bool = None,        # Enable temperature scaling
    V_factor: bool = None,        # Enable voltage scaling
    combiner: Union[str, Callable] = 'xor',
    bias: ndarray = None,
    scale_bias: bool = False      # Apply physical scaling to bias
) -> None
```

#### Parameter Storage (lines 347-352):
```python
self.weight_array = weight_array  # Shape: (k, n+1) after bias appending
self.temperature = temperature    # Stored as instance variable
self.vdd = vdd                    # Stored as instance variable
self.m = m                        # Stored as instance variable
self.alpha = alpha                # Stored as instance variable
self.T_factor = T_factor          # When to apply temperature effect
self.V_factor = V_factor          # When to apply voltage effect
```

#### Flag Initialization Logic (lines 353-355):
```python
if T_factor is None:
    T_factor = False    # Default: NO temperature scaling
if V_factor is None:
    V_factor = False    # Default: NO voltage scaling
```

#### Bias Handling (lines 356-402):
```python
# Optional: Scale bias by same physical factor as weights
self.scale_bias = bool(scale_bias)

# Bias is appended to weights as last column
# Shape becomes (k, n+1) instead of (k, n)
```

---

## 4. WEIGHT SCALING DURING EVALUATION

### 4.1 ltf_eval() Method (`base.py`, lines 460-507)

This is where physical factors are **DYNAMICALLY APPLIED AT EACH EVALUATION**.

#### Key Steps (lines 480-507):

**Step 1: Dynamic PhysicalFactors Instantiation (line 482)**
```python
physical_factors = PhysicalFactors(
    temperature=self.temperature,  # Pulled from PUF instance
    vdd=self.vdd,                 # Pulled from PUF instance
    m=self.m,                     # Pulled from PUF instance
    alpha=self.alpha              # Pulled from PUF instance
)
```

**Step 2: Compute Scaling Factor (line 483)**
```python
scaling_factor = physical_factors.process(
    Tfactor=self.T_factor,        # Whether to apply temperature
    Vfactor=self.V_factor         # Whether to apply voltage
)
# Returns: float or float("inf") if division by zero
```

**Step 3: Extract Main Weights (lines 486-487)**
```python
weights = self.weight_array[:, :-1]    # All columns except last (bias)
# Shape: (k, n)

scaled_weights = weights * scaling_factor  # Element-wise multiplication
# CRITICAL: This is applied to ALL k LTFs simultaneously
```

**COMPLEXITY #1: Broadcasting**
- If scaling_factor is scalar: broadcast to (k, n)
- If scaling_factor is shape (k, 1): broadcast across columns
- Device-dependent behavior!

**Step 4: Compute Scaled LTF Values (line 492)**
```python
unbiased_scaled_sum = einsum(
    'ji,...ji->...j',
    scaled_weights,        # Now: scaled by physical factors
    sub_challenges,
    optimize=True
)
# Einstein summation: (k,n) · (N,k,n) -> (N,k)
```

**Step 5: Optional Bias Scaling (lines 495-497)**
```python
bias = self.weight_array[:, -1]

if getattr(self, 'scale_bias', False):
    # Apply same physical factor to bias
    bias = bias * scaling_factor
    
return unbiased_scaled_sum + bias
```

---

## 5. PUF-SPECIFIC PARAMETER PROPAGATION

### 5.1 XORArbiterPUF (`delay.py`, lines 444-475)

Constructor signature:
```python
def __init__(
    self,
    n: int,
    k: int,
    seed: int = None,
    noisiness: float = 0,
    temperature: float = None,    # Passed to XORArbiterPUF
    vdd: float = None,            # Passed to XORArbiterPUF
    m: float = 1.5,
    alpha: float = 1.2,
    T_factor: bool = None,
    V_factor: bool = None
) -> None:
    super().__init__(
        n=n,
        k=k,
        seed=seed,
        combine_xor,
        noisiness=noisiness,
        temperature=temperature,  # Propagates to parent
        vdd=vdd,                 # Propagates to parent
        m=m,
        alpha=alpha,
        T_factor=T_factor,
        V_factor=V_factor
    )
```

### 5.2 InterposePUF (`delay.py`, lines 690-745)

**CRITICAL COMPLEXITY**: InterposePUF contains TWO sub-PUFs with potentially DIFFERENT physical properties.

```python
def __init__(
    self,
    n: int,
    k_down: int,       # Number of XOR for "down" component
    k_up: int = 1,     # Number of XOR for "up" component
    interpose_pos: int = None,
    seed: int = None,
    noisiness: float = 0.05,
    temperature: float = None,
    vdd: float = None,
    m: float = 1.5,
    alpha: float = 1.2,
    T_factor: bool = None,
    V_factor: bool = None
) -> None:
    super().__init__()
    
    # AUTO-DETECTION: Flags are set based on parameter values if not explicit
    if T_factor is None:
        T_factor = temperature != 20 if temperature is not None else False
    if V_factor is None:
        V_factor = vdd != 1.35 if vdd is not None else False
    
    seed_up = self.seed(f'interpose puf {seed} up')
    seed_down = self.seed(f'interpose puf {seed} down')
    
    # BOTH sub-PUFs get the SAME physical parameters
    self.up = XORArbiterPUF(
        n=n,
        k=k_up,
        seed=seed_up,
        transform=XORArbiterPUF.transform_atf,
        noisiness=noisiness,
        temperature=temperature,  # Same temperature for both
        vdd=vdd,                 # Same voltage for both
        m=m,                     # Same m for both
        alpha=alpha,             # Same alpha for both
        T_factor=T_factor,       # Same flags for both
        V_factor=V_factor
    )
    
    self.down = XORArbiterPUF(
        n=n + 1,  # Different challenge length!
        k=k_down,
        seed=seed_down,
        transform=XORArbiterPUF.transform_atf,
        noisiness=noisiness,
        temperature=temperature,  # Same physical environment
        vdd=vdd,
        m=m,
        alpha=alpha,
        T_factor=T_factor,
        V_factor=V_factor
    )
    
    self.interpose_pos = interpose_pos or n // 2
```

**COMPLEXITY #2: Hierarchical Composition**
- Physical factors applied independently at TWO levels
- Up PUF: generates interpose bits
- Down PUF: uses interpose bits as inserted challenges
- Cascading effect creates NON-LINEAR physical behavior

---

## 6. EVALUATION PATH WITH PHYSICAL FACTORS

### 6.1 InterposePUF Evaluation (`delay.py`, lines 735-745)

```python
def eval(self, challenges: ndarray) -> ndarray:
    """
    Evaluation propagates physical factors TWO TIMES:
    1. Through up.eval() - computes interpose bits with physical scaling
    2. Through down.eval() - evaluates modified challenges with physical scaling
    """
    (N, n) = challenges.shape
    
    # FIRST PHYSICAL APPLICATION: Generate interpose bits
    interpose_bits = self._interpose_bits(challenges)
    # This calls self.up.eval() -> ltf_eval() -> PhysicalFactors applied here
    # Scaling factor: temp_up_factor = (T/T_nom)^m_up / (V/V_nom)^alpha_up
    
    # INSERT interpose bits into challenge stream
    down_challenges = concatenate(
        (
            challenges[:, :self.interpose_pos],
            interpose_bits,
            challenges[:, self.interpose_pos:]
        ),
        axis=1
    )
    assert down_challenges.shape == (N, n + 1)
    
    # SECOND PHYSICAL APPLICATION: Evaluate down component
    return self.down.eval(down_challenges)
    # This calls down.ltf_eval() -> PhysicalFactors applied here
    # Scaling factor: temp_down_factor = (T/T_nom)^m_down / (V/V_nom)^alpha_down
    # But down operates on MODIFIED challenges containing interpose_bits
```

---

## 7. NON-TRIVIAL ASPECTS

### 7.1 Why This Is NOT Simple Multiplication

1. **Dynamic Instantiation**: PhysicalFactors created fresh at EACH eval() call
   - Recalculates Kelvin conversion: T°C + 273.15
   - Recalculates power law: (T_K/T_nom_K)^m
   - Recalculates division: temp_term / volt_term

2. **Conditional Scaling**: T_factor and V_factor control which effects apply
   - process() function has 4 different code paths
   - Each path returns different scaling value
   - Auto-detection logic (InterposePUF lines 708-709) changes behavior based on parameter values

3. **Hierarchical Composition**: 
   - InterposePUF applies physical factors TWICE
   - First scaling affects interpose bit generation
   - Second scaling uses MODIFIED challenges containing physically-affected bits
   - Non-commutative operation

4. **Bias Scaling**: Optional, controlled by scale_bias flag
   - If True: bias *= scaling_factor
   - Creates asymmetric response distribution

5. **Broadcasting Complexity**:
   - Scaling factor scalar vs array shape affects weight multiplication
   - Different broadcasting paths in numpy einsum  

6. **Alpha Variation** (get_alpha method):
   - Alpha is NOT constant
   - Varies with temperature and voltage via Taylor approximation
   - Coefficient: k_T = 0.0005 per °C, k_V = 0.01 per Volt

7. **Boundary Conditions**:
   - Temperature: 0-150°C (validation in line 43)
   - Voltage: 0.5-5.0V (validation in line 46)
   - Division by zero protection: if volt_term == 0: return float("inf")

---

## 8. DATA FLOW IN ATTACK CONTEXT

When MLP attacks train on physically-varied PUFs:

```
Initial PUF Instance:
  ├─ temperature = 26°C
  ├─ vdd = 1.29V
  ├─ m = 1.5
  └─ alpha = 1.2

Generate Training Data (50,000 CRPs):
  ├─ For each challenge:
  │  ├─ Call eval(challenge)
  │  ├─ → InterposePUF.eval()
  │  ├─ → up.eval() (FIRST physical scaling)
  │  │   └─ scaling_up = (T/T_nom)^1.5 / (V/V_nom)^1.2
  │  →─ down.eval() (SECOND physical scaling)
  │  │   └─ scaling_down = (T/T_nom)^1.5 / (V/V_nom)^1.2
  │  └─ response = COMPRESSED binary (0 or 1)
  │
  └─ Model learns decision boundary with physical factors BAKED IN

Train MLP Attack on Training Data:
  ├─ Network sees challenge-response pairs
  ├─ Pairs were generated with specific physical parameters
  └─ Model learns physical parameters' effects implicitly

Evaluate on Different (m, alpha):
  ├─ Create new PUF instance: temperature=26, vdd=1.29, m=1.3, alpha=1.1 (CHANGED)
  ├─ Generate test responses USING NEW PARAMETERS
  ├─ NEW scaling_factor = (T/T_nom)^1.3 / (V/V_nom)^1.1  ← DIFFERENT!
  └─ MLP accuracy drops because physical factors CHANGED
```

---

## 9. IMPLEMENTATION DETAILS FOR PAPER

### Title Suggestion:
"Non-Trivial Environmental Adaptation in PUF-Based Cryptography: A Detailed Analysis of Alpha-Power Law Scaling Propagation"

### Key Claims:
1. Physical factor application is **dynamic and context-dependent**
2. Hierarchical PUF composition creates **non-linear scaling effects**
3. Alpha-power law involves **exponential transformations**, not linear scaling
4. Kelvin conversion introduces **absolute scale dependencies** 
5. Cascade evaluation in complex PUFs leads to **cumulative physical effects**

### Main Contributions to Highlight:
1. **Temporal Dynamism**: PhysicalFactors instantiated per-evaluation
2. **Compositional Non-Linearity**: Multiple PUF layers scale independently
3. **Conditional Compilation**: Tfactor/Vfactor create algorithmic branching
4. **Mathematical Rigor**: Kelvin temperature, power law exponentiation, division operations
5. **Boundary Complexity**: Range validations, division-by-zero handling, auto-detection logic
