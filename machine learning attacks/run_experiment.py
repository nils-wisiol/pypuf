import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

from pypuf.simulation import (
    ArbiterPUF,
    XORArbiterPUF,
    FeedForwardArbiterPUF,
    InterposePUF,
    LightweightSecurePUF,
    PermutationPUF
)

from pypuf.io import ChallengeResponseSet
from pypuf.attack import MLPAttack2021
from pypuf.metrics import similarity


# -------------------------------------------------
# Experiment Configuration
# -------------------------------------------------

n_bits = 64
puf_seed = 123

fixed_temp = 26
fixed_vdd = 1.29

n_train = 50000
n_test = 2000

m_range = np.array([1.2, 1.3, 1.4, 1.5, 1.6])
alpha_range = np.array([1.0, 1.1, 1.2, 1.3, 1.4])

CPU_CORES = multiprocessing.cpu_count()


# -------------------------------------------------
# PUF Architectures
# -------------------------------------------------

puf_types = {
    "Arbiter": ArbiterPUF,
    "XORArbiter": XORArbiterPUF,
    "FeedForward": FeedForwardArbiterPUF,
    "Interpose": InterposePUF,
    "LightweightSecure": LightweightSecurePUF,
    "Permutation": PermutationPUF
}


# -------------------------------------------------
# Parameter helpers
# -------------------------------------------------

def get_puf_params(puf_class, m=1.0, alpha=1.0):

    puf_name = puf_class.__name__

    base = dict(
        n=n_bits,
        seed=puf_seed,
        noisiness=0.05,
        temperature=fixed_temp,
        vdd=fixed_vdd,
        m=m,
        alpha=alpha
    )

    if puf_name == "XORArbiterPUF":
        base["k"] = 4
    elif puf_name == "FeedForwardArbiterPUF":
        base["ff"] = [(10,20),(30,40)]
    elif puf_name == "LightweightSecurePUF":
        base["k"] = 4
    elif puf_name == "PermutationPUF":
        base["k"] = 4
    elif puf_name == "InterposePUF":
        base["k_down"] = 4
        base["k_up"] = 1

    return base


# -------------------------------------------------
# Train Attack
# -------------------------------------------------

def train_attack(puf_class):

    params = get_puf_params(puf_class)

    puf = puf_class(**params)

    train_crps = ChallengeResponseSet.from_simulation(
        puf,
        N=n_train,
        seed=puf_seed
    )

    model = MLPAttack2021(
        train_crps,
        seed=42,
        net=[n_bits,n_bits],
        epochs=50,
        lr=1e-3,
        bs=128,
        early_stop=0.01
    ).fit()

    return model


# -------------------------------------------------
# Single parameter evaluation
# -------------------------------------------------

def compute_single_accuracy(puf_class, m, alpha, trained_model):

    params = get_puf_params(puf_class, m, alpha)

    p_eval = puf_class(**params)

    acc = similarity(
        p_eval,
        trained_model,
        seed=42,
        N=n_test
    )[0]

    return acc


# -------------------------------------------------
# Run sweep for ONE PUF
# -------------------------------------------------

def run_parameter_sweep(puf_name, puf_class):

    print(f"\nRunning {puf_name}")

    trained_model = train_attack(puf_class)

    param_pairs = [(m,alpha) for m in m_range for alpha in alpha_range]

    results = Parallel(
        n_jobs=max(1, CPU_CORES // 2),
        verbose=5
    )(
        delayed(compute_single_accuracy)(
            puf_class,
            m,
            alpha,
            trained_model
        )
        for (m,alpha) in param_pairs
    )

    acc_grid = np.array(results).reshape(len(m_range),len(alpha_range))

    return puf_name, acc_grid


# -------------------------------------------------
# Main Execution (Parallel across PUFs)
# -------------------------------------------------

if __name__ == "__main__":

    results = Parallel(
        n_jobs=min(len(puf_types), CPU_CORES),
        verbose=10
    )(
        delayed(run_parameter_sweep)(name, puf)
        for name, puf in puf_types.items()
    )

    summary_rows = {}

    for name, grid in results:

        df = pd.DataFrame(
            grid,
            index=np.round(m_range,2),
            columns=np.round(alpha_range,2)
        )

        df.to_csv(f"{name}_m_alpha_accuracy.csv")

        flat = grid.flatten()

        summary_rows[name] = dict(
            Mean=np.mean(flat),
            Std=np.std(flat),
            Min=np.min(flat),
            Max=np.max(flat)
        )

        # Heatmap
        plt.figure(figsize=(6,5))

        im = plt.imshow(
            grid,
            cmap="viridis",
            origin="lower",
            interpolation="nearest"
        )

        plt.colorbar(im,label="MLP Accuracy")

        plt.xticks(np.arange(len(alpha_range)), alpha_range)
        plt.yticks(np.arange(len(m_range)), m_range)

        plt.xlabel("Alpha")
        plt.ylabel("m")

        plt.title(f"{name} PUF\nMLP Accuracy vs Alpha-Power Parameters")

        plt.tight_layout()
        plt.savefig(f"{name}_m_alpha_heatmap.png")
        plt.close()

    summary_df = pd.DataFrame(summary_rows).T
    summary_df.to_csv("PUF_parameter_sensitivity_summary.csv")

    print("\nSummary:")
    print(summary_df)