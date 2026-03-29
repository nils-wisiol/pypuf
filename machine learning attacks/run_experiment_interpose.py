import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from pypuf.simulation import FeedForwardArbiterPUF
from pypuf.io import ChallengeResponseSet
from pypuf.attack import MLPAttack2021
from pypuf.metrics import similarity


# -------------------------------------------------
# 1. Experiment Setup
# -------------------------------------------------

n_bits = 64
puf_seed = 123

# Fixed environment
fixed_temp = 26
fixed_vdd = 1.29

n_train = 50000
n_test = 2000

# Parameter sweep
m_range = np.array([1.2, 1.3, 1.4, 1.5, 1.6])
alpha_range = np.array([1.0, 1.1, 1.2, 1.3, 1.4])

acc_grid = np.zeros((len(m_range), len(alpha_range)))


# -------------------------------------------------
# 2. Single Experiment Function (Parallel Unit)
# -------------------------------------------------

def run_attack(i, j, m, alpha):

    print(f"Running m={m:.2f}, alpha={alpha:.2f}")

    # Create PUF
    puf = FeedForwardArbiterPUF(
        n=n_bits,
        ff=[(10, 20), (30, 40)],
        seed=puf_seed,
        noisiness=0.05,
        temperature=fixed_temp,
        vdd=fixed_vdd,
        m=m,
        alpha=alpha
    )

    # Generate CRPs
    train_crps = ChallengeResponseSet.from_simulation(
        puf,
        N=n_train,
        seed=puf_seed
    )

    # Train model
    model = MLPAttack2021(
        train_crps,
        seed=42,
        net=[n_bits, n_bits],
        epochs=50,
        lr=1e-3,
        bs=128,
        early_stop=0.01
    ).fit()

    # Test PUF
    p_test = FeedForwardArbiterPUF(
        n=n_bits,
        ff=[(10, 20), (30, 40)],
        seed=puf_seed,
        noisiness=0.05,
        temperature=fixed_temp,
        vdd=fixed_vdd,
        m=m,
        alpha=alpha
    )

    acc = similarity(p_test, model, seed=42, N=n_test)[0]

    return i, j, acc


# -------------------------------------------------
# 3. Main Execution (Required for Windows)
# -------------------------------------------------

if __name__ == "__main__":

    tasks = []

    # Use available CPU cores
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:

        for i, m in enumerate(m_range):
            for j, alpha in enumerate(alpha_range):

                tasks.append(
                    executor.submit(run_attack, i, j, m, alpha)
                )

        for future in as_completed(tasks):

            i, j, acc = future.result()
            acc_grid[i, j] = acc


    # -------------------------------------------------
    # 4. Save Results
    # -------------------------------------------------

    df = pd.DataFrame(
        acc_grid,
        index=np.round(m_range, 2),
        columns=np.round(alpha_range, 2)
    )

    df.to_csv("feedforward_m_alpha_accuracy.csv")

    print("\nAccuracy Grid:")
    print(df)


    # -------------------------------------------------
    # 5. Plot Heatmap
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        acc_grid,
        cmap="viridis",
        origin="lower",
        interpolation="nearest"
    )

    cbar = fig.colorbar(im)
    cbar.set_label("MLP Accuracy")

    ax.set_xticks(np.arange(len(alpha_range)))
    ax.set_yticks(np.arange(len(m_range)))

    ax.set_xticklabels(alpha_range)
    ax.set_yticklabels(m_range)

    ax.set_xlabel("Alpha")
    ax.set_ylabel("m")

    ax.set_title("FeedForward Arbiter PUF - MLP Attack Accuracy vs Alpha–Power Law Parameters")

    plt.tight_layout()
    plt.show()