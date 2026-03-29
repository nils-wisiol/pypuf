import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

# ==============================
# 1. LOAD DATA
# ==============================

def load_data(pattern):
    files = glob.glob(pattern)
    data_list = []

    for f in files:
        df = pd.read_csv(f)
        data_list.append(df)

    return pd.concat(data_list, ignore_index=True)


arbiter_data = load_data("Arbiter PUF*.csv")
ff_data = load_data("Feed-Forward Arbiter*.csv")

datasets = [
    arbiter_data,
    ff_data,
]

titles = [
    "Arbiter PUF",
    "Feed-Forward Arbiter PUF",
]

# ==============================
# 2. CREATE GLOBAL TEMP/VOLT GRID
# ==============================

temps_global = sorted(
    set().union(*[set(df["Temperature"]) for df in datasets])
)

volts_global = sorted(
    set().union(*[set(df["Voltage"]) for df in datasets])
)

# ==============================
# 3. BUILD HEATMAP GRIDS
# ==============================

heatmaps = []

for df in datasets:
    grid = np.full((len(temps_global), len(volts_global)), np.nan)

    for _, row in df.iterrows():
        i = temps_global.index(row["Temperature"])
        j = volts_global.index(row["Voltage"])
        grid[i, j] = row["Accuracy"]

    heatmaps.append(grid)

# ==============================
# 4. PLOTTING
# ==============================

cmap = plt.get_cmap("inferno")

fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True, sharey=True, constrained_layout=True)

for i, ax in enumerate(axes):
    im = ax.imshow(
        heatmaps[i],
        cmap=cmap,
        origin="lower",
        aspect="auto"
    )

    ax.set_title(titles[i])

# ==============================
# 5. AXIS TICKS
# ==============================

temp_ticks = np.arange(0, len(temps_global), max(1, len(temps_global)//6))
volt_ticks = np.arange(0, len(volts_global), max(1, len(volts_global)//6))

for ax in axes:
    ax.set_xticks(volt_ticks)
    ax.set_yticks(temp_ticks)
    ax.set_xticklabels([round(volts_global[k], 2) for k in volt_ticks])
    ax.set_yticklabels([temps_global[k] for k in temp_ticks])

axes[0].set_ylabel("Temperature (°C)")
axes[0].set_xlabel("Voltage (V)")
axes[1].set_xlabel("Voltage (V)")

# ==============================
# 6. COLORBAR
# ==============================

cbar = fig.colorbar(im, ax=axes, shrink=0.7)
cbar.set_label("MLP Attack Accuracy")

plt.show()