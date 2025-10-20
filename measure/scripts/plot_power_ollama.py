#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- Chemins des données ---
BASE_PATH = "measure/data"
ORIN_PATH = os.path.join(BASE_PATH, "Orin_3600")
RTX_PATH = os.path.join(BASE_PATH, "L40Sx2_3600")

# --- Conditions expérimentales (nombre d’utilisateurs) ---
CONDITIONS = [1,10,100]


# --- Chargement des données Orin ---
def load_orin_data(cond):
    file_path = os.path.join(ORIN_PATH, f"consommation_energie_jetson_{cond}.csv")
    df = pd.read_csv(file_path)
    df = df.sort_values("timestamp")
    # Conversion mW → W
    df["gpu_power"] = df["gpu_power"] / 1000.0
    return df


# --- Chargement et fusion des données RTX (2 GPUs) ---
def load_rtx_data(cond):
    gpu_dfs = []
    for gpu_idx in [0, 1]:
        file_path = os.path.join(RTX_PATH, f"consommation_energie_gpu_{gpu_idx}_{cond}.csv")
        df = pd.read_csv(file_path)
        df = df.sort_values("timestamp")
        gpu_dfs.append(df)

    merged = pd.merge_asof(
        gpu_dfs[0].sort_values("timestamp"),
        gpu_dfs[1].sort_values("timestamp"),
        on="timestamp",
        suffixes=("_gpu0", "_gpu1"),
        direction="nearest"
    )
    merged["total_power"] = merged["gpu_power_gpu0"] + merged["gpu_power_gpu1"]
    return merged


# --- Chargement global pour déterminer les bornes ---
orin_dfs = {cond: load_orin_data(cond) for cond in CONDITIONS}
rtx_dfs = {cond: load_rtx_data(cond) for cond in CONDITIONS}

# Détermination des limites communes
x_min = min(df["timestamp"].min() for df in list(orin_dfs.values()) + list(rtx_dfs.values()))
x_max = max(df["timestamp"].max() for df in list(orin_dfs.values()) + list(rtx_dfs.values()))
y_min = 0
y_max = max(
    max(df["gpu_power"].max() for df in orin_dfs.values()),
    max(df["total_power"].max() for df in rtx_dfs.values())
) * 1.1  # marge 10%

# --- Création des sous-graphiques ---
fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, sharey=True)
fig.suptitle("Power consumption comparison : Jetson Orin vs 2×L40S", fontsize=14, fontweight="bold")

for i, cond in enumerate(CONDITIONS):
    # --- ORIN ---
    df_orin = orin_dfs[cond]
    ax_orin = axes[i, 0]
    ax_orin.plot(df_orin["timestamp"], df_orin["gpu_power"], color="tab:green", label=f"Jetson Orin ({cond} users)")
    ax_orin.fill_between(df_orin["timestamp"], df_orin["gpu_power"], color="tab:green", alpha=0.3)
    ax_orin.set_title(f"Orin - {cond} users")
    ax_orin.legend()
    ax_orin.grid(True, linestyle="--", alpha=0.3)

    # --- RTX8000 ---
    df_rtx = rtx_dfs[cond]
    ax_rtx = axes[i, 1]
    #ax_rtx.plot(df_rtx["timestamp"], df_rtx["total_power"], color="tab:blue", label="Total 2×RTX8000", linewidth=0.2)
    ax_rtx.fill_between(df_rtx["timestamp"], df_rtx["gpu_power_gpu0"], color="tab:orange", alpha=0.3, label="GPU 0")
    ax_rtx.fill_between(df_rtx["timestamp"],
                        df_rtx["gpu_power_gpu0"],
                        df_rtx["gpu_power_gpu0"] + df_rtx["gpu_power_gpu1"],
                        color="tab:red", alpha=0.3, label="GPU 1")
    ax_rtx.set_title(f"2×L40S - {cond} users")
    ax_rtx.legend()
    ax_rtx.grid(True, linestyle="--", alpha=0.3)

# --- Ajout de la ligne verticale à 3600 secondes avec légende ---
for ax_row in axes:
    for ax in ax_row:
        # On ajoute une ligne invisible avec un label pour la légende
        ax.plot([], [], color='gray', linestyle='--', label='End of the bench (3600s)')
        ax.axvline(x=3600, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.legend()  # Met à jour la légende pour inclure la ligne verticale
# --- Axes et mise en page ---
for ax_row in axes:
    for ax in ax_row:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("Power (W)")
for ax in axes[-1, :]:
    ax.set_xlabel("Time (s)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

