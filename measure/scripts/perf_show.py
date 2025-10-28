#!/usr/bin/env python3
import os
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# Constantes globales
EGM = 79.1  # gCO2/kWh
seven_years = 61320 * 3600

# --- Fonctions principales ---
def get_gpu_info_from_env():
    num_gpus = int(os.environ.get("BENCH_NUM_GPU", 0))
    PUE = float(os.environ.get("BENCH_PUE", 1.0))
    gpus = {}
    for gpu_id in range(num_gpus):
        prefix = f"BENCH_GPU_{gpu_id}"
        gpus[gpu_id] = {
            "name": os.environ.get(f"{prefix}_NAME"),
            "impact": float(os.environ.get(f"{prefix}_IMPACT", 0)),  # kgCO2eq
        }
    return gpus, PUE


def load_power_profiles(gpus, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = ["mistral:7b", "gpt-oss:20b", "gemma3:12b"]
    power_profiles = {gpu_id: {model: {} for model in models} for gpu_id in gpus}
    for gpu_id in gpus:
        for model in models:
            for nb_user in user_counts:
                file_path = f"/tmp/save_data/consommation_energie_gpu_{gpu_id}_{nb_user}_{model}.csv"
                if os.path.isfile(file_path):
                    data = pd.read_csv(file_path)
                    if not data.empty:
                        power_profiles[gpu_id][model][nb_user] = data
                    else:
                        print(f"Fichier vide : {file_path}")
                else:
                    print(f"Fichier introuvable : {file_path}")
    return power_profiles


def compute_energy(power_profile):
    timestamps = power_profile["timestamp"]
    power = power_profile["gpu_power"]
    mask = np.isfinite(power)
    power=power[mask]
    timestamps=timestamps[mask]
    print(f"Trapz : {np.trapz(power, timestamps)}")
    return np.trapz(power, timestamps) / 3_600_000  # kWh


def compute_duration(power_profile):
    timestamps = power_profile["timestamp"]
    return timestamps.iloc[-1] - timestamps.iloc[0]


def compute_impact(energy_kWh, impact_manufacturing_kg, PUE, duration):
    energy_kWh *= PUE
    co2_usage = energy_kWh * EGM  # gCO2eq
    soft_manufacturing_kg = (duration / seven_years) * impact_manufacturing_kg
    total_impact = co2_usage + (soft_manufacturing_kg * 1000)  # gCO2eq
    print(total_impact)
    return total_impact, co2_usage, soft_manufacturing_kg * 1000

def plot_power_profiles(power_profiles, gpus, user_counts=[1, 10, 100], models=None):

    for model in models:
        for nb_user in user_counts:
            fig, ax = plt.subplots(figsize=(10, 5))
            for gpu_id, gpu in gpus.items():
                if nb_user in power_profiles[gpu_id][model]:
                    ax.plot(
                        power_profiles[gpu_id][model][nb_user]["timestamp"],
                        power_profiles[gpu_id][model][nb_user]["gpu_power"],
                        label=f"{gpu['name']} (GPU {gpu_id})"
                    )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Power (W)")
            ax.set_title(f"Power consumption profiles - {model} ({nb_user} users)")
            ax.legend()
            ax.grid(True)
            os.makedirs("images/power_profiles", exist_ok=True)
            plt.savefig(f"images/power_profiles/power_profile_{model}_{nb_user}_users.png", bbox_inches="tight")
            plt.close()

# --- Agrégation des impacts globaux ---
def aggregate_global_impacts(impacts, models, user_counts):
    global_impacts = {model: {} for model in models}
    for model in models:
        for nb_user in user_counts:
            total = sum(
                gpu_data[model].get(nb_user, {}).get("total", 0)
                for gpu_data in impacts.values()
            )
            usage = sum(
                gpu_data[model].get(nb_user, {}).get("usage", 0)
                for gpu_data in impacts.values()
            )
            manufacturing = sum(
                gpu_data[model].get(nb_user, {}).get("manufacturing", 0)
                for gpu_data in impacts.values()
            )
            global_impacts[model][nb_user] = {
                "total": total,
                "usage": usage,
                "manufacturing": manufacturing,
            }
    print(global_impacts)
    return global_impacts

def save_global_impacts_to_csv(global_impacts,impacts, filename="measure/data/global_impacts.csv"):
    # Préparer les lignes
    rows = []

    # --- Impacts individuels (par GPU)
    for gpu_name, gpu_data in impacts.items():
        for model, model_data in gpu_data.items():
            for nb_user, impact_values in model_data.items():
                rows.append({
                    "gpu": gpu_name,
                    "model": model,
                    "nb_user": nb_user,
                    "scope": "per_gpu",
                    "total": impact_values.get("total", 0),
                    "usage": impact_values.get("usage", 0),
                    "manufacturing": impact_values.get("manufacturing", 0)
                })

    # --- Impact global (tous GPU)
    for model, user_data in global_impacts.items():
        for nb_user, impact_values in user_data.items():
            rows.append({
                "gpu": "GLOBAL",  # balise pour le total
                "model": model,
                "nb_user": nb_user,
                "scope": "global",
                "total": impact_values.get("total", 0),
                "usage": impact_values.get("usage", 0),
                "manufacturing": impact_values.get("manufacturing", 0)
            })

    # --- Écriture CSV
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["gpu", "model", "nb_user", "scope", "total", "usage", "manufacturing"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Données enregistrées dans {filename}")
# --- Plot global : impact total ---
def plot_impact_bar_global(global_impacts, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = list(global_impacts.keys())

    colors = sns.color_palette("husl", n_colors=len(models))
    model_color_map = {model: colors[i] for i, model in enumerate(models)}

    plt.figure(figsize=(14, 8))
    bar_width = 0.25
    group_spacing = 1.5

    for user_idx, nb_user in enumerate(user_counts):
        group_start = user_idx * (len(models) * group_spacing)
        for model_idx, model in enumerate(models):
            x_pos = group_start + model_idx * group_spacing
            impact_value = global_impacts[model][nb_user]["total"]
            plt.bar(
                x_pos,
                impact_value,
                width=bar_width,
                color=model_color_map[model],
                edgecolor="black",
                label=f"{model}" if user_idx == 0 else "",
            )
        if user_idx < len(user_counts) - 1:
            separator_x = (group_start-1) + len(models) * group_spacing - bar_width / 2
            plt.axvline(x=separator_x + bar_width, color="gray", linestyle="--", linewidth=1.5)

    x_ticks = [
        user_idx * (len(models) * group_spacing) + (len(models) * group_spacing) / 2 - 0.5
        for user_idx in range(len(user_counts))
    ]
    x_labels = [f"{nb_user} user{'s' if nb_user > 1 else ''}" for nb_user in user_counts]

    plt.xticks(x_ticks, x_labels)
    plt.xlabel("Number of users")
    plt.ylabel("Global warming potential (gCO2eq)")
    plt.title("Global impact (all GPUs combined)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    os.makedirs("images/impact_plots", exist_ok=True)
    plt.savefig("images/impact_plots/global_impact_by_model.png", bbox_inches="tight", dpi=300)
    plt.close()


# --- Plot global : proportion fabrication vs utilisation ---
def plot_manufacturing_vs_usage_global(global_impacts, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = list(global_impacts.keys())

    plt.figure(figsize=(14, 8))
    bar_width = 0.4
    positions = np.arange(len(models) * len(user_counts))

    for model_idx, model in enumerate(models):
        for user_idx, nb_user in enumerate(user_counts):
            idx = model_idx * len(user_counts) + user_idx
            data = global_impacts[model][nb_user]
            manufacturing = data["manufacturing"]
            usage = data["usage"]
            total = manufacturing + usage
            plt.bar(idx, manufacturing / total * 100, width=bar_width, color="b", label="Manufacturing" if idx == 0 else "")
            plt.bar(idx, usage / total * 100, width=bar_width, bottom=manufacturing / total * 100, color="g", label="Usage" if idx == 0 else "")
        

    xticks_labels = [f"{model}\n({nb_user} users)" for model in models for nb_user in user_counts]
    plt.xticks(positions, xticks_labels, rotation=45, ha="right")
    plt.ylabel("Proportion (%)")
    plt.title("Global proportion manufacturing vs usage (all GPUs combined)")
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    os.makedirs("images/proportion_plots", exist_ok=True)
    plt.savefig("images/proportion_plots/global_proportion_all_models.png", bbox_inches="tight", dpi=300)
    plt.close()


# --- Main ---
if __name__ == "__main__":
    gpus, PUE = get_gpu_info_from_env()
    user_counts = json.loads(os.environ.get("BENCH_USERS", "[1,10,100]"))
    models = ["mistral:7b", "gpt-oss:20b", "gemma3:12b"]

    power_profiles = load_power_profiles(gpus, user_counts, models)
    impacts = {gpu["name"]: {model: {} for model in models} for gpu in gpus.values()}

    # Calcul des impacts individuels
    for gpu_id, gpu in gpus.items():
        for model in models:
            for nb_user in user_counts:
                if nb_user in power_profiles[gpu_id][model]:
                    print(f"{gpu_id}:{model}:{nb_user}")
                    energy_kWh = compute_energy(power_profiles[gpu_id][model][nb_user])
                    print(f"Energy: {energy_kWh}")
                    duration = compute_duration(power_profiles[gpu_id][model][nb_user])

                    print(f"Duration: {duration}")
                    total, usage, manufacturing = compute_impact(energy_kWh, gpu["impact"], PUE, duration)
                    impacts[gpu["name"]][model][nb_user] = {
                        "total": total,
                        "usage": usage,
                        "manufacturing": manufacturing,
                    }

    # Agrégation globale
    global_impacts = aggregate_global_impacts(impacts, models, user_counts)
    save_global_impacts_to_csv(global_impacts,impacts)
    plot_power_profiles(power_profiles, gpus, user_counts, models)
    # Affichage console
    for model in models:
        for nb_user in user_counts:
            data = global_impacts[model][nb_user]
            print(f"[GLOBAL] {model} ({nb_user} users) → total: {data['total']} gCO2eq")

    # Plots globaux
    plot_impact_bar_global(global_impacts, user_counts, models)
    plot_manufacturing_vs_usage_global(global_impacts, user_counts, models)

