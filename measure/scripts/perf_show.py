#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

# Constantes globales
EGM = 79.1  # gCO2/kWh
seven_years = 61320 * 3600


# Récupérer les variables d'environnement
def get_gpu_info_from_env():
    num_gpus = int(os.environ.get("BENCH_NUM_GPU", 0))
    PUE = float(os.environ.get("BENCH_PUE", 1.0))
    gpus = {}
    for gpu_id in range(num_gpus):
        prefix = f"BENCH_GPU_{gpu_id}"
        gpus[gpu_id] = {
            "name": os.environ.get(f"{prefix}_NAME"),
            "impact": float(os.environ.get(f"{prefix}_IMPACT", 0)),  # Impact de fabrication (kgCO2eq)
        }
        print(os.environ.get(f"BENCH_GPU_0_IMPACT",123456))
        print(PUE)
    return gpus, PUE

# Charger les profils de consommation pour chaque modèle, GPU et nombre d'utilisateurs
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

# Calculer l'énergie consommée (en kWh)
def compute_energy(power_profile):
    timestamps = power_profile["timestamp"]
    power = power_profile["gpu_power"]
    return np.trapz(power, timestamps) / 1000  # Conversion en kWh
def compute_duration(power_profile):
    timestamps = power_profile["timestamp"]
    duration= timestamps[len(timestamps)-1] - timestamps[0]
    return duration


# Calculer l'impact environnemental (en mgCO2eq)
def compute_impact(energy_kWh, impact_manufacturing_kg, gpu_name, PUE,duration):
    energy_kWh *= PUE
    co2_emission = energy_kWh * EGM  # Émission CO2 due à l'énergie (gCO2eq)
    soft_manufacturing_kg = (duration/seven_years)*impact_manufacturing_kg
    total_impact = co2_emission + (soft_manufacturing_kg * 1000)  # Ajouter l'impact de fabrication (mgCO2eq)
    return total_impact, co2_emission, soft_manufacturing_kg * 1000

# Tracer les profils de consommation pour chaque modèle et condition (nombre d'utilisateurs)
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

# Tracer l'impact environnemental pour chaque condition (nombre d'utilisateurs) séparément
def plot_impact_bar(impacts, gpus, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = ["mistral:7b", "gpt-oss:20b", "gemma3:12b"]

    # Définir une palette de couleurs pour les modèles
    colors = sns.color_palette("husl", n_colors=len(models))
    model_color_map = {model: colors[i] for i, model in enumerate(models)}

    # Préparer les positions des barres
    n_models = len(models)
    n_gpus = len(gpus)
    n_user_counts = len(user_counts)

    # Largeur de chaque barre
    bar_width = 0.5  # Réduire la largeur pour éviter le chevauchement

    # Espacement entre les groupes de barres (par nombre d'utilisateurs)
    group_spacing = 1.5  # Espace entre les groupes de barres

    # Créer la figure
    plt.figure(figsize=(16, 8))

    # Pour chaque nombre d'utilisateurs
    for user_idx, nb_user in enumerate(user_counts):
        # Position de départ pour ce groupe
        group_start = user_idx * (n_models * group_spacing)

        # Pour chaque modèle
        for model_idx, model in enumerate(models):
            # Position de la barre pour ce modèle
            x_pos = group_start + model_idx * group_spacing

            # Pour chaque GPU
            for gpu_idx, gpu_id in enumerate(gpus):
                gpu_name = gpus[gpu_id]["name"]
                impact_value = impacts[gpu_name][model].get(nb_user, {"total": 0})["total"]

                # Position de la barre pour ce GPU
                x = x_pos + gpu_idx * bar_width

                # Tracer la barre
                plt.bar(
                    x,
                    impact_value,
                    width=bar_width,
                    color=model_color_map[model],
                    label=f"{model}" if gpu_idx == 0 else "",
                    edgecolor="black",
                    linewidth=0.5,
                )
        if user_idx < n_user_counts - 1:
            separator_x = (group_start-1) + n_models * group_spacing - bar_width / 2
            plt.axvline(x=separator_x + bar_width, color="gray", linestyle="--", linewidth=1.5)

    # Personnaliser les ticks et les labels de l'axe des x
    x_ticks = [
        (user_idx * (n_models * group_spacing) + (n_models * group_spacing) / 2)-0.75
        for user_idx in range(n_user_counts)
    ]
    x_labels = [f"{nb_user} user{'s' if nb_user > 1 else ''}" for nb_user in user_counts]

    plt.xticks(x_ticks, x_labels, fontsize=12)
    plt.xlabel("Number of users", fontsize=14)
    plt.ylabel("Global warming potential (mgCO2eq)", fontsize=14)
    plt.title("Global warming potential by model and by number of users", fontsize=16, pad=20)

    # Ajouter une légende unique par modèle
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    plt.legend(unique_labels.values(), unique_labels.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    # Ajouter une grille et ajuster la mise en page
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Sauvegarder le graphique
    os.makedirs("images/impact_plots", exist_ok=True)
    plt.savefig("images/impact_plots/impact_by_model_and_users.png", bbox_inches="tight", dpi=300)
    plt.close()

# Tracer la proportion fabrication vs. utilisation pour chaque modèle
def plot_manufacturing_vs_usage(impacts, gpus, user_counts=[1, 10, 100], models=None):
    plt.figure(figsize=(14, 8))
    bar_width = 0.2
    x = np.arange(len(user_counts) * len(gpus) * len(models))
    for model_idx, model in enumerate(models):
        for gpu_idx, gpu_id in enumerate(gpus):
            gpu_name = gpus[gpu_id]["name"]
            for user_idx, nb_user in enumerate(user_counts):
                idx = model_idx * len(gpus) * len(user_counts) + gpu_idx * len(user_counts) + user_idx
                if nb_user in impacts[gpu_name][model]:
                    manufacturing = impacts[gpu_name][model][nb_user]["manufacturing"]
                    usage = impacts[gpu_name][model][nb_user]["usage"]
                    total = manufacturing + usage
                    plt.bar(idx, manufacturing / total * 100, width=bar_width, color='b', label="Fabrication" if idx == 0 else "")
                    plt.bar(idx, usage / total * 100, width=bar_width, bottom=manufacturing / total * 100, color='g', label="Utilisation" if idx == 0 else "")
                    plt.text(idx, 50, f"{nb_user}", ha='center', va='center')
    plt.xlabel("Model, GPU and number of users")
    plt.ylabel("Proportion (%)")
    plt.title("Proportion manufacturing vs. use phase per model")
    xticks_labels = [f"{model}\n{gpus[gpu_id]['name']} ({nb_user})" for model in models for gpu_id in gpus for nb_user in user_counts]
    plt.xticks(x + bar_width / 2, xticks_labels, rotation=45, ha='right')
    plt.gca().yaxis.set_major_formatter(PercentFormatter())
    plt.legend()
    plt.grid(True)
    os.makedirs("images/proportion_plots", exist_ok=True)
    plt.savefig("images/proportion_plots/proportion_all_models.png", bbox_inches="tight")
    plt.close()

# --- Exécution principale ---
if __name__ == "__main__":
    gpus, PUE = get_gpu_info_from_env()
    user_counts = json.loads(os.environ.get("BENCH_USERS","[1,10,100]"))

    models = ["mistral:7b", "gpt-oss:20b", "gemma3:12b"]
    power_profiles = load_power_profiles(gpus, user_counts, models)
    impacts = {gpu["name"]: {model: {} for model in models} for gpu in gpus.values()}
    for gpu_id, gpu in gpus.items():
        for model in models:
            for nb_user in user_counts:
                if nb_user in power_profiles[gpu_id][model]:
                    energy_kWh = compute_energy(power_profiles[gpu_id][model][nb_user])
                    duration = compute_duration(power_profiles[gpu_id][model][nb_user])
                    total_impact, usage_impact, manufacturing_impact = compute_impact(energy_kWh, gpu["impact"], gpu["name"], PUE, duration)
                    impacts[gpu["name"]][model][nb_user] = {
                        "total": total_impact,
                        "usage": usage_impact,
                        "manufacturing": manufacturing_impact
                    }
                
                    print(f"Impact pour {gpu['name']} ({model}, {nb_user} utilisateurs): {total_impact:.2f} mgCO2eq")
    print(impacts)
    plot_power_profiles(power_profiles, gpus, user_counts, models)
    plot_impact_bar(impacts, gpus, user_counts, models)
    plot_manufacturing_vs_usage(impacts, gpus, user_counts, models)

