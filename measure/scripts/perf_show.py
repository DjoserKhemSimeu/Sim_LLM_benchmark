#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

# Constantes globales
EGM = 79.1  # gCO2/kWh

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
    return gpus, PUE

# Charger les profils de consommation pour chaque modèle, GPU et nombre d'utilisateurs
def load_power_profiles(gpus, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = ["mistral:7b","gpt-oss:20b","gemma3:12b"]
  # Remplacez par vos modèles réels
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

# Calculer l'impact environnemental (en mgCO2eq)
def compute_impact(energy_kWh, impact_manufacturing_kg, gpu_name, PUE):
    large_scale_gpus = ["A100", "H100", "L40S", "A40"]
    if any(name in gpu_name for name in large_scale_gpus):
        energy_kWh *= PUE
    co2_emission = energy_kWh * EGM  # Émission CO2 due à l'énergie (gCO2eq)
    total_impact = co2_emission + (impact_manufacturing_kg * 1000)  # Ajouter l'impact de fabrication (mgCO2eq)
    return total_impact, co2_emission, impact_manufacturing_kg * 1000  # Retourne aussi les composantes

# Tracer les profils de consommation pour chaque modèle
def plot_power_profiles(power_profiles, gpus, user_counts=[1, 10, 100], models=None):
    for gpu_id, gpu in gpus.items():
        for model in models:
            for nb_user in user_counts:
                if nb_user in power_profiles[gpu_id][model]:
                    plt.figure(figsize=(10, 5))
                    plt.plot(power_profiles[gpu_id][model][nb_user]["timestamp"],
                             power_profiles[gpu_id][model][nb_user]["gpu_power"],
                             label=f"{gpu['name']} ({nb_user} utilisateurs, {model})")
                    plt.xlabel("Temps (s)")
                    plt.ylabel("Puissance (W)")
                    plt.title(f"Profil de consommation - {gpu['name']} ({model}, {nb_user} utilisateurs)")
                    plt.legend()
                    plt.grid(True)
                    os.makedirs("images/power_profiles", exist_ok=True)
                    plt.savefig(f"images/power_profiles/power_profile_{gpu['name']}_{model}_{nb_user}_users.png")
                    plt.close()

# Tracer l'impact environnemental pour tous les modèles (un seul graphique)
def plot_impact_bar(impacts, gpus, user_counts=[1, 10, 100], models=None):
    plt.figure(figsize=(14, 8))
    bar_width = 0.2
    x = np.arange(len(user_counts))

    # Créer une palette de couleurs pour les modèles
    colors = sns.color_palette("husl", len(models))

    for model_idx, model in enumerate(models):
        for gpu_idx, gpu_id in enumerate(gpus):
            gpu_name = gpus[gpu_id]["name"]
            impact_values = [impacts[gpu_name][model].get(nb_user, {"total": 0})["total"] for nb_user in user_counts]
            offset = model_idx * len(gpus) * bar_width + gpu_idx * bar_width
            plt.bar(x + offset, impact_values, width=bar_width, label=f"{gpu_name} ({model})", color=colors[model_idx])

    plt.xlabel("Nombre d'utilisateurs")
    plt.ylabel("Impact environnemental (mgCO2eq)")
    plt.title("Impact environnemental par modèle et nombre d'utilisateurs")
    plt.xticks(x + (len(models) * len(gpus) * bar_width) / 2, [str(nb_user) for nb_user in user_counts])
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True)
    os.makedirs("images/impact_plots", exist_ok=True)
    plt.savefig("images/impact_plots/impact_vs_users_all_models.png", bbox_inches="tight")
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

    plt.xlabel("Modèle, GPU et nombre d'utilisateurs")
    plt.ylabel("Proportion (%)")
    plt.title("Proportion fabrication vs. utilisation par modèle")
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
    # Récupérer les infos des GPU et le PUE depuis les variables d'environnement
    gpus, PUE = get_gpu_info_from_env()
    user_counts = [1, 10, 100]
    models = ["mistral:7b","gpt-oss:20b","gemma3:12b"]
  # Remplacez par vos modèles réels

    # Charger les profils de consommation
    power_profiles = load_power_profiles(gpus, user_counts, models)

    # Calculer l'impact environnemental pour chaque GPU, modèle et nombre d'utilisateurs
    impacts = {gpu["name"]: {model: {} for model in models} for gpu in gpus.values()}
    for gpu_id, gpu in gpus.items():
        for model in models:
            for nb_user in user_counts:
                if nb_user in power_profiles[gpu_id][model]:
                    energy_kWh = compute_energy(power_profiles[gpu_id][model][nb_user])
                    total_impact, usage_impact, manufacturing_impact = compute_impact(energy_kWh, gpu["impact"], gpu["name"], PUE)
                    impacts[gpu["name"]][model][nb_user] = {
                        "total": total_impact,
                        "usage": usage_impact,
                        "manufacturing": manufacturing_impact
                    }
                    print(f"Impact pour {gpu['name']} ({model}, {nb_user} utilisateurs): {total_impact:.2f} mgCO2eq")

    # Tracer les profils de consommation pour chaque modèle
    plot_power_profiles(power_profiles, gpus, user_counts, models)

    # Tracer l'impact environnemental pour tous les modèles (un seul graphique)
    plot_impact_bar(impacts, gpus, user_counts, models)

    # Tracer la proportion fabrication vs. utilisation pour chaque modèle
    plot_manufacturing_vs_usage(impacts, gpus, user_counts, models)

