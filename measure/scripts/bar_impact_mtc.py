import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from math import log10

# --- Paramètres A100 de Référence (tirés de l'étude ou supposés réalistes) ---
# Ces valeurs sont nécessaires pour calculer F_tot,A100
A100_REF = {
    "die_area": 826,  # mm²
    "tech_node": 7.0,  # nm
    "mem_size": 40.0,  # GB (pour la version HBM2e standard)
    "tdp": 400.0,  # W
    "density": 65600000,  # Circuits Intégrés/cm² (Valeur illustrative)
}
TARGET_CATEGORIES = [
    "GWP - Climate change",
    "ADPf - Resource use, fossils",
    "WU - Water use",
]

# --- Fonctions de chargement et de récupération (modifiées pour inclure densité/mem_tech_node) ---


def get_gpu_info_from_env():
    # ... (fonction légèrement modifiée pour gérer proc_density et mem_tech_node)
    num_gpus = int(os.environ.get("BENCH_NUM_GPU", 0))
    print(f"Num GPUs {num_gpus}")
    gpus = {}
    for gpu_id in range(num_gpus):
        prefix = f"BENCH_GPU_{gpu_id}"
        gpus[gpu_id] = {
            "name": os.environ.get(f"{prefix}_NAME"),
            "die_area": float(os.environ.get(f"{prefix}_DIE_AREA", 0)),  # proc_die_area
            "tdp": float(os.environ.get(f"{prefix}_TDP", 0)),
            "proc_tech_node": float(os.environ.get(f"{prefix}_TECH_NODE", 0)),
            "mem_type": os.environ.get(f"{prefix}_MEM_TYPE"),
            "mem_size": float(os.environ.get(f"{prefix}_MEM_SIZE", 0)),
            # NOUVEAUX PARAMÈTRES NÉCESSAIRES :
            "density": float(os.environ.get(f"{prefix}_DENSITY", A100_REF["density"])),
            "foundry": os.environ.get(f"{prefix}_FOUNDRY"),
            "date": int(os.environ.get(f"{prefix}_RELEASE_DATE", 0)),
            "fu": os.environ.get(f"{prefix}_FU"),
        }
    return gpus


def load_a100_impact_data(csv_path="./data/more-than-carbon-data.csv"):
    """Charge et prépare les impacts bruts de la A100 par composant."""
    try:
        df_a100 = pd.read_csv(csv_path, sep=";")
    except FileNotFoundError:
        print(f"Erreur: Le fichier {csv_path} est introuvable.")
        return None, None

    # Garder les impacts bruts pour les colonnes Main dies et Heatsink
    a100_impacts_raw = df_a100.set_index("Impact Category")[
        ["Main dies", "Heatsink"]
    ].to_dict("index")
    all_categories = df_a100["Impact Category"].tolist()

    print("Impacts bruts Main dies et Heatsink de l'A100 chargés.")
    return a100_impacts_raw, all_categories


# --- Fonction Principale avec la nouvelle logique de proportionalité ---


def calculate_proportionality_factor(gpu, ref=A100_REF):
    """Calcule F_GPU_chip et F_heatsink et F_tot pour un GPU donné."""

    # Sécurité pour éviter la division par zéro
    # if gpu["proc_tech_node"] == 0 or gpu["mem_tech_node"] == 0:
    #    return 0, 0, 0

    # 1. Calcul de gamma (γ)
    # gamma = gpu["mem_tech_node"] / gpu["proc_tech_node"]
    g1 = 0.7
    g2 = 0.3
    # 2. Calcul du Facteur F_GPU_chip
    # F_GPU_chip ∝ γ(proc_die_area × proc_density) + (mem_size × mem_density) / mem_tech_node

    term1_chip = g1 * ((gpu["die_area"] * gpu["density"]) / (826 * 65600000))
    term2_chip = g2 * (gpu["mem_size"]) / 80
    F_GPU_chip = term1_chip + term2_chip

    # 3. Calcul du Facteur F_heatsink
    # F_heatsink ∝ GPU_TDP
    F_heatsink = gpu["tdp"]

    # 4. Facteur Total F_tot
    F_tot = F_GPU_chip + F_heatsink

    return F_GPU_chip, F_heatsink, F_tot


def main_impact_mtc():
    # --- Chargement des données A100 de référence ---
    a100_impacts_raw, all_categories = load_a100_impact_data()
    if not a100_impacts_raw:
        return

    gpus = get_gpu_info_from_env()

    # --- 1. Calcul du Facteur Total de Proportionalité pour l'A100 de Référence ---
    # Utilisez les caractéristiques fixes de l'A100
    F_GPU_chip_A100, F_heatsink_A100, F_tot_A100 = calculate_proportionality_factor(
        A100_REF
    )

    if F_tot_A100 == 0:
        print(
            "Erreur: Le facteur de proportionalité de l'A100 de référence est zéro. Vérifiez A100_REF."
        )
        return

    print(f"\nFacteur Total de Proportionalité A100 (F_tot,A100) : {F_tot_A100:.2e}")

    # --- 2. Calcul des Impacts pour TOUS les GPUs ---

    summary_data = []

    for gpu_id, gpu in gpus.items():
        # 2.1. Calculer F_tot pour le GPU actuel
        F_GPU_chip, F_heatsink, F_tot_GPU = calculate_proportionality_factor(gpu)

        if F_tot_GPU == 0:
            print(
                f"Avertissement: Facteur F_tot nul pour le GPU {gpu['name']}. Skipping."
            )
            continue

        total_impacts = {"Hardware": gpu["name"], "FU": gpu["fu"], "F_tot": F_tot_GPU}

        for cat in all_categories:
            cat_short = cat.split(" - ")[0]

            # 2.2. Calculer l'impact total de l'A100 (Main dies + Heatsink) pour cette catégorie
            impact_A100_total = (
                a100_impacts_raw[cat]["Main dies"] + a100_impacts_raw[cat]["Heatsink"]
            )

            # 2.3. Estimer l'impact du GPU actuel par produit en croix (Proportionalité)
            # Impact_GPU = (Impact_A100_total * F_tot_GPU) / F_tot_A100
            impact_GPU_chip_estimated = (impact_A100_total * F_GPU_chip) / F_tot_A100
            impact_GPU_heatsink_estimated = (
                impact_A100_total * F_heatsink
            ) / F_tot_A100
            impact_GPU_estimated = (impact_A100_total * F_tot_GPU) / F_tot_A100

            total_impacts[cat_short] = impact_GPU_estimated

            # Sauvegarde des variables d'environnement (y compris pour GWP, ADPf, WU)
            env_var_name = f"BENCH_GPU_{gpu_id}_IMPACT_{cat_short.replace(' ', '_')}"
            env_var_name_heat = (
                f"BENCH_GPU_{gpu_id}_IMPACT_{cat_short.replace(' ', '_')}_HEAT"
            )
            env_var_name_chip = (
                f"BENCH_GPU_{gpu_id}_IMPACT_{cat_short.replace(' ', '_')}_CHIP"
            )

            os.environ[env_var_name] = str(impact_GPU_estimated)
            os.environ[env_var_name_heat] = str(impact_GPU_heatsink_estimated)
            os.environ[env_var_name_chip] = str(impact_GPU_chip_estimated)

        summary_data.append(total_impacts)

    print("\n--- ✅ Calculs d'Impact Terminés (Nouvelle Proportionalité) ---")

    # --- 3. Affichage Récapitulatif (Concentration sur GWP, ADPf, WU) ---

    df_summary = pd.DataFrame(summary_data)

    # Identifier les colonnes d'impact qui nous intéressent + F_tot
    impact_cols_short = [c.split(" - ")[0] for c in TARGET_CATEGORIES]
    display_cols = ["Hardware", "FU", "F_tot"] + impact_cols_short

    # Assurez-vous que les colonnes existent avant de les afficher
    display_cols = [col for col in display_cols if col in df_summary.columns]
    df_display = df_summary[display_cols]

    # Arrondir pour une meilleure lisibilité
    for col in impact_cols_short + ["F_tot"]:
        if col in df_display.columns:
            df_display[col] = df_display[col].round(4)

    print(
        "\n## 📋 Récapitulatif des Impacts de Co-conception (Méthode de Proportionalité)"
    )
    print(
        "> Les valeurs sont estimées par produit en croix basé sur F_tot et l'impact A100 (Main dies + Heatsink)."
    )
    print(df_display.to_markdown(index=False))

    # --- 4. Création du Heatmap (pour tous les 16 facteurs, comme demandé précédemment) ---

    # Identifier les colonnes d'impact pour la normalisation (tous les 16)
    all_impact_cols_short = [c.split(" - ")[0] for c in all_categories]

    # Normalisation et Heatmap (comme dans la réponse précédente)
    df_normalized = df_summary.copy()

    scaler = MinMaxScaler()
    df_normalized[all_impact_cols_short] = scaler.fit_transform(
        df_normalized[all_impact_cols_short]
    )

    fu_order = ["Edge", "Gaming", "Desktop", "Large-scale"]
    df_normalized["FU"] = pd.Categorical(
        df_normalized["FU"], categories=fu_order, ordered=True
    )
    df_normalized = df_normalized.sort_values(["FU", "Hardware"])

    heatmap_data = df_normalized.set_index("Hardware")[all_impact_cols_short]

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        heatmap_data,
        cmap="viridis",
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Impact Relatif (Normalisé 0-1)", "pad": 0.02},
    )

    plt.title(
        "Impact de Fabrication Relatif des GPUs (16 Catégories) - Méthode Proportionalité",
        fontsize=16,
    )
    plt.xlabel("Catégories d'Impact", fontsize=14)
    plt.ylabel("Hardware (Trié par Catégorie d'Usage)", fontsize=14)

    current_idx = 0
    for fu in fu_order:
        count = df_normalized[df_normalized["FU"] == fu].shape[0]
        if count > 0 and current_idx > 0:
            plt.axhline(current_idx, color="red", linestyle="-", linewidth=2)

        if count > 0:
            plt.text(
                heatmap_data.shape[1] + 0.5,
                current_idx + count / 2,
                fu,
                verticalalignment="center",
                horizontalalignment="left",
                fontsize=10,
                color="black",
            )
        current_idx += count

    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig("manufacturing_impact_heatmap_16_categories_proportionality.png")
    print(
        "\n✅ Heatmap sauvegardé avec la méthode de proportionalité : 'manufacturing_impact_heatmap_16_categories_proportionality.png'"
    )
