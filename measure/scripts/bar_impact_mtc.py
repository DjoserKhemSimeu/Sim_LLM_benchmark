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
    g1 = 0.7 # Poids pour la surface du die
    g2 = 0.3 # Poids pour la taille de la mémoire
    # 2. Calcul du Facteur F_GPU_chip
    # F_GPU_chip ∝ γ(proc_die_area × proc_density) + (mem_size × mem_density) / mem_tech_node

    term1_chip = g1 * ((gpu["die_area"] * gpu["density"]) / (ref["die_area"] * ref["density"])) # Normalisation par rapport à l'A100
    term2_chip = g2 * (gpu["mem_size"] / ref["mem_size"]) # Normalisation par rapport à l'A100
    F_GPU_chip = term1_chip + term2_chip

    # 3. Calcul du Facteur F_heatsink
    # F_heatsink ∝ GPU_TDP
    F_heatsink = gpu["tdp"]/ref["tdp"]
    # 4. Facteur Total F_tot
    F_tot = F_GPU_chip + F_heatsink

    return F_GPU_chip, F_heatsink, F_tot


def main_impact_mtc():
    # --- Chargement des données A100 de référence ---
    a100_impacts_raw, all_categories = load_a100_impact_data()
    if not a100_impacts_raw:
        return

    gpus = get_gpu_info_from_env()


    # --- 1. Calcul des Impacts pour TOUS les GPUs ---

    summary_data = []

    for gpu_id, gpu in gpus.items():
        # 2.1. Calculer F_tot pour le GPU actuel
        F_GPU_chip, F_heatsink, F_tot_GPU = calculate_proportionality_factor(gpu)
        print(
            f"\nCalcul des impacts pour le GPU {gpu['name']} (ID: {gpu_id}) avec F_tot = {F_tot_GPU:.2e}"
        )
        if F_tot_GPU == 0:
            print(
                f"Avertissement: Facteur F_tot nul pour le GPU {gpu['name']}. Skipping."
            )
            continue

        total_impacts = {"Hardware": f"{gpu['name']}_{gpu_id}", "FU": gpu["fu"], "F_tot": F_tot_GPU}

        for cat in all_categories:
            cat_short = cat.split(" - ")[0]

            # 2.2. Calculer l'impact total de l'A100 (Main dies + Heatsink) pour cette catégorie
            impact_A100_total = (
                a100_impacts_raw[cat]["Main dies"] + a100_impacts_raw[cat]["Heatsink"]
            )

            # 2.3. Estimer l'impact du GPU actuel par produit en croix (Proportionalité)
            # Impact_GPU = (Impact_A100_total * F_tot_GPU) / F_tot_A100
            impact_GPU_chip_estimated = a100_impacts_raw[cat]["Main dies"] * F_GPU_chip
            impact_GPU_heatsink_estimated = a100_impacts_raw[cat]["Heatsink"] * F_heatsink
            impact_GPU_estimated = impact_GPU_chip_estimated + impact_GPU_heatsink_estimated

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
    df_summary.to_csv("data/manufacturing_impact_summary_mtc.csv", index=False)

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

    # --- 4. Création d'un bar plot 100% empilé pour Main dies vs Heatsink ---
    # Pour chaque catégorie d'impact, on somme la contribution estimée
    # des GPUs pour Main dies et Heatsink séparément, puis on trace
    # une barre empilée normalisée à 100% montrant les parts relatives.

    if len(gpus) == 0:
        print("Aucun GPU détecté (BENCH_NUM_GPU=0) — pas de graphique empilé généré.")
    else:
        cats_short = [c.split(" - ")[0] for c in all_categories]
        sum_main = {cs: 0.0 for cs in cats_short}
        sum_heat = {cs: 0.0 for cs in cats_short}

        # Somme des contributions Main dies et Heatsink sur tous les GPUs
        for gpu_id, gpu in gpus.items():
            F_GPU_chip, F_heatsink, F_tot_GPU = calculate_proportionality_factor(gpu)
            if F_tot_GPU == 0:
                continue
            for cat in all_categories:
                cs = cat.split(" - ")[0]
                main_val = a100_impacts_raw[cat]["Main dies"] * F_GPU_chip
                heat_val = a100_impacts_raw[cat]["Heatsink"] * F_heatsink
                sum_main[cs] += main_val
                sum_heat[cs] += heat_val

        df_comp = pd.DataFrame({"Main dies": sum_main, "Heatsink": sum_heat})

        # Calculer pourcentages normalisés à 100% par catégorie
        df_pct = df_comp.div(df_comp.sum(axis=1), axis=0).fillna(0) * 100

        # Tracé empilé vertical par catégorie (Main dies en bas, Heatsink au-dessus)
        fig, ax = plt.subplots(figsize=(16, 6))
        x = range(len(df_pct))
        colors = ["#8dd3c7", "#fb8072"]  # choix simple : vert clair / rouge clair

        ax.bar(x, df_pct["Main dies"].values, label="Main dies (GPU chip)", color=colors[0], edgecolor="white")
        ax.bar(x, df_pct["Heatsink"].values, bottom=df_pct["Main dies"].values, label="Heatsink", color=colors[1], edgecolor="white")

        # Annoter pour chaque catégorie : total absolu (Main+Heatsink) avec unité
        try:
            unit_map = pd.read_csv("./data/more-than-carbon-data.csv", sep=";").set_index("Impact Category")["Unit"].to_dict()
        except Exception:
            unit_map = {cat: "" for cat in all_categories}

        totals_abs = (df_comp.sum(axis=1)).values
        texts = []
        for i, cat in enumerate(all_categories):
            short = cat.split(" - ")[0]
            unit = unit_map.get(cat, "")
            tot = totals_abs[i]
            y_center = df_pct.iloc[i].sum() / 2.0  # centre en % (ex: 50)
            # On place provisoirement le texte avec va='bottom' (on ajustera ensuite)
            t = ax.text(
                i,
                y_center,
                f"{tot:.3g} {unit}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90,
                fontweight="bold",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
            )
            texts.append((t, y_center))

        # Forcer le rendu pour que get_window_extent() renvoie de vraies dimensions
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()

        # Ajuster verticalement chaque texte pour qu'il soit centré (adapté à sa hauteur)
        for t, y_center in texts:
            bbox_disp = t.get_window_extent(renderer=renderer)          # bbox en pixels
            bbox_data = inv.transform_bbox(bbox_disp)                  # bbox en coordonnées de données
            height_data = bbox_data.height                             # hauteur en unités y de l'axe
            # Comme on a utilisé va='bottom', pour centrer on place le bas à y_center - height/2
            t.set_y(y_center - height_data / 2.0)

        ax.set_xticks(x)
        ax.set_xticklabels(df_pct.index, rotation=90)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Contribution (%) — normalisé à 100% par catégorie")
        ax.set_title("Répartition Main dies vs Heatsink par catégorie d'impact (somme des GPUs)")
        ax.legend()
        plt.tight_layout()

        outname = "manufacturing_impact_main_vs_heatsink_percent.png"
        plt.savefig(outname, dpi=200)
        print(f"\n✅ Bar plot 100% empilé sauvegardé : '{outname}'")
