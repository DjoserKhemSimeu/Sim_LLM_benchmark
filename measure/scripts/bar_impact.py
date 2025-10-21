import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from scipy.stats import pearsonr

# Récupérer les informations des GPU depuis les variables d'environnement
def get_gpu_info_from_env():
    num_gpus = int(os.environ.get("BENCH_NUM_GPU"))
    print(f"Num GPUs {num_gpus}")
    gpus = {}
    for gpu_id in range(num_gpus):
        prefix = f"BENCH_GPU_{gpu_id}"
        gpus[gpu_id] = {
            "name": os.environ.get(f"{prefix}_NAME"),
            "die_area": float(os.environ.get(f"{prefix}_DIE_AREA", 0)),
            "tdp": float(os.environ.get(f"{prefix}_TDP", 0)),
            "tech_node": os.environ.get(f"{prefix}_TECH_NODE"),
            "mem_type": os.environ.get(f"{prefix}_MEM_TYPE"),
            "mem_size": float(os.environ.get(f"{prefix}_MEM_SIZE", 0)),
            "foundry": os.environ.get(f"{prefix}_FOUNDRY"),
            "date": int(os.environ.get(f"{prefix}_RELEASE_DATE", 0)),
            "fu": os.environ.get(f"{prefix}_FU"),
        }
    return gpus
def main_impact():
# Charger les données de densité mémoire
    df_mem = pd.read_csv('data/mem_density.csv', delimiter=';')
    mem_data = df_mem.set_index('name')['gCO2/GB'].to_dict()

# Charger les données GWP
    xls = pd.ExcelFile('data/Environmental-Footprint-ICs.xlsx')

    df_gwp = xls.parse('GWP', header=3)
    df_gwp = df_gwp.iloc[:, 1:]
    df_gwp['iN\n[nm/mix]'] = pd.to_numeric(df_gwp['iN\n[nm/mix]'], errors='coerce')
    df_gwp["Year"] = pd.to_numeric(df_gwp["Year"], errors='coerce')
    df_gwp["Value"] = pd.to_numeric(df_gwp["Value"], errors='coerce')
    print(df_gwp)
# Récupérer les informations des GPU
    gpus = get_gpu_info_from_env()
    print(f"GPUs: {gpus}")
# Préparer les données pour le calcul d'impact
    hardware_dict = {gpu["name"]: [] for gpu in gpus.values()}

    for _, row in df_gwp.iterrows():
        category = row["Category"]
        value = row["Value"]
        if pd.isna(value):
            continue
        if category in ["Literature", "Database"]:
            in_value = row['iN\n[nm/mix]']
            if not pd.isna(in_value):
                for gpu in gpus.values():
                    if gpu["tech_node"] == in_value:
                        hardware_dict[gpu["name"]].append(value)
        elif category in ["Industry", "Roadmapping"]:
            year_value = row["Year"]
            if not pd.isna(year_value):
                for gpu in gpus.values():
                    if gpu["date"] == year_value:
                        if category == "Industry" and gpu["foundry"] == row["Authors"]:
                            hardware_dict[gpu["name"]].append(value)
                        elif category == "Roadmapping":
                            hardware_dict[gpu["name"]].append(value)
    print(f"Hardware : {hardware_dict}")
# Calculer l'impact environnemental
    all_impact_values = []
    hardware_names = []
    hardware_fu = []
    hardware_dates = []

    for gpu_id, gpu in gpus.items():
        name = gpu["name"]
        if name in hardware_dict:
            die_area_value = gpu["die_area"]
            mem_name = gpu["mem_type"]
            mem_size = gpu["mem_size"]
            impact_values = [(val * die_area_value * 0.01) + ((mem_size * mem_data[mem_name]) / 1000) for val in hardware_dict[name]]
            all_impact_values.extend(impact_values)
            hardware_names.extend([name] * len(impact_values))
            hardware_fu.extend([gpu["fu"]] * len(impact_values))
            hardware_dates.extend([gpu["date"]] * len(impact_values))

# Créer un DataFrame pour le boxplot
    df_boxplot = pd.DataFrame({
        "Hardware": hardware_names,
        "Impact Environnemental": all_impact_values,
        "FU": hardware_fu,
        "Date": hardware_dates
    })
    print(df_boxplot)

# Calculer la corrélation
#correlation, p_value = pearsonr(df_boxplot['Date'], df_boxplot['Impact Environnemental'])
#print(f"Coefficient de corrélation de Pearson: {correlation}")
#print(f"P-value: {p_value}")

# Interprétation
#if p_value < 0.05:
        #print("La corrélation est statistiquement significative.")
#else:
        #print("La corrélation n'est pas statistiquement significative.")

# Définir l'ordre des catégories FU
    fu_order = ["Edge", "Gaming", "Desktop", "Large-scale"]

# Trier le DataFrame selon l'ordre des catégories FU
    df_boxplot['FU'] = pd.Categorical(df_boxplot['FU'], categories=fu_order, ordered=False)
    df_boxplot = df_boxplot.sort_values('FU')

# Créer une palette de couleurs basée sur les dates de sortie
    norm = plt.Normalize(df_boxplot['Date'].min(), df_boxplot['Date'].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    df_boxplot['color'] = df_boxplot['Date'].apply(lambda x: sm.to_rgba(x))

# Créer le boxplot
    plt.figure(figsize=(16, 10))
    unique_hardware = df_boxplot["Hardware"].unique()
    for i, hardware in enumerate(unique_hardware):
        subset = df_boxplot[df_boxplot["Hardware"] == hardware]
        color = subset['color'].iloc[0]
        sns.boxplot(x=[hardware] * len(subset), y=subset["Impact Environnemental"], hue=[hardware] * len(subset), palette=[color], dodge=False, legend=False)

# Ajouter des barres verticales en pointillé pour séparer les catégories
    current_x = 0
    mids = [0.5, 2.5, 6.5, 13]
    for idx, fu_category in enumerate(fu_order):
        subset = df_boxplot[df_boxplot["FU"] == fu_category]
        unique_hardware = subset["Hardware"].unique()
        plt.axvline(x=current_x + len(unique_hardware) - 0.5, color='black', linestyle='--')
        plt.text(mids[idx], max(df_boxplot["Impact Environnemental"]), fu_category, horizontalalignment='center', fontsize=15, color='black')
        current_x += len(unique_hardware)

# Ajouter une barre de couleur
    cbar = plt.colorbar(sm, ax=plt.gca(), pad=0.02, fraction=0.02)
    cbar.set_label('Date', rotation=270, labelpad=15)
    plt.ylabel('Manufacturing impact (kgCO2 eq)', fontsize=14)
    plt.title('Manufacturing impact by Hardware, grouped by category', fontsize=16)
    plt.xticks(rotation=90, fontsize=14)
    plt.tight_layout()

# Sauvegarder le graphique
    plt.savefig("manufacturing_impact_by_hardware.png")

# Sauvegarder l'impact de chaque GPU dans des variables d'environnement
    for gpu_id, gpu in gpus.items():
        name = gpu["name"]
        if name in hardware_dict:
            die_area_value = gpu["die_area"]
            mem_name = gpu["mem_type"]
            mem_size = gpu["mem_size"]
            impact_values = [(val * die_area_value * 0.01) + ((mem_size * mem_data[mem_name]) / 1000) for val in hardware_dict[name]]
            avg_impact = sum(impact_values) / len(impact_values)
            os.environ[f"BENCH_GPU_{gpu_id}_IMPACT"] = str(avg_impact)
            print(f"Variable d'environnement définie : BENCH_GPU_{gpu_id}_IMPACT = {os.environ.get(f'BENCH_GPU_{gpu_id}_IMPACT')}")
            print(f"The manufatucring impact of the GPU {gpu_id} is : {avg_impact}")

    print("Graphique sauvegardé et impacts enregistrés dans les variables d'environnement.")

