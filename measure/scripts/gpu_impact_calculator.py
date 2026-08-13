import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Spécifications de Référence pour les GPUs de Base ---
# Il faut impérativement renseigner les caractéristiques physiques (die_area, tdp, etc.)
# pour chaque GPU présent dans "more-than-carbon-data.csv" qui servira de base.
# Remarque : Les noms ici doivent correspondre à ceux de la colonne GPU_Name du CSV.
BASE_GPUS_SPECS = {
    "A100 SXM4 40GB": {
        "die_area": 826.0,      
        "tdp": 400.0,           
        "density": 65600000.0,  
        "mem_size": 40.0        
    },
    "A100 PCIe 40GB": {
        "die_area": 826.0,      
        "tdp": 250.0,           
        "density": 65600000.0,  
        "mem_size": 40.0        
    },
    "GH200": { 
        "die_area": 814.0, 
        "tdp": 700.0,
        "density": 98300000.0,
        "mem_size": 96.0
    },
    "Nvidia P100": {
        "die_area": 610.0,  
        "tdp": 250.0,       
        "density": 25100000.0, 
        "mem_size": 16.0    
    },
    "RTXA4500": {
        "die_area": 628.0,
        "tdp": 200.0,
        "density": 45100000.0,
        "mem_size": 20.0
    },
    "Titan RTX": {
        "die_area": 754.0,
        "tdp": 280.0,
        "density": 24700000.0,
        "mem_size": 24.0
    },
    "GEFORCE GTX 1080 Ti": {
        "die_area": 471.0,
        "tdp": 250.0,
        "density": 25100000.0,
        "mem_size": 11.0
    },
    "Nvidia L4": {
        "die_area": 295.0,
        "tdp": 72.0,
        "density": 121000000.0,
        "mem_size": 24.0
    }
}

TARGET_CATEGORIES = [
    "GWP - Climate change",
    "ADPe - Resource use, minerals and metals",
    "WU - Water use",
]

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_gpu_info_from_env():
    num_gpus_str = os.environ.get("BENCH_NUM_GPU", "0")
    print(f"[DEBUG MTC] Valeur brute de BENCH_NUM_GPU lue: '{num_gpus_str}'")
    
    try:
        num_gpus = int(num_gpus_str)
    except ValueError:
        print(f"[ERREUR MTC] BENCH_NUM_GPU n'est pas un entier valide : {num_gpus_str}")
        return {}
        
    gpus = {}
    for gpu_id in range(num_gpus):
        prefix = f"BENCH_GPU_{gpu_id}"
        gpus[gpu_id] = {
            "name": os.environ.get(f"{prefix}_NAME", f"GPU_{gpu_id}"),
            "die_area": float(os.environ.get(f"{prefix}_DIE_AREA", 0)), 
            "tdp": float(os.environ.get(f"{prefix}_TDP", 0)),
            "mem_size": float(os.environ.get(f"{prefix}_MEM_SIZE", 0)),
            "density": float(os.environ.get(f"{prefix}_DENSITY", BASE_GPUS_SPECS["A100 SXM4 40GB"]["density"])),
            "fu": os.environ.get(f"{prefix}_FU", ""),
        }
        print(f"[DEBUG MTC] Paramètres lus pour {prefix}: {gpus[gpu_id]}")
    return gpus

def load_base_gpus_impacts():
    """Cherche le CSV dans plusieurs dossiers courants possibles."""
    possible_paths = [
        "more-than-carbon-data.csv",
        "data/more-than-carbon-data.csv",
        "measure/data/more-than-carbon-data.csv",
        "../data/more-than-carbon-data.csv"
    ]
    
    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break
            
    if csv_path is None:
        print(f"[ERREUR CRITIQUE MTC] Fichier 'more-than-carbon-data.csv' introuvable.")
        print(f"[DEBUG MTC] Dossier d'exécution actuel : {os.getcwd()}")
        return None, None

    print(f"[DEBUG MTC] Chargement de la base depuis : {csv_path}")
    df = pd.read_csv(csv_path, sep=";")

    comp_cols = ["Casing", "Heatsink", "PCB", "Main dies", "POP", "Upstream transport"]
    for c in comp_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["Total"] = df[[c for c in comp_cols if c in df.columns]].sum(axis=1)
    
    base_impacts = {}
    all_categories = df["Impact Category"].unique().tolist()
    gpu_names = df["GPU_Name"].unique()
    
    for gpu_name in gpu_names:
        if gpu_name not in BASE_GPUS_SPECS:
            continue
        base_impacts[gpu_name] = {}
        df_gpu = df[df["GPU_Name"] == gpu_name].set_index("Impact Category")
        for cat in all_categories:
            if cat in df_gpu.index:
                base_impacts[gpu_name][cat] = {
                    "heatsink": df_gpu.loc[cat, "Heatsink"],
                    "chip": df_gpu.loc[cat, "Main dies"],
                    "total": df_gpu.loc[cat, "Total"]
                }
            else:
                base_impacts[gpu_name][cat] = {"heatsink": 0.0, "chip": 0.0, "total": 0.0}

    return base_impacts, all_categories

def compute_F_heatsink(gpu_target, base_specs, base_impacts, category):
    impacts, dists = [], []
    dims = ["tdp", "die_area"]
    
    for b_name, b_impacts_cat in base_impacts.items():
        b_spec = base_specs[b_name]
        F_heat_b = b_impacts_cat[category]["heatsink"]
        
        sum_dims = sum(gpu_target[d] / b_spec[d] for d in dims)
        impact = (sum_dims / len(dims)) * F_heat_b
        impacts.append(impact)
        
        dist = sum(abs((gpu_target[d] / b_spec[d]) - 1) for d in dims)
        dists.append(dist)
        
    impacts = np.array(impacts)
    dists = np.array(dists)
    
    # --- RÈGLE D'EXCLUSION POUR GPU IDENTIQUE ---
    min_dist_idx = np.argmin(dists)
    if dists[min_dist_idx] < 1e-6: # Si la distance est minuscule
        pond_normalized = np.zeros_like(dists)
        pond_normalized[min_dist_idx] = 1.0 # 100% sur ce GPU
    else:
        # --- TON ALGORITHME SOFTMAX ORIGINAL ---
        pond = 1 - softmax(dists)
        pond_normalized = pond / np.sum(pond) if np.sum(pond) > 0 else pond
        
    f_impacts = np.dot(impacts, pond_normalized)
    return f_impacts


def compute_F_chip(gpu_target, base_specs, base_impacts, category):
    impacts, dists = [], []
    dims = ["die_area", "density", "mem_size"]
    
    for b_name, b_impacts_cat in base_impacts.items():
        b_spec = base_specs[b_name]
        F_chip_b = b_impacts_cat[category]["chip"]
        
        sum_dims = sum(gpu_target[d] / b_spec[d] for d in dims)
        impact = (sum_dims / len(dims)) * F_chip_b
        impacts.append(impact)
        
        dist = sum(abs((gpu_target[d] / b_spec[d]) - 1) for d in dims)
        dists.append(dist)
        
    impacts = np.array(impacts)
    dists = np.array(dists)
    
    # --- RÈGLE D'EXCLUSION POUR GPU IDENTIQUE ---
    min_dist_idx = np.argmin(dists)
    if dists[min_dist_idx] < 1e-6:
        pond_normalized = np.zeros_like(dists)
        pond_normalized[min_dist_idx] = 1.0
    else:
        # --- TON ALGORITHME SOFTMAX ORIGINAL ---
        pond = 1 - softmax(dists)
        pond_normalized = pond / np.sum(pond) if np.sum(pond) > 0 else pond
        
    f_impacts = np.dot(impacts, pond_normalized)
    return f_impacts


def compute_alpha(gpu_target, f_heatsink_target, f_chip_target, base_specs, base_impacts, category):
    impacts, dists = [], []
    dims = ["tdp", "die_area", "density", "mem_size"]
    
    for b_name, b_impacts_cat in base_impacts.items():
        b_spec = base_specs[b_name]
        F_tot_b = b_impacts_cat[category]["total"]
        F_chip_b = b_impacts_cat[category]["chip"]
        F_heat_b = b_impacts_cat[category]["heatsink"]
        
        denom = b_impacts_cat[category]["chip"] + b_impacts_cat[category]["heatsink"]
        alpha_ratio_b = (F_tot_b - denom) / denom if denom != 0 else 0
        impact = (f_heatsink_target + f_chip_target) * alpha_ratio_b
        impacts.append(impact)
        
        dist = sum(abs((gpu_target[d] / b_spec[d]) - 1) for d in dims)
        dists.append(dist)
        
    impacts = np.array(impacts)
    dists = np.array(dists)
    
    # --- RÈGLE D'EXCLUSION POUR GPU IDENTIQUE ---
    min_dist_idx = np.argmin(dists)
    if dists[min_dist_idx] < 1e-6:
        pond_normalized = np.zeros_like(dists)
        pond_normalized[min_dist_idx] = 1.0
    else:
        # --- TON ALGORITHME SOFTMAX ORIGINAL ---
        pond = 1 - softmax(dists)
        pond_normalized = pond / np.sum(pond) if np.sum(pond) > 0 else pond
        
    f_impacts = np.dot(impacts, pond_normalized)
    return f_impacts

def main_impact():
    print("\n--- [START] main_impact ---")
    base_impacts, all_categories = load_base_gpus_impacts()
    if not base_impacts:
        print("[ABORT] Impossible de charger les impacts de base.")
        return

    gpus = get_gpu_info_from_env()
    if not gpus:
        print("[ABORT] Aucun GPU détecté via l'environnement.")
        return

    summary_data = []

    for gpu_id, gpu in gpus.items():
        # Vérification stricte des variables
        missing_vars = [d for d in ["die_area", "tdp", "mem_size", "density"] if gpu.get(d, 0) == 0]
        if missing_vars:
            print(f"[ATTENTION MTC] Le GPU {gpu['name']} a des variables manquantes ou à 0 : {missing_vars}. Je le saute.")
            continue

        print(f"[INFO MTC] Calcul des impacts pour {gpu['name']} (ID: {gpu_id})...")
        total_impacts = {"Hardware": f"{gpu['name']}_{gpu_id}", "FU": gpu["fu"]}

        for cat in TARGET_CATEGORIES:
            cat_short = cat.split(" - ")[0]

            f_heat = compute_F_heatsink(gpu, BASE_GPUS_SPECS, base_impacts, cat)
            f_chip = compute_F_chip(gpu, BASE_GPUS_SPECS, base_impacts, cat)
            alpha = compute_alpha(gpu, f_heat, f_chip, BASE_GPUS_SPECS, base_impacts, cat)
            f_total = f_heat + f_chip + alpha

            total_impacts[cat_short] = f_total

            # Export explicite des variables d'environnement
            os.environ[f"BENCH_GPU_{gpu_id}_IMPACT_{cat_short.replace(' ', '_')}"] = str(f_total)
            print(f"  -> Export: BENCH_GPU_{gpu_id}_IMPACT_{cat_short.replace(' ', '_')} = {f_total}")

        summary_data.append(total_impacts)

    if not summary_data:
        print("[ABORT] Aucune donnée calculée (tous les GPUs ont été sautés).")
        return

    # Sauvegarde CSV (optionnel, pour vérification)
    os.makedirs("data", exist_ok=True)
    df_summary = pd.DataFrame(summary_data)
    csv_cols = ["Hardware", "FU"] + [c.split(" - ")[0] for c in TARGET_CATEGORIES]
    df_summary[csv_cols].to_csv("data/manufacturing_impact_summary_mtc.csv", index=False)
    print("\n--- [END] main_impact. Calculs réussis ! ---")

if __name__ == "__main__":
    main_impact()