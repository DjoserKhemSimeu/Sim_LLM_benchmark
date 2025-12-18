#!/usr/bin/env python3
import os
import json
import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import importlib.util
# Constants
seven_years = 61320 * 3600  # seconds in 7 years (~61320 hours?) keep same as perf_show
three_years = 26298 * 3600
ITERATION = int(os.environ.get("BENCH_ITERATION", 10))
# --- Helpers to load inputs ---

def get_gpu_info_from_env():
    num_gpus = int(os.environ.get("BENCH_NUM_GPU", 0))
    PUE = float(os.environ.get("BENCH_PUE", 1.0))
    gpus = {}
    for gpu_id in range(num_gpus):
        prefix = f"BENCH_GPU_{gpu_id}"
        gpus[gpu_id] = {
            "name": os.environ.get(f"{prefix}_NAME", f"GPU_{gpu_id}"),
            # fallback impact used in original perf_show but we will override with manufacturing CSV
            "impact": float(os.environ.get(f"{prefix}_IMPACT", 0.0)),
        }
    return gpus, PUE


def load_manufacturing_impacts(path="data/manufacturing_impact_summary_mtc.csv"):
    p = Path(path)
    if not p.exists():
        print(f"Warning: manufacturing summary not found at {path}. Expected output from bar_impact_mtc.")
        return pd.DataFrame()
    df = pd.read_csv(p)
    # Ensure Hardware column exists
    if "Hardware" not in df.columns:
        raise ValueError("manufacturing CSV must contain 'Hardware' column")
    df = df.set_index("Hardware")
    return df


def load_electricity_impacts(path="data/Electricity_impacts.csv", factors=["GWP", "ADPe", "WU"]):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Electricity impacts file not found: {path}")
    df = pd.read_csv(p, sep=";")
    # Map short names
    mapping = {}
    for _, row in df.iterrows():
        cat = str(row.get("Impact Category", ""))
        short = cat.split(" - ")[0] if " - " in cat else cat
        try:
            val = float(row.get("Value", 0))
        except Exception:
            val = 0.0
        mapping[short] = val
    # Return only requested factors with fallback 0
    return {f: mapping.get(f, 0.0) for f in factors}


def load_sucess_rates():
    # If a precomputed grouped results CSV exists, use it to avoid rescanning files
    cached_group = Path("measure/data/pytest_grouped_results.csv")
    if cached_group.exists():
        try:
            df_grouped = pd.read_csv(cached_group)
            # ensure nb_users is integer when possible
            if "nb_users" in df_grouped.columns:
                try:
                    df_grouped["nb_users"] = df_grouped["nb_users"].astype(int)
                except Exception:
                    pass
            print(f"Using cached grouped results from {cached_group}")
            return df_grouped
        except Exception as e:
            print(f"Failed to read cached grouped results {cached_group}: {e}. Recomputing...")

    BASE_DIR = "agent_env"
    folder_pattern = re.compile(r"agent_env_user_(.+)_(\d+)_(\d+)_(\d+)")

    rows = []

    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        match = folder_pattern.match(folder)
        if not match:
            continue

        model, nb_users, agent_id, run_id = match.groups()
        nb_users = int(nb_users)
        agent_id = int(agent_id)
        run_id = int(run_id)

        # Chemin vers app.py
        app_file = os.path.join(folder_path, "dummy_agent", "app.py")

        # Cas 1 : app.py existe → tester la fonction addition()
        if os.path.isfile(app_file):
            try:
                # Charger dynamiquement le module
                spec = importlib.util.spec_from_file_location("app_module", app_file)
                app_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(app_module)

                # Tester la fonction addition()
                result = app_module.addition(2, 3)  # Test avec 2 + 3 = 5
                if result == 5:
                    percent = 100  # Succès
                else:
                    percent = 0     # Échec
            except Exception as e:
                print(f"Erreur lors de l'exécution de {app_file}: {e}")
                percent = 0
        else:
            percent = 0

        rows.append({
            "model": model,
            "nb_users": nb_users,
            "agent_id": agent_id,
            "run_id": run_id,
            "percent": percent
        })

    # DataFrame complet
    df = pd.DataFrame(rows)

    # Regroupement par modèle et nb_users
    df_grouped = (
        df.groupby(["model", "nb_users"])["percent"]
        .mean()
        .reset_index()
        .rename(columns={"percent": "mean_success_percent"})
    )

    print("\n=== Pourcentage moyen de réussite par modèle & nb_users ===")
    print(df_grouped)
    df_grouped.to_csv("measure/data/pytest_grouped_results.csv", index=False)
    df.to_csv("measure/data/pytest_raw_results.csv", index=False)

    return df_grouped


def load_sucess_rates_old():
    BASE_DIR = "agent_env"

    # Regex pour extraire le pourcentage dans stdout
    percent_pattern = re.compile(r"\[(\d+)%\]")

    # Regex pour extraire model, nb_users, id depuis le nom du dossier
    folder_pattern = re.compile(r"agent_env_user_(.+)_(\d+)_(\d+)_(\d+)")

    rows = []

    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        match = folder_pattern.match(folder)
        if not match:
            continue

        model, nb_users,agent_id, run_id = match.groups()
        nb_users = int(nb_users)
        agent_id = int(agent_id)
        run_id = int(run_id)

        # Chemin du fichier JSON
        json_file = os.path.join(folder_path, "dummy_agent", "pytest_results.json")

        # -------------------------------
        #  CAS 1 : fichier présent → lire
        #  CAS 2 : pas de fichier → percent = 0
        # -------------------------------
        if os.path.isfile(json_file):
            with open(json_file, "r") as f:
                data = json.load(f)

            stdout = data.get("stdout", "")
            percent_match = percent_pattern.search(stdout)

            percent = int(percent_match.group(1)) if percent_match else 0
        else:
            # Aucun fichier → échec complet
            percent = 0

        rows.append({
            "model": model,
            "nb_users": nb_users,
            "agent_id": agent_id,
            "run_id": run_id,
            "percent": percent
        })

    # DataFrame complet
    df = pd.DataFrame(rows)

    # ----------------------------------------
    #     REGROUPEMENT PAR MODEL / NB_USERS
    # ----------------------------------------
    df_grouped = (
        df.groupby(["model", "nb_users"])["percent"]
        .mean()
        .reset_index()
        .rename(columns={"percent": "mean_success_percent"})
    )
    
    print("\n=== Pourcentage moyen de réussite par modèle & nb_users ===")
    print(df_grouped)
    df_grouped.to_csv("measure/data/pytest_grouped_results.csv", index=False)
    df.to_csv("measure/data/pytest_raw_results.csv", index=False)

    return df_grouped

# --- Energy computation helpers (copied from perf_show) ---

def compute_energy(power_profile):
    timestamps = power_profile["timestamp"]
    power = power_profile["gpu_power"]
    mask = np.isfinite(power)
    power = power[mask]
    timestamps = timestamps[mask]
    return np.trapz(power, timestamps) / 3_600_000  # kWh


def compute_duration(power_profile):
    timestamps = power_profile["timestamp"]
    return timestamps.iloc[-1] - timestamps.iloc[0]


# --- Impact computation for multiple factors ---
def compute_impact_mtc(energy_kWh, manuf_row, electricity_factors, PUE, duration):
    # energy adjusted by PUE
    energy_kWh_eff = energy_kWh * PUE
    results = {}
    for factor, elec_val in electricity_factors.items():
        # usage impact = energy * elec_val (units depend on factor)
        usage = energy_kWh_eff * elec_val
        manuf_value = 0.0
        if manuf_row is not None and factor in manuf_row:
            try:
                manuf_value = float(manuf_row.get(factor, 0.0))
            except Exception:
                manuf_value = 0.0
        soft_manuf = (duration / three_years) * manuf_value
        total = usage + soft_manuf
        results[factor] = {
            "total": total,
            "usage": usage,
            "manufacturing": soft_manuf,
        }
    return results


# --- Load power profiles similar to perf_show (reads /tmp files) ---

def load_power_profiles(gpus, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = ["mistral:7b", "gpt-oss:20b", "gemma3:12b"]
    power_profiles = {gpu_id: {model: {} for model in models} for gpu_id in gpus}
    for gpu_id in gpus:
        for model in models:
            for nb_user in user_counts:
                # Try to gather ITERATION files named with an iteration suffix.
                series_list = []
                found_any = False
                for it in range(ITERATION):
                    file_path_iter = f"/tmp/save_data/consommation_energie_gpu_{gpu_id}_{nb_user}_{model}_{it}.csv"
                    if os.path.isfile(file_path_iter):
                        try:
                            df = pd.read_csv(file_path_iter)
                            if df.empty:
                                print(f"Warning: empty file {file_path_iter}")
                                continue
                            if "timestamp" not in df.columns or "gpu_power" not in df.columns:
                                print(f"Warning: expected columns missing in {file_path_iter}")
                                continue
                            # Ensure timestamps are numeric and rows are sorted
                            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                            df = df.dropna(subset=["timestamp", "gpu_power"]).sort_values("timestamp")
                            s = df.set_index("timestamp")["gpu_power"].astype(float)
                            s.name = f"iter_{it}"
                            series_list.append(s)
                            found_any = True
                        except Exception as e:
                            print(f"Warning: failed to read {file_path_iter}: {e}")

                # Fallback: if no iter files found, try the old single-file name
                if not found_any:
                    file_path = f"/tmp/save_data/consommation_energie_gpu_{gpu_id}_{nb_user}_{model}.csv"
                    if os.path.isfile(file_path):
                        try:
                            df = pd.read_csv(file_path)
                            if df.empty:
                                print(f"Warning: empty file {file_path}")
                            elif "timestamp" in df.columns and "gpu_power" in df.columns:
                                df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                                df = df.dropna(subset=["timestamp", "gpu_power"]).sort_values("timestamp")
                                s = df.set_index("timestamp")["gpu_power"].astype(float)
                                s.name = "iter_0"
                                series_list.append(s)
                                found_any = True
                            else:
                                print(f"Warning: expected columns missing in {file_path}")
                        except Exception as e:
                            print(f"Warning: failed to read {file_path}: {e}")

                if not found_any:
                    print(f"Warning: no power profile files found for gpu={gpu_id}, model={model}, users={nb_user}")
                    continue

                # Concatenate all iteration series on timestamp index and compute mean across iterations
                try:
                    # Build a common timestamp index (union of all timestamps)
                    union_idx = np.unique(np.concatenate([s.index.values for s in series_list]))

                    # Reindex and interpolate each iteration on the union index
                    reindexed = []
                    for s in series_list:
                        s2 = s.reindex(union_idx)
                        # linear interpolate, allow filling at boundaries
                        s2 = s2.interpolate(method="linear", limit_direction="both")
                        reindexed.append(s2)

                    df_concat = pd.concat(reindexed, axis=1, join="outer")
                    print(df_concat.head())
                    # compute mean across iterations (skip NaN)
                    df_mean = df_concat.mean(axis=1, skipna=True).reset_index()
                    df_mean.columns = ["timestamp", "gpu_power"]
                    # sort by timestamp
                    df_mean = df_mean.sort_values("timestamp").reset_index(drop=True)
                    power_profiles[gpu_id][model][nb_user] = df_mean
                except Exception as e:
                    print(f"Warning: failed to aggregate iterations for gpu={gpu_id}, model={model}, users={nb_user}: {e}")

    return power_profiles

def plot_power_profiles(power_profiles, gpus, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = ["mistral:7b", "gpt-oss:20b", "gemma3:12b"]
    for model in models:
        for nb_user in user_counts:
            fig, ax = plt.subplots(figsize=(10, 5))
            for gpu_id, gpu in gpus.items():
                if nb_user in power_profiles[gpu_id][model]:
                    ax.plot(
                        power_profiles[gpu_id][model][nb_user]["timestamp"],
                        power_profiles[gpu_id][model][nb_user]["gpu_power"],
                        label=f"{gpu['name']} (GPU {gpu_id})",
                    )
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Power (W)")
            ax.set_title(f"Power consumption profiles - {model} ({nb_user} users)")
            ax.legend()
            ax.grid(True)
            os.makedirs("images/power_profiles", exist_ok=True)
            plt.savefig(
                f"images/power_profiles/power_profile_{model}_{nb_user}_users.png",
                bbox_inches="tight",
            )
            plt.close()


# --- Aggregation for multiple factors ---
def aggregate_global_impacts_mtc(impacts, models, user_counts, factors):
    # returns dict[factor][model][nb_user] = {total, usage, manufacturing}
    global_impacts = {factor: {model: {} for model in models} for factor in factors}
    for factor in factors:
        for model in models:
            for nb_user in user_counts:
                total = 0.0
                usage = 0.0
                manufacturing = 0.0
                for gpu_name, gpu_data in impacts.items():
                    if model in gpu_data and nb_user in gpu_data[model]:
                        vals = gpu_data[model][nb_user].get(factor, {})
                        total += vals.get("total", 0.0)
                        usage += vals.get("usage", 0.0)
                        manufacturing += vals.get("manufacturing", 0.0)
                global_impacts[factor][model][nb_user] = {
                    "total": total,
                    "usage": usage,
                    "manufacturing": manufacturing,
                }
    return global_impacts


def save_global_impacts_to_csv_mtc(global_impacts, impacts, filename="measure/data/global_impacts_mtc.csv"):
    rows = []
    # per-gpu
    for gpu_name, gpu_data in impacts.items():
        for model, model_data in gpu_data.items():
            for nb_user, impact_values in model_data.items():
                # impact_values is dict[factor] -> dict
                for factor, vals in impact_values.items():
                    rows.append(
                        {
                            "gpu": gpu_name,
                            "model": model,
                            "nb_user": nb_user,
                            "factor": factor,
                            "scope": "per_gpu",
                            "total": vals.get("total", 0.0),
                            "usage": vals.get("usage", 0.0),
                            "manufacturing": vals.get("manufacturing", 0.0),
                        }
                    )
    # global
    for factor, models_data in global_impacts.items():
        for model, user_data in models_data.items():
            for nb_user, vals in user_data.items():
                rows.append(
                    {
                        "gpu": "GLOBAL",
                        "model": model,
                        "nb_user": nb_user,
                        "factor": factor,
                        "scope": "global",
                        "total": vals.get("total", 0.0),
                        "usage": vals.get("usage", 0.0),
                        "manufacturing": vals.get("manufacturing", 0.0),
                    }
                )
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["gpu", "model", "nb_user", "factor", "scope", "total", "usage", "manufacturing"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Données enregistrées dans {filename}")


# --- Plotting functions (one per factor) ---

def plot_impact_bar_global_mtc(global_impacts, factors, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = list(next(iter(global_impacts.values())).keys())
    for factor in factors:
        colors = sns.color_palette("husl", n_colors=len(models))
        model_color_map = {model: colors[i] for i, model in enumerate(models)}

        plt.figure(figsize=(14, 8))
        bar_width = 0.25
        group_spacing = 1.5

        for user_idx, nb_user in enumerate(user_counts):
            group_start = user_idx * (len(models) * group_spacing)
            for model_idx, model in enumerate(models):
                x_pos = group_start + model_idx * group_spacing
                impact_value = global_impacts[factor][model][nb_user]["total"]
                plt.bar(
                    x_pos,
                    impact_value,
                    width=bar_width,
                    color=model_color_map[model],
                    edgecolor="black",
                    label=f"{model}" if user_idx == 0 else "",
                )

            if user_idx < len(user_counts) - 1:
                separator_x = (group_start + len(models) * group_spacing - bar_width / 2 ) - 0.75
                plt.axvline(x=separator_x + bar_width, color="gray", linestyle="--", linewidth=1.5)

        x_ticks = [
            user_idx * (len(models) * group_spacing) + (len(models) * group_spacing) / 2 - 0.5
            for user_idx in range(len(user_counts))
        ]
        x_labels = [f"{nb_user} user{'s' if nb_user > 1 else ''}" for nb_user in user_counts]

        plt.xticks(x_ticks, x_labels)
        plt.xlabel("Number of users")
        plt.ylabel(f"Impact ({factor})")
        plt.title(f"Global impact (sum over all GPUs) - {factor}")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        outdir = "images/impact_plots"
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/global_impact_by_model_{factor}.png", bbox_inches="tight", dpi=300)
        plt.close()


def plot_manufacturing_vs_usage_global_mtc(global_impacts, factors, user_counts=[1, 10, 100], models=None):
    if models is None:
        models = list(next(iter(global_impacts.values())).keys())

    for factor in factors:
        plt.figure(figsize=(14, 8))
        bar_width = 0.4
        positions = np.arange(len(models) * len(user_counts))

        for model_idx, model in enumerate(models):
            for user_idx, nb_user in enumerate(user_counts):
                idx = model_idx * len(user_counts) + user_idx
                data = global_impacts[factor][model][nb_user]
                manufacturing = data["manufacturing"]
                usage = data["usage"]
                plt.bar(idx, manufacturing, width=bar_width, color="b", label="Manufacturing" if idx == 0 else "")
                plt.bar(idx, usage, width=bar_width, bottom=manufacturing, color="g", label="Usage" if idx == 0 else "")

        xticks_labels = [f"{model}\n({nb_user} users)" for model in models for nb_user in user_counts]
        plt.xticks(positions, xticks_labels, rotation=45, ha="right")
        plt.ylabel(f"Impact ({factor})")
        plt.title(f"Global impact breakdown (manufacturing vs usage) - {factor}")
        plt.legend()
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        outdir = "images/proportion_plots"

        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/global_breakdown_all_models_{factor}.png", bbox_inches="tight", dpi=300)
        plt.close()


def plot_combined_global_impact(global_impacts, factors, user_counts=[1, 10, 100], models=None):
    """Trace un seul bar plot global où les 3 facteurs sont affichés
    côte-à-côte pour chaque combinaison (model, user_count) et utilisent
    trois axes Y distincts avec leurs unités provenant du CSV more-than-carbon.
    """
    if models is None:
        models = list(next(iter(global_impacts.values())).keys())
    df_sucess = load_sucess_rates()
    # Use explicit unit conversions and labels requested by user
    # Scale values for display: GWP kg -> g ; ADPe kg -> mg ; WU m^3 -> L
    scale_map = {factors[0]: 1000.0, factors[1]: 1e6, factors[2]: 1000.0}
    unit_labels = {factors[0]: "g CO2 eq", factors[1]: "mg Sb eq", factors[2]: "L"}

    # Layout x positions: grouped by user_counts, inner by model
    n_models = len(models)
    n_users = len(user_counts)
    group_spacing = 1.5
    bar_width = 0.2

    # Prepare figure and three y-axes
    fig, ax_gwp = plt.subplots(figsize=(16, 7))
    ax_adpe = ax_gwp.twinx()
    ax_wu = ax_gwp.twinx()
    # offset the third axis
    ax_wu.spines["right"].set_position(("axes", 1.12))

    # Color by model
    colors = sns.color_palette("tab10", n_colors=n_models)
    model_color_map = {models[i]: colors[i] for i in range(n_models)}

    # Define factor colors for axes and legend
    factor_color_map = {factors[0]: "tab:blue", factors[1]: "tab:orange", factors[2]: "tab:green"}
    # Hatch patterns per factor to visually distinguish factors when bars overlap
    factor_hatch_map = {factors[0]: "///", factors[1]: "\\\\", factors[2]: "xxx"}

    # offsets for factors within each model-slot
    factor_offsets = {factors[0]: -bar_width, factors[1]: 0.0, factors[2]: +bar_width}
    axis_map = {factors[0]: ax_gwp, factors[1]: ax_adpe, factors[2]: ax_wu}

    # Collect x ticks and labels
    x_positions = []
    x_labels = []

    # Plot bars
    for user_idx, nb_user in enumerate(user_counts):
        group_start = user_idx * (n_models * group_spacing)
        for model_idx, model in enumerate(models):
            base_x = group_start + model_idx * group_spacing
            # We'll record GWP (primary axis) display value per slot to position success annotations
            gwp_disp_for_slot = 0.0
            for factor in factors:
                ax = axis_map[factor]
                offset = factor_offsets[factor]
                x_pos = base_x + offset
                try:
                    val = global_impacts[factor][model][nb_user]["total"]
                except Exception:
                    val = 0.0
                # scale to requested display unit
                val_disp = val * scale_map.get(factor, 1.0)
                if factor == factors[0]:
                    gwp_disp_for_slot = val_disp
                ax.bar(
                    x_pos,
                    val_disp,
                    width=bar_width * 0.9,
                    color=model_color_map[model],
                    edgecolor="black",
                    alpha=0.9,
                    hatch=factor_hatch_map.get(factor, ""),
                    linewidth=0.7,
                )

            # only once per model slot, collect for xticks
            x_positions.append(base_x)
            x_labels.append(f"{model}\n({nb_user})")
            # lookup mean success percent for this (model, nb_user) from df_sucess
            percent = 0
            try:
                if df_sucess is not None and not df_sucess.empty:
                    row = df_sucess[(df_sucess['model'] == model) & (df_sucess['nb_users'] == nb_user)]
                    if not row.empty:
                        # some aggregated results may name the percent column differently
                        if 'mean_success_percent' in row.columns:
                            percent = float(row['mean_success_percent'].iloc[0])
                        elif 'percent' in row.columns:
                            percent = float(row['percent'].iloc[0])
                        else:
                            # fallback: try 'mean' or first numeric column
                            for c in row.columns:
                                if pd.api.types.is_numeric_dtype(row[c]):
                                    percent = float(row[c].iloc[0])
                                    break
            except Exception:
                percent = 0

            # store annotation info on primary axis (GWP)
            if 'success_annotations' not in locals():
                success_annotations = []
            success_annotations.append((base_x, gwp_disp_for_slot, percent))

        # draw separator after each user group (except last)
        if user_idx < n_users - 1:
            sep_x = group_start + (n_models - 1) * group_spacing + group_spacing / 2.0
            # draw on primary axis (will appear across twins)
            ax_gwp.axvline(x=sep_x, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)

    # Set xticks at the center of each model slot across all groups
    ax_gwp.set_xticks(x_positions)
    ax_gwp.set_xticklabels(x_labels, rotation=45, ha="right")

    # Y labels with units (no colored axes)
    ax_gwp.set_ylabel(f"{factors[0]} ({unit_labels.get(factors[0], '')})")
    ax_adpe.set_ylabel(f"{factors[1]} ({unit_labels.get(factors[1], '')})")
    ax_wu.set_ylabel(f"{factors[2]} ({unit_labels.get(factors[2], '')})")

    ax_gwp.set_title("Global impact consolidated - GWP / ADPe / WU")

    # Place success percentage annotations above each model slot on the primary axis
    try:
        ylim = ax_gwp.get_ylim()
        y_offset = 0.02 * (ylim[1] - ylim[0])
        for x_pos, gwp_val, percent in success_annotations:
            y = gwp_val + y_offset
            color = 'green' if percent > 49 else 'red'
            ax_gwp.text(x_pos, y, f"{percent:.0f}%", ha='center', va='bottom', color=color, fontweight='bold')
    except Exception:
        # if any issue placing annotations, continue without crashing
        pass

    # Create legends for models and factors
    import matplotlib.patches as mpatches

    model_patches = [mpatches.Patch(color=model_color_map[m], label=m) for m in models]
    # factor legend: show hatch patterns (white fill, black edge)
    factor_patches = [mpatches.Patch(facecolor='white', edgecolor='black', hatch=factor_hatch_map.get(f, ''), label=f) for f in factors]

    # Place legend: models on upper left, factors on upper right
    leg1 = ax_gwp.legend(handles=model_patches, title="Models", bbox_to_anchor=(0, 1.02), loc="lower left")
    ax_gwp.add_artist(leg1)
    ax_gwp.legend(handles=factor_patches, title="Factors", bbox_to_anchor=(1.0, 1.02), loc="lower right")

    plt.tight_layout()
    outdir = "images/combined_impact"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(f"{outdir}/global_combined_impacts.png", bbox_inches="tight", dpi=300)
    plt.close()


# --- Main flow ---
if __name__ == "__main__":
    gpus, PUE = get_gpu_info_from_env()
    user_counts = json.loads(os.environ.get("BENCH_USERS", "[1,10,100]"))
    models = json.loads(os.environ.get("BENCH_MODELS", '["mistral:7b","gpt-oss:20b","gemma3:12b"]'))
    

    # load inputs
    manuf_df = load_manufacturing_impacts()
    factors = ["GWP", "ADPe", "WU"]
    electricity_factors = load_electricity_impacts(path="data/Electricity_impacts.csv", factors=factors)

    power_profiles = load_power_profiles(gpus, user_counts, models)

    # prepare impacts data structure
    impacts = {f"{gpus[gpu_id]['name']}_{gpu_id}": {model: {} for model in models} for gpu_id in gpus}

    # compute per-GPU impacts per factor
    for gpu_id, gpu in gpus.items():
        hw_key = f"{gpu['name']}_{gpu_id}"
        for model in models:
            for nb_user in user_counts:
                if nb_user in power_profiles[gpu_id][model]:
                    energy_kWh = compute_energy(power_profiles[gpu_id][model][nb_user])
                    duration = compute_duration(power_profiles[gpu_id][model][nb_user])
                    manuf_row = None
                    if hw_key in manuf_df.index:
                        manuf_row = manuf_df.loc[hw_key].to_dict()
                    res = compute_impact_mtc(energy_kWh, manuf_row, electricity_factors, PUE, duration)
                    impacts[hw_key][model][nb_user] = res

    # aggregate
    global_impacts = aggregate_global_impacts_mtc(impacts, models, user_counts, factors)
    save_global_impacts_to_csv_mtc(global_impacts, impacts)

    # print summary
    for factor in factors:
        for model in models:
            for nb_user in user_counts:
                data = global_impacts[factor][model][nb_user]
                
                print(f"[GLOBAL][{factor}] {model} ({nb_user} users) → total: {data['total']:.4g}")

    # plots
    plot_power_profiles(power_profiles, gpus, user_counts, models)
    plot_impact_bar_global_mtc(global_impacts, factors, user_counts, models)
    plot_manufacturing_vs_usage_global_mtc(global_impacts, factors, user_counts, models)
    # combined plot with three Y axes (GWP, ADPe, WU)
    plot_combined_global_impact(global_impacts, factors, user_counts, models)
