#!/usr/bin/env python3
import os
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
seven_years = 61320 * 3600  # seconds in 7 years (~61320 hours?) keep same as perf_show

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
        soft_manuf = (duration / seven_years) * manuf_value
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
                file_path = f"/tmp/save_data/consommation_energie_gpu_{gpu_id}_{nb_user}_{model}.csv"
                if os.path.isfile(file_path):
                    data = pd.read_csv(file_path)
                    if not data.empty:
                        power_profiles[gpu_id][model][nb_user] = data
                    else:
                        print(f"⚠️ Fichier vide : {file_path}")
                else:
                    print(f"⚠️ Fichier introuvable : {file_path}")
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
    models = ["mistral_7b", "gpt-oss_20b", "gemma3_12b"]

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
