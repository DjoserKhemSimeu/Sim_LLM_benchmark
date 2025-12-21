#!/usr/bin/env python3
"""Plot comparison of observed impacts (g CO2 eq) between infrastructures.

Reads `scripts/impacts.csv` (semicolon-separated) with columns:
infrastructure;model;nb_users;scope;total;usage;manufacturing;Success

Produces grouped bar charts per `model` with x-axis = number of users,
bars = infrastructures, and a green marker for success / red X for failure.
Saves image to `images/impact_compare.png` by default.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


def read_impacts(path: str) -> pd.DataFrame:
    # file uses semicolon separator; header row is present but first column name may contain a comma
    df = pd.read_csv(path, sep=';', engine='python')

    # Normalize/guess column names robustly (case-insensitive and tolerant to malformations)
    orig_cols = list(df.columns)
    stripped = [c.strip() for c in orig_cols]
    new_cols = []
    for c in stripped:
        lc = c.lower()
        if ',' in c or lc.startswith('gpu'):
            new_cols.append('infrastructure')
        elif lc in ('model',):
            new_cols.append('model')
        elif lc in ('nb_users', 'nb users', 'nbusers'):
            new_cols.append('nb_users')
        elif lc in ('scope',):
            new_cols.append('scope')
        elif lc in ('total',):
            new_cols.append('total')
        elif lc in ('usage',):
            new_cols.append('usage')
        elif lc in ('manufacturing',):
            new_cols.append('manufacturing')
        elif lc in ('success',):
            new_cols.append('Success')
        else:
            new_cols.append(c)

    df.columns = new_cols

    # If some expected columns are still missing, try case-insensitive match or guess the last column for Success
    lc_map = {c.lower(): c for c in df.columns}
    for col in ['infrastructure', 'model', 'nb_users', 'total', 'Success']:
        if col not in df.columns:
            key = col.lower()
            if key in lc_map:
                df = df.rename(columns={lc_map[key]: col})
            else:
                # Special heuristic: if Success is missing, assume last column is Success when it contains boolean-like strings
                if col == 'Success' and len(df.columns) >= 1:
                    sample = df.iloc[:, -1].astype(str).str.lower().head(10).tolist()
                    if any(s in ('true', 'false', '1', '0', 'yes', 'no') for s in sample):
                        df = df.rename(columns={df.columns[-1]: 'Success'})
                        continue
                raise ValueError(f"Missing expected column '{col}' in {path}")

    # Convert types
    # nb_users may be numeric but read as object; try to convert to int
    try:
        df['nb_users'] = pd.to_numeric(df['nb_users'], errors='coerce').astype(int)
    except Exception:
        # fallback: extract digits
        df['nb_users'] = df['nb_users'].astype(str).str.extract(r'(\d+)').astype(int)

    # total is expected in g CO2 eq already; ensure numeric
    df['total'] = pd.to_numeric(df['total'], errors='coerce')

    # ensure usage/manufacturing numeric if present; if one missing, infer from total
    if 'usage' in df.columns:
        df['usage'] = pd.to_numeric(df['usage'], errors='coerce')
    if 'manufacturing' in df.columns:
        df['manufacturing'] = pd.to_numeric(df['manufacturing'], errors='coerce')
    if 'usage' not in df.columns and 'manufacturing' in df.columns:
        df['usage'] = df['total'] - df['manufacturing']
    if 'manufacturing' not in df.columns and 'usage' in df.columns:
        df['manufacturing'] = df['total'] - df['usage']

    # Success column to boolean
    if df['Success'].dtype == object:
        df['Success'] = df['Success'].astype(str).str.lower().isin(['true', '1', 'yes'])
    else:
        df['Success'] = df['Success'].astype(bool)

    return df


def plot_impacts(df: pd.DataFrame, out_path: str, figsize=(14, 6)) -> None:
    models = sorted(df['model'].unique())
    num_models = len(models)

    # layout: 1 column per model if few models, else 2 cols
    ncols = 1 if num_models <= 3 else 2
    nrows = int(np.ceil(num_models / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1] * nrows))
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Increase font sizes for readability
    title_fs = 24
    label_fs = 18
    tick_fs = 15
    legend_fs = 15
    value_fs = 15

    # Global color palette for infrastructures
    infrastructures = sorted(df['infrastructure'].unique())
    cmap = plt.get_cmap('tab10')
    colors = {infra: cmap(i % 10) for i, infra in enumerate(infrastructures)}

    for ax_idx, model in enumerate(models):
        ax = axes[ax_idx]
        dfm = df[df['model'] == model]
        nb_users = sorted(dfm['nb_users'].unique())
        x = np.arange(len(nb_users))

        n_infra = len(infrastructures)
        total_width = 0.8
        bar_width = total_width / max(1, n_infra)

        max_height = 0.0
        for j, infra in enumerate(infrastructures):
            usages = []
            manufs = []
            successes = []
            for nb in nb_users:
                row = dfm[(dfm['infrastructure'] == infra) & (dfm['nb_users'] == nb)]
                if not row.empty:
                    u = float(row['usage'].values[0]) if 'usage' in row.columns else float(row['total'].values[0])
                    m = float(row['manufacturing'].values[0]) if 'manufacturing' in row.columns else float(row['total'].values[0] - u)
                    usages.append(u)
                    manufs.append(m)
                    successes.append(bool(row['Success'].values[0]))
                else:
                    usages.append(0.0)
                    manufs.append(0.0)
                    successes.append(None)

            positions = x - total_width / 2 + j * bar_width + bar_width / 2

            # plot usage (bottom) and manufacturing (top with hatch)
            bars_usage = ax.bar(positions, usages, width=bar_width, color=colors[infra], alpha=0.9, label=infra if ax_idx == 0 else None, edgecolor='black')
            bars_manuf = ax.bar(positions, manufs, width=bar_width, bottom=usages, color=colors[infra], alpha=0.6, hatch='///', edgecolor='black')

            # update max height for marker placement
            stack_tops = [u + m for u, m in zip(usages, manufs)]
            max_height = max(max_height, max(stack_tops) if stack_tops else 0)

            # add success/failure markers and small numeric labels (total)
            for k in range(len(positions)):
                h = stack_tops[k]
                succ = successes[k]
                cx = positions[k]
                y_marker = h + 0.15 * max_height
                if succ is True:
                    ax.plot(cx, y_marker, marker='o', color='green', markersize=12, zorder=5)
                elif succ is False:
                    ax.plot(cx, y_marker, marker='x', color='red', markersize=12, zorder=5)

                # numeric value above bar (total)
                ax.text(cx, h + max(0.005 * max_height, 0.05), f"{h:.2f}", ha='center', va='bottom', fontsize=value_fs)

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(n)) for n in nb_users])
        ax.set_xlabel('Nombre d\'utilisateurs')
        ax.set_ylabel('Impact total (g CO2 eq)')
        ax.set_title(f'Model: {model}')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

    # remove empty axes
    for i in range(len(models), len(axes)):
        fig.delaxes(axes[i])

    # Legend for infrastructures
    handles = [plt.Line2D([0], [0], color=colors[infra], lw=8) for infra in infrastructures]
    # Infrastructure legend: centered above the figure (use bbox to avoid overlap)
    infra_legend = fig.legend(
        handles,
        infrastructures,
        title='Infrastructure',
        loc='upper center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=min(6, len(infrastructures)),
        fontsize=legend_fs,
    )
    fig.add_artist(infra_legend)

    # Legend for components (usage vs manufacturing): place centered below the plots
    patch_usage = mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Usage')
    patch_manuf = mpatches.Patch(facecolor='lightgray', edgecolor='black', hatch='///', label='Manufacturing')
    fig.legend(
        handles=[patch_usage, patch_manuf],
        loc='lower center',
        bbox_to_anchor=(0.5, 0.04),
        ncol=2,
        fontsize=legend_fs,
    )

    # Add explanation for markers centered below component legend
    fig.text(0.5, 0.01, 'Marker: green circle = success, red × = failure', ha='center', fontsize=legend_fs)

    # leave extra room at the top and bottom for the legends
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main(csv_path: str, out: Optional[str]):
    df = read_impacts(csv_path)
    if out is None:
        out = os.path.join('images', 'impact_compare.png')
    plot_impacts(df, out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Compare observed impacts (g CO2 eq) between infrastructures')
    p.add_argument('--csv', '-c', default=os.path.join('scripts', 'impacts.csv'), help='Path to impacts CSV (semicolon-separated)')
    p.add_argument('--out', '-o', default=os.path.join('images', 'impact_compare.png'), help='Output image path')
    args = p.parse_args()
    main(args.csv, args.out)
