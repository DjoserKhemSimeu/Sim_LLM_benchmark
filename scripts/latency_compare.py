#!/usr/bin/env python3
"""
Plot latency comparisons between infrastructures per model and number of users.

Reads a CSV (default `scripts/all_data.csv`) with columns:
  Infrastructure;nb_users;avg_latency_s;model

Produces a figure with one subplot per model. For each subplot the x-axis is `nb_users`
and bars correspond to infrastructures (grouped). Bars show mean latency and optional
error bars (std) if multiple samples exist.

Usage:
  python scripts/latency_compare.py --csv scripts/all_data.csv --ms --out images/latency_compare.png

Options:
  --csv PATH    input CSV (sep=';')
  --ms          plot latencies in milliseconds instead of seconds
  --logy        use logarithmic y-axis
  --out PATH    output image path

"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path: Path):
    df = pd.read_csv(path, sep=';')
    # normalize column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if 'avg_latency_s' not in df.columns:
        raise ValueError("CSV must contain column 'avg_latency_s'")
    # ensure numeric types
    df['nb_users'] = pd.to_numeric(df['nb_users'], errors='coerce')
    df['avg_latency_s'] = pd.to_numeric(df['avg_latency_s'], errors='coerce')
    df = df.dropna(subset=['nb_users','avg_latency_s','Infrastructure','model'])
    return df


def prepare_summary(df: pd.DataFrame):
    # aggregate stats per Infrastructure / nb_users / model
    agg = df.groupby(['model','nb_users','Infrastructure'])['avg_latency_s'].agg(['mean','std','count']).reset_index()
    agg = agg.rename(columns={'mean':'lat_mean_s','std':'lat_std_s','count':'n'})
    return agg


def plot_latency(agg: pd.DataFrame, outpath: Path, to_ms=False, logy=False):
    sns.set_style('whitegrid')
    models = sorted(agg['model'].unique())
    infrastructures = sorted(agg['Infrastructure'].unique())
    users = sorted(agg['nb_users'].unique())

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6), sharey=not logy)
    if n_models == 1:
        axes = [axes]

    width = 0.8 / len(infrastructures)  # total width per group

    for ax, model in zip(axes, models):
        data_m = agg[agg['model']==model]
        x_base = np.arange(len(users))

        for i, infra in enumerate(infrastructures):
            d = data_m[data_m['Infrastructure']==infra].set_index('nb_users').reindex(users)
            means = d['lat_mean_s'].values
            stds = d['lat_std_s'].values
            # replace nan with zeros for plotting (will appear empty)
            means = np.nan_to_num(means, nan=0.0)
            stds = np.nan_to_num(stds, nan=0.0)
            if to_ms:
                means = means * 1000.0
                stds = stds * 1000.0
            x = x_base + (i - (len(infrastructures)-1)/2.0) * width
            ax.bar(x, means, width=width*0.95, label=infra, yerr=stds, capsize=3)

        ax.set_xticks(x_base)
        ax.set_xticklabels([str(int(u)) for u in users])
        ax.set_xlabel('Number of users')
        unit = 'ms' if to_ms else 's'
        ax.set_title(f"Model: {model}")
        ax.legend(title='Infrastructure', bbox_to_anchor=(1.02, 1), loc='upper left')
        if logy:
            ax.set_yscale('log')
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax.set_ylabel(f'Average Latency ({unit})')

    plt.tight_layout()
    outdir = outpath.parent
    outdir.mkdir(parents=True, exist_ok=True)
    plt.show()
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='scripts/all_data.csv')
    parser.add_argument('--out', default='images/latency_compare.png')
    parser.add_argument('--ms', action='store_true', help='Plot latencies in milliseconds')
    parser.add_argument('--logy', action='store_true', help='Use logarithmic y axis')
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f'Input file not found: {path}')
        return

    df = load_data(path)
    agg = prepare_summary(df)
    plot_latency(agg, Path(args.out), to_ms=args.ms, logy=args.logy)
    print(f'Latency comparison plot saved to {args.out}')


if __name__ == '__main__':
    main()
