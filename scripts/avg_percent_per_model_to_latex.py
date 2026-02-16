#!/usr/bin/env python3
"""
Lit `measure/data/pytest_raw_results.csv`, calcule la moyenne du champ `percent`
pour chaque `model` et affiche un joli tableau LaTeX sur stdout.

Usage:
    python3 scripts/avg_percent_per_model_to_latex.py [file1.csv file2.csv ...]
"""
import csv
import sys
from collections import defaultdict
from statistics import mean


def format_latex_table(rows, caption="Average percent per model"):
    header = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\begin{tabular}{l r r}\n"
        "\\hline\n"
        "Model & Average (\\%) & Count \\\\ \n"
        "\\hline\n"
    )
    lines = [header]
    for model, avg, cnt in rows:
        lines.append(f"{model} & {avg:0.1f} & {cnt} \\\\ \n")
    footer = (
        "\\hline\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        "\\end{table}\n"
    )
    lines.append(footer)
    return "".join(lines)


def main():
    paths = sys.argv[1:]
    if not paths:
        # default to both known files if no args provided
        paths = [
            "../Sim_Hydra_matrix/measure/data/pytest_raw_results.csv",
            "../Sim_Chuc_matrix/measure/data/pytest_raw_results.csv",
            "../Sim_Grouille_matrix/measure/data/pytest_raw_results.csv",
        ]
        paths_dummy = [
            "../Sim_Hydra/measure/data/pytest_raw_results.csv",
            "../Sim_Chuc/measure/data/pytest_raw_results.csv",
            "../Sim_Grouille/measure/data/pytest_raw_results.csv",
        ]

    data = defaultdict(list)

    for path in paths:
        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                if "model" not in reader.fieldnames or "percent" not in reader.fieldnames:
                    print(f"Fichier CSV invalide — colonnes attendues: 'model', 'percent'. Found: {reader.fieldnames} in {path}")
                    continue
                for row in reader:
                    model = row.get("model", "")
                    perc = row.get("percent", "")
                    if model == "" or perc == "":
                        continue
                    try:
                        p = float(perc)
                    except ValueError:
                        continue
                    data[model].append(p)
        except FileNotFoundError:
            print(f"Fichier non trouvé: {path}")

    results = []
    for model, vals in data.items():
        if vals:
            results.append((model, mean(vals), len(vals)))

    # Trier par moyenne décroissante
    results.sort(key=lambda x: (-x[1], x[0]))

    table = format_latex_table(results, caption="Moyenne du pourcentage par modèle (fichiers: %s)" % ", ".join(paths))
    print(table)


    data = defaultdict(list)
    for path  in paths_dummy:
        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                if "model" not in reader.fieldnames or "percent" not in reader.fieldnames:
                    print(f"Fichier CSV invalide — colonnes attendues: 'model', 'percent'. Found: {reader.fieldnames} in {path}")
                    continue
                for row in reader:
                    model = row.get("model", "")
                    perc = row.get("percent", "")
                    if model == "" or perc == "":
                        continue
                    try:
                        p = float(perc)
                    except ValueError:
                        continue
                    data[model].append(p)
        except FileNotFoundError:
            print(f"Fichier non trouvé: {path}")

    results = []
    for model, vals in data.items():
        if vals:
            results.append((model, mean(vals), len(vals)))

    # Trier par moyenne décroissante
    results.sort(key=lambda x: (-x[1], x[0]))

    table = format_latex_table(results, caption="Moyenne du pourcentage par modèle (fichiers: %s)" % ", ".join(paths_dummy))
    print(table)


if __name__ == "__main__":
    main()
