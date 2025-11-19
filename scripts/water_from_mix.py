#!/usr/bin/env python3
"""
Calcule la consommation d'eau (L) associée à la production de 1 kWh
selon un fichier CSV de mix énergétique.

Format attendu (séparateur `;`):
Source;Percentage;Water Use(L/MWh)
Wind;14.0;279
...

Exemple:
    python scripts/water_from_mix.py --csv data/Energetic_mix_Fr.csv --kwh 1

Le script affiche un tableau avec:
- WaterUse_L_per_kWh (L/kWh pour chaque source)
- Contribution_L_per_kWh (pondérée par la part du mix)
- Total L/kWh

Il peut aussi écrire un fichier de sortie CSV ou JSON via --out
"""

import argparse
import pandas as pd
from pathlib import Path
import json
import sys


def compute_water_per_kwh(df, kwh=1.0):
    # s'attend à des colonnes: Source, Percentage, Water Use(L/MWh)
    if "Percentage" not in df.columns or "Water Use(L/MWh)" not in df.columns:
        raise ValueError("Le fichier doit contenir les colonnes 'Percentage' et 'Water Use(L/MWh)'")

    # Nettoyage simple
    df = df.copy()
    df["Percentage"] = pd.to_numeric(df["Percentage"], errors="coerce").fillna(0.0)
    df["Water Use(L/MWh)"] = pd.to_numeric(df["Water Use(L/MWh)"], errors="coerce").fillna(0.0)

    # convertir L/MWh -> L/kWh (1 MWh = 1000 kWh)
    df["WaterUse_L_per_kWh"] = df["Water Use(L/MWh)"] / 1000.0 * (kwh)

    # Conversion en m^3: 1 m3 = 1000 L
    df["WaterUse_m3_per_kWh"] = df["WaterUse_L_per_kWh"] / 1000.0

    # contribution pondérée: (percentage/100) * WaterUse_L_per_kWh
    df["Contribution_L_per_kWh"] = (df["Percentage"] / 100.0) * df["WaterUse_L_per_kWh"]

    # Contribution en m3
    df["Contribution_m3_per_kWh"] = (df["Percentage"] / 100.0) * df["WaterUse_m3_per_kWh"]

    total = df["Contribution_L_per_kWh"].sum()
    total_m3 = df["Contribution_m3_per_kWh"].sum()
    return df, total, total_m3


def main():
    parser = argparse.ArgumentParser(description="Calcul consommation d'eau par kWh selon mix énergétique")
    parser.add_argument("--csv", "-c", default="data/Energetic_mix_Fr.csv", help="Chemin vers le CSV du mix énergétique (séparateur ;) ")
    parser.add_argument("--kwh", "-k", type=float, default=1.0, help="Nombre de kWh à calculer (par défaut 1)")
    parser.add_argument("--out", "-o", help="Fichier de sortie (CSV ou JSON) pour le tableau détaillé")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Fichier introuvable: {csv_path}", file=sys.stderr)
        sys.exit(2)

    # Lire en supposant séparateur ; (comme votre fichier)
    try:
        df = pd.read_csv(csv_path, sep=";")
    except Exception as e:
        print(f"Erreur lecture CSV: {e}", file=sys.stderr)
        sys.exit(3)

    df_result, total, total_m3 = compute_water_per_kwh(df, kwh=args.kwh)

    # Affichage lisible
    display_cols = [
        c
        for c in [
            "Source",
            "Percentage",
            "Water Use(L/MWh)",
            "WaterUse_L_per_kWh",
            "WaterUse_m3_per_kWh",
            "Contribution_L_per_kWh",
            "Contribution_m3_per_kWh",
        ]
        if c in df_result.columns
    ]
    df_print = df_result[display_cols].copy()
    df_print["WaterUse_L_per_kWh"] = df_print["WaterUse_L_per_kWh"].map(lambda x: round(x, 6))
    df_print["Contribution_L_per_kWh"] = df_print["Contribution_L_per_kWh"].map(lambda x: round(x, 6))

    print("\nConsommation d'eau par source (pour {0} kWh):\n".format(args.kwh))
    print(df_print.to_markdown(index=False))
    print("\n**Total water consumption per {0} kWh: {1:.6f} L ({2:.6f} m^3)**\n".format(args.kwh, total, total_m3))

    # Écriture éventuelle du résultat détaillé
    if args.out:
        out_path = Path(args.out)
        if out_path.suffix.lower() == ".csv":
            df_result.to_csv(out_path, index=False, sep=";")
            print(f"Détails enregistrés dans {out_path}")
        elif out_path.suffix.lower() in (".json", ".ndjson"):
            df_result.to_json(out_path, orient="records", force_ascii=False)
            print(f"Détails enregistrés dans {out_path}")
        else:
            # par défaut JSON
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(df_result.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
            print(f"Détails enregistrés dans {out_path}")


if __name__ == "__main__":
    main()
