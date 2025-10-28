#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import signal
import re
from configs.config import set_env_from_gpu_config
from measure.scripts.bar_impact import main_impact

BENCH_SCRIPT = "scripts/multi_gpu_bench.py"
MANUFACTURING_IMPACT_SCRIPT = "measure/scripts/bar_impact.py"
EVALUATION_SCRIPT = "measure/scripts/perf_show.py"
MODELS = ["mistral:7b", "gpt-oss:20b", "gemma3:12b"]


def main():
    parser = argparse.ArgumentParser(description="Lance un benchmark après configuration des GPU.")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier JSON de configuration des GPU.")
    args = parser.parse_args()
    # Libérer les ports utilisés par Ollama avant la prochaine itération


    set_env_from_gpu_config(args.config)

        
    main_impact()
    # Afficher toutes les variables d'environnement
    for key, value in os.environ.items():
        if key.startswith("BENCH_GPU_"):
            print(f"{key}: {value}")

    # Exécution finale du script d'évaluation
    print(f"Lancement du script d'évaluation : {args.config}")
    process = subprocess.Popen(
        [sys.executable, EVALUATION_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    for line in process.stdout:
        print(line, end="")
    return_code = process.wait()
    if return_code != 0:
        print("\n--- ERREUR D'EXÉCUTION (évaluation) ---")
        stderr_output = process.stderr.read()
        print("Backtrace :\n" + stderr_output)
        sys.exit(1)

if __name__ == "__main__":
    main()

