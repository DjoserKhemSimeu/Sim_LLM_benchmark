#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
from configs.config import set_env_from_gpu_config


BENCH_SCRIPT = "scripts/multi_gpu_bench.py"
MANUFACTURING_IMPACT_SCRIPT = "measure/scripts/bar_impact.py"
EVALUATION_SCRIPT = "measure/scripts/perf_show.py"
MODELS=["mistral:7b","gpt-oss:20b","gemma3:12b"]

def main():
    parser = argparse.ArgumentParser(description="Lance un benchmark après configuration des GPU.")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier JSON de configuration des GPU.")
    args = parser.parse_args()

    # 1. Définir les variables d'environnement depuis le JSON
    for model in MODELS:
        print(f"Running the Sim LLM benchmark with the model: {model} ")
        os.environ["BENCH_MODEL"] = model

        set_env_from_gpu_config(args.config)



        process = subprocess.Popen([sys.executable, MANUFACTURING_IMPACT_SCRIPT], stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
# Lire et afficher les sorties en temps réel
        for line in process.stdout:
            print(line, end="")


        return_code = process.wait()

        if return_code != 0:
            print("Erreur lors de l'exécution d manu impact")
            sys.exit(1)

# 2. Lancer le script de benchmark
        if not os.path.exists(BENCH_SCRIPT):
            print(f"Erreur : Le fichier {BENCH_SCRIPT} n'existe pas.")
            sys.exit(1)

        print(f"Lancement du benchmark avec la configuration : {args.config}")
        process = subprocess.Popen([sys.executable, BENCH_SCRIPT], stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
# Lire et afficher les sorties en temps réel
        for line in process.stdout:
            print(line, end="")


        return_code = process.wait()

        if return_code != 0:
            print("\n--- ERREUR D'EXÉCUTION ---")
            stderr_output = process.stderr.read()
            print("Backtrace :\n" + stderr_output)
            sys.exit(1)

            

    print(f"Lancement du script d'évaluation : {args.config}")
    process = subprocess.Popen([sys.executable, EVALUATION_SCRIPT], stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
    # Lire et afficher les sorties en temps réel
    for line in process.stdout:
        print(line, end="")
    

    return_code = process.wait()

    if return_code != 0:
        print("\n--- ERREUR D'EXÉCUTION ---")
        stderr_output = process.stderr.read()
        print("Backtrace :\n" + stderr_output)
        sys.exit(1)

        




if __name__ == "__main__":
    main()

