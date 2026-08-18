#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import time
import signal
import re
from configs.config import set_env_from_gpu_config
import json
from measure.scripts.gpu_impact_calculator import main_impact

BENCH_SCRIPT = "scripts/multi_gpu_bench.py"
EVALUATION_SCRIPT = "measure/scripts/evaluation.py"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + os.pathsep + "."


def detecter_ports_ollama():
    """Détecte les ports utilisés par Ollama."""
    try:
        cmd = "lsof -i -P -n | grep LISTEN | grep ollama"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        ports = set()
        for line in result.stdout.splitlines():
            match = re.search(r":(\d+)\s", line)
            if match:
                port = int(match.group(1))
                ports.add(port)
        return list(ports)
    except Exception as e:
        print(f"Erreur lors de la détection des ports : {e}")
        return []


def tuer_tous_processus_ollama():
    """Tue tous les processus liés à Ollama."""
    try:
        # Trouver tous les PIDs des processus "ollama"
        cmd = "pgrep -f ollama"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        pids = result.stdout.strip().split()

        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Processus Ollama (PID: {pid}) arrêté.")
            except Exception as e:
                print(f"Erreur pour PID {pid}: {e}")

        # Attendre 2 secondes pour laisser le temps aux processus de s'arrêter
        time.sleep(2)

        # Vérifier si des processus Ollama restent
        cmd = "pgrep -f ollama"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(
                "Certains processus Ollama n'ont pas pu être arrêtés. Tentative avec SIGKILL..."
            )
            for pid in result.stdout.strip().split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Processus Ollama (PID: {pid}) forcé à s'arrêter.")
                except Exception as e:
                    print(f"Erreur pour PID {pid}: {e}")

    except Exception as e:
        print(f"Erreur lors de l'arrêt des processus Ollama: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Lance un benchmark après configuration des GPU."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin vers le fichier JSON de configuration des GPU.",
    )
    parser.add_argument("--skip-bench", action="store_true", help="Passe l'inférence et va direct à l'évaluation")
    parser.add_argument("--skip-eval", action="store_true", help="Passe l'évaluation")
    args = parser.parse_args()


    # ==========================================
    # PHASE 1 : BENCHMARK (Inférence)
    # ==========================================
    if not args.skip_bench:
        # Libérer les ports utilisés par Ollama avant la prochaine itération
        ports_ollama = detecter_ports_ollama()
        if ports_ollama:
            print(f"Libération des ports utilisés par Ollama : {ports_ollama}")
            tuer_tous_processus_ollama()
        else:
            print("Aucun port Ollama détecté.")

        MODELS = []
        with open(args.config, "r") as f:
            config = json.load(f)
            MODELS = config["Models"]

        for model in MODELS:
            print(f"Running the Sim LLM benchmark with the model: {model}")


            os.environ["BENCH_MODEL"] = model
            set_env_from_gpu_config(args.config)

            # 2. Une fois compilé, on met à jour la variable pour la suite du benchmark
            custom_model = f"{model.replace(':', '-')}-swe"
            os.environ["BENCH_MODEL"] = custom_model

            # ==========================================
            # PHASE 1.1 : GPU impact calculation
            # ==========================================
            main_impact()

            # Afficher toutes les variables d'environnement
            for key, value in os.environ.items():
                if key.startswith("BENCH_GPU_"):
                    print(f"{key}: {value}")


            # Exécution du benchmark
            if not os.path.exists(BENCH_SCRIPT):
                print(f"Erreur : Le fichier {BENCH_SCRIPT} n'existe pas.")
                sys.exit(1)

            print(f"Lancement du benchmark avec la configuration : {args.config}")
            process = subprocess.Popen(
                [sys.executable, BENCH_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="")
            return_code = process.wait()

            if return_code != 0:
                print("\n--- ERREUR D'EXÉCUTION (benchmark) ---")
                stderr_output = process.stderr.read()
                print("Backtrace :\n" + stderr_output)
                sys.exit(1)

            # Libérer les ports utilisés par Ollama avant la prochaine itération
            ports_ollama = detecter_ports_ollama()
            if ports_ollama:
                print(f"Libération des ports utilisés par Ollama : {ports_ollama}")
                tuer_tous_processus_ollama()
            else:
                print("Aucun port Ollama détecté.")
    else:
        print("--- Mode Skip : Inférence ignorée ---")


    # ==========================================
    # PHASE 2 : ÉVALUATION (SWE-Bench & Perf)
    # ==========================================
    if not args.skip_eval:
        SWE_EVAL_SCRIPT = "scripts/run_all_evals.sh"
        print(f"\n--- Lancement de l'évaluation SWE-bench via {SWE_EVAL_SCRIPT} ---")
        if os.path.exists(SWE_EVAL_SCRIPT):
            try:
                # On utilise subprocess.run avec 'bash' pour s'assurer qu'il s'exécute correctement
                # et on attend qu'il termine avant de passer à perf_show
                subprocess.run(["bash", SWE_EVAL_SCRIPT], check=True)
                print("--- Évaluation SWE-bench terminée ---")
            except subprocess.CalledProcessError as e:
                print(f"Erreur critique lors de l'exécution du script Bash SWE-bench : {e}")
                sys.exit(1)
        else:
            print(f"Avertissement : Le fichier {SWE_EVAL_SCRIPT} n'a pas été trouvé. On passe à la suite.")

        # Exécution finale du script d'évaluation
        print(f"Lancement du script d'évaluation : {args.config}")
        process = subprocess.Popen(
            [sys.executable, EVALUATION_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
        return_code = process.wait()

        if return_code != 0:
            print("\n--- ERREUR D'EXÉCUTION (évaluation) ---")
            stderr_output = process.stderr.read()
            print("Backtrace :\n" + stderr_output)
            sys.exit(1)
    else:
        print("--- Mode Skip : Évaluation ignorée ---")


if __name__ == "__main__":
    main()