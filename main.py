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


def detecter_ports(engine: str):
    """Détecte les ports utilisés par un moteur spécifique (ollama ou vllm)."""
    try:
        # Trouver d'abord les PIDs du moteur
        cmd_pids = f"pgrep -f {engine}"
        res_pids = subprocess.run(cmd_pids, shell=True, capture_output=True, text=True)
        pids = res_pids.stdout.strip().split()
        
        if not pids:
            return []

        ports = set()
        # Chercher les ports d'écoute pour ces PIDs spécifiquement
        for pid in pids:
            cmd_lsof = f"lsof -i -P -n -a -p {pid} | grep LISTEN"
            res_lsof = subprocess.run(cmd_lsof, shell=True, capture_output=True, text=True)
            for line in res_lsof.stdout.splitlines():
                match = re.search(r":(\d+)\s", line)
                if match:
                    ports.add(int(match.group(1)))
        return list(ports)
    except Exception as e:
        print(f"Erreur lors de la détection des ports pour {engine} : {e}")
        return []


def tuer_tous_processus(engine: str):
    """Tue tous les processus liés au moteur d'inférence (ollama ou vllm)."""
    try:
        # Trouver tous les PIDs des processus
        cmd = f"pgrep -f {engine}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        pids = result.stdout.strip().split()

        if not pids:
            return

        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Processus {engine.capitalize()} (PID: {pid}) arrêté.")
            except Exception as e:
                print(f"Erreur pour PID {pid}: {e}")

        # Attendre 2 secondes pour laisser le temps aux processus de s'arrêter
        time.sleep(2)

        # Vérifier si des processus restent
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout.strip():
            print(f"Certains processus {engine} n'ont pas pu être arrêtés. Tentative avec SIGKILL...")
            for pid in result.stdout.strip().split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Processus {engine.capitalize()} (PID: {pid}) forcé à s'arrêter.")
                except Exception as e:
                    print(f"Erreur pour PID {pid}: {e}")

    except Exception as e:
        print(f"Erreur lors de l'arrêt des processus {engine}: {e}")


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

    if args.skip_bench:
        set_env_from_gpu_config(args.config)
        main_impact()
        print("--- Mode Skip : Inférence ignorée ---")
    else:
        # Lecture de la config pour connaître le moteur et les modèles avant de commencer
        with open(args.config, "r") as f:
            config = json.load(f)
            MODELS = config["Models"]
            inference_engine = str(config.get("Inference_Engine", "ollama")).lower()

        # Libérer les ports utilisés par le moteur d'inférence avant la première itération
        ports_engine = detecter_ports(inference_engine)
        if ports_engine:
            print(f"Libération des ports utilisés par {inference_engine} : {ports_engine}")
            tuer_tous_processus(inference_engine)
        else:
            print(f"Aucun port {inference_engine} détecté avant le benchmark.")

        for model in MODELS:
            print(f"Running the Sim LLM benchmark with the model: {model}")

            os.environ["BENCH_MODEL"] = model
            
            # Cette fonction lit la config, setup le TOML, ET lance les scripts serveurs via run_front_bash_script
            set_env_from_gpu_config(args.config)

            # 2. On met à jour le nom du modèle pour le script multi_gpu_bench
            # Uniquement Ollama a besoin de la concaténation "-swe" due au Modelfile
            if inference_engine == "ollama":
                custom_model = f"{model.replace(':', '-')}-swe"
                os.environ["BENCH_MODEL"] = custom_model
            else:
                os.environ["BENCH_MODEL"] = model

            # ==========================================
            # PHASE 1.1 : GPU impact calculation
            # ==========================================
            main_impact()

            # Afficher toutes les variables d'environnement utiles au debug
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

            # Nettoyage des serveurs d'inférence démarrés par set_env_from_gpu_config
            ports_engine = detecter_ports(inference_engine)
            if ports_engine:
                print(f"Arrêt des instances {inference_engine} de cette itération (Ports: {ports_engine})")
                tuer_tous_processus(inference_engine)


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
