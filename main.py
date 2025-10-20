#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import signal
import re
from configs.config import set_env_from_gpu_config

BENCH_SCRIPT = "scripts/multi_gpu_bench.py"
MANUFACTURING_IMPACT_SCRIPT = "measure/scripts/bar_impact.py"
EVALUATION_SCRIPT = "measure/scripts/perf_show.py"
MODELS = ["mistral:7b", "gpt-oss:20b", "gemma3:12b"]

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
            print("Certains processus Ollama n'ont pas pu être arrêtés. Tentative avec SIGKILL...")
            for pid in result.stdout.strip().split():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    print(f"Processus Ollama (PID: {pid}) forcé à s'arrêter.")
                except Exception as e:
                    print(f"Erreur pour PID {pid}: {e}")

    except Exception as e:
        print(f"Erreur lors de l'arrêt des processus Ollama: {e}")

def main():
    parser = argparse.ArgumentParser(description="Lance un benchmark après configuration des GPU.")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier JSON de configuration des GPU.")
    args = parser.parse_args()
    # Libérer les ports utilisés par Ollama avant la prochaine itération
    ports_ollama = detecter_ports_ollama()
    if ports_ollama:
        print(f"Libération des ports utilisés par Ollama : {ports_ollama}")
        tuer_tous_processus_ollama()
    else:
        print("Aucun port Ollama détecté.")


    for model in MODELS:
        print(f"Running the Sim LLM benchmark with the model: {model}")
        os.environ["BENCH_MODEL"] = model
        set_env_from_gpu_config(args.config)

        # Exécution du script d'impact
        process = subprocess.Popen(
            [sys.executable, MANUFACTURING_IMPACT_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
        return_code = process.wait()
        if return_code != 0:
            print("\n--- ERREUR D'EXÉCUTION (impact) ---")
            stderr_output = process.stderr.read()
            print("Backtrace :\n" + stderr_output)
            sys.exit(1)

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
            bufsize=1
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

