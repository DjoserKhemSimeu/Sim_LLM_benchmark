#!/usr/bin/env python3
import argparse
import json
import requests
import os
from utils.utils_file import run_front_bash_script
from typing import Dict, Any
import tomli_w
import subprocess
import shutil

def set_env_from_gpu_config(config_path: str) -> None:
    """Lit le fichier JSON et définit les variables d'environnement."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Nombre total de GPU
    num_gpus = len(config["gpus"])
    tegra = 0
    if shutil.which("tegrastats") is not None:
        tegra = 1
        
    # Récupération du moteur d'inférence (en minuscules pour éviter les erreurs de casse)
    inference_engine = str(config.get("Inference_Engine", "ollama")).lower()

    os.environ["BENCH_TEGRA"] = str(tegra)
    os.environ["BENCH_NUM_GPU"] = str(num_gpus)
    os.environ["BENCH_PUE"] = str(config["PUE"])
    os.environ["BENCH_USERS"] = json.dumps(config["Nb_users"])
    os.environ["BENCH_ITERATION"] = str(config.get("Iteration", 10))
    os.environ["BENCH_INFERENCE_ENGINE"] = inference_engine
    os.environ["BENCH_MODELS"] = json.dumps(config["Models"])
    os.environ["BENCH_ISSUES"] = json.dumps(config["SWEbench_issues"])

    # Base model : "mistral:7b" pour ollama, ou "mistralai/Mistral-7B-Instruct-v0.2" pour vLLM
    base_model = os.environ.get("BENCH_MODEL", "mistral:7b")

    # Initialisation commune de la config TOML
    toml_config = {
        "temperature": 0.0,
        "system_message": "You are an assistant",
        "ollama_instances": {}, # Gardé tel quel si votre pipeline (ex: mini-swe-agent) lit cette clé spécifique
    }

    # Pour chaque GPU, définir les variables d'environnement et alimenter le TOML
    for gpu_id, gpu_info in config["gpus"].items():
        prefix = f"BENCH_GPU_{gpu_id}"
        os.environ[f"{prefix}_NAME"] = gpu_info["nom"]
        os.environ[f"{prefix}_DIE_AREA"] = str(gpu_info["die_area"])
        os.environ[f"{prefix}_TDP"] = str(gpu_info["tdp"])
        os.environ[f"{prefix}_TECH_NODE"] = gpu_info["tech_node"]
        os.environ[f"{prefix}_MEM_TYPE"] = gpu_info["type_memoire"]
        os.environ[f"{prefix}_MEM_SIZE"] = str(gpu_info["taille_memoire"])
        os.environ[f"{prefix}_FOUNDRY"] = gpu_info["foundry"]
        os.environ[f"{prefix}_RELEASE_DATE"] = gpu_info["date_sortie"]
        os.environ[f"{prefix}_FU"] = gpu_info["fu"]
        os.environ[f"{prefix}_DENSITY"] = str(gpu_info["density"])

        toml_config["ollama_instances"][f"localhost:{53100 + int(gpu_id)}"] = int(gpu_id)

    print(f"Variables d'environnement définies pour {num_gpus} GPU(s).")

    # ==========================================
    # BRANCHEMENT SELON LE MOTEUR D'INFÉRENCE
    # ==========================================
    if inference_engine == "ollama":
        # Logique spécifique OLLAMA
        custom_model = f"{base_model.replace(':', '-')}-swe"
        toml_config["model"] = custom_model
        
        # 1. CRÉATION AUTOMATIQUE DU MODELFILE
        print(f"Création du Modelfile pour {custom_model} (basé sur {base_model})...")
        modelfile_content = f"""FROM {base_model}
PARAMETER num_gpu 99
PARAMETER num_thread 4
"""
        with open("Modelfile", "w", encoding="utf-8") as f:
            f.write(modelfile_content)
            
        # Écriture de la configuration TOML
        with open("configs/config.toml", "wb") as f:
            tomli_w.dump(toml_config, f)

        # Lancement du batch script Ollama
        run_front_bash_script(
            "scripts/ollama-batch-servers.sh", os.environ["BENCH_NUM_GPU"], base_model
        )

        # 2. COMPILATION DU MODÈLE DANS OLLAMA
        print(f"Compilation en cours du modèle {custom_model} dans Ollama...")
        try:
            env_create = os.environ.copy()
            # On cible la première instance qu'on vient juste de démarrer via le bash
            env_create["OLLAMA_HOST"] = "localhost:53100"

            # subprocess exécute la commande 'ollama create' de manière synchrone
            subprocess.run(["ollama", "create", custom_model, "-f", "Modelfile"], env=env_create, check=True)
            print(f"Modèle {custom_model} créé avec succès et prêt pour l'inférence !")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de la compilation du modèle : {e}")
        except FileNotFoundError:
            print("L'exécutable 'ollama' n'a pas été trouvé.")

    elif inference_engine == "vllm":
        # Logique spécifique vLLM
        # vLLM n'a pas besoin de Modelfile ni de compilation
        custom_model = base_model
        toml_config["model"] = custom_model
        
        # Écriture de la configuration TOML
        with open("configs/config.toml", "wb") as f:
            tomli_w.dump(toml_config, f)

        print(f"Lancement des serveurs vLLM pour le modèle {custom_model}...")
        # Lancement du batch script vLLM
        run_front_bash_script(
            "scripts/vllm-batch-servers.sh", os.environ["BENCH_NUM_GPU"], base_model
        )
        print("Serveurs vLLM initialisés avec succès !")
        
    else:
        print(f"Attention: Moteur d'inférence inconnu ('{inference_engine}'). Fin du script.")


def main():
    parser = argparse.ArgumentParser(
        description="Définir des variables d'environnement à partir d'un fichier JSON de configuration GPU."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin vers le fichier JSON de configuration des GPU.",
    )
    args = parser.parse_args()

    set_env_from_gpu_config(args.config)

    # Afficher les variables définies (optionnel)
    for key, value in os.environ.items():
        if key.startswith("BENCH_"):
            print(f"{key}={value}")

if __name__ == "__main__":
    main()
