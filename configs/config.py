#!/usr/bin/env python3
import argparse
import json
import os
from utils.utils_file import run_front_bash_script
from typing import Dict, Any
import tomli_w

def set_env_from_gpu_config(config_path: str) -> None:
    """Lit le fichier JSON et définit les variables d'environnement."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Nombre total de GPU
    num_gpus = len(config["gpus"])
    os.environ["BENCH_NUM_GPU"] = str(num_gpus)
    os.environ["BENCH_PUE"] = str(config["PUE"])
    os.environ["BENCH_USERS"]= json.dumps(config["Nb_users"])
    model = os.environ.get("BENCH_MODEL","mistral:7b")
    toml_config = {
        "model": model,
        "system_message": "You are an assistant",
        "ollama_instances": {},
    }
    # Pour chaque GPU, définir les variables d'environnement
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

        toml_config["ollama_instances"][f"127.0.0.1:{53100 + int(gpu_id)}"] = int(gpu_id)
    


    with open("configs/config.toml", "wb") as f:
        tomli_w.dump(toml_config, f)



    run_front_bash_script("scripts/ollama-batch-servers.sh",os.environ["BENCH_NUM_GPU"],model)
    print(f"Variables d'environnement définies pour {num_gpus} GPU(s).")

def main():
    parser = argparse.ArgumentParser(description="Définir des variables d'environnement à partir d'un fichier JSON de configuration GPU.")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier JSON de configuration des GPU.")
    args = parser.parse_args()

    set_env_from_gpu_config(args.config)

    # Afficher les variables définies (optionnel)
    for key, value in os.environ.items():
        if key.startswith("BENCH_"):
            print(f"{key}={value}")
    

if __name__ == "__main__":
    main()


