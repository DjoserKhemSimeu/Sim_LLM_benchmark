#!/usr/bin/env python3
import argparse
import json
import requests
import os
from utils.utils_file import run_front_bash_script
from typing import Dict, Any
import tomli_w
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
    os.environ["BENCH_TEGRA"] = str(tegra)
    os.environ["BENCH_NUM_GPU"] = str(num_gpus)
    os.environ["BENCH_MANUFACTURE_DATA"] = config["MANUFACTURE_DATA"]
    print(f"Using manufacturing data set: {config['MANUFACTURE_DATA']}")
    os.environ["BENCH_PUE"] = str(config["PUE"])
    os.environ["BENCH_USERS"] = json.dumps(config["Nb_users"])
    os.environ["BENCH_MODELS"] = json.dumps(config["Models"])
    os.environ["BENCH_ITERATION"] = str(config.get("Iteration", 10))
    os.environ["BENCH_GIT_SSH"] = config["GitHub_SSH"]
    os.environ["BENCH_OWNER"] = config["Owner"]
    os.environ["BENCH_REPO_NAME"] = config["Repo_Name"]

    model = os.environ.get("BENCH_MODEL", "mistral:7b")
    toml_config = {
        "model": model,
        "temperature": 0.0,
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
        os.environ[f"{prefix}_DENSITY"] = str(gpu_info["density"])

        toml_config["ollama_instances"][f"localhost:{53100 + int(gpu_id)}"] = int(
            gpu_id
        )
    
    with open("configs/config.toml", "wb") as f:
        tomli_w.dump(toml_config, f)

    run_front_bash_script(
        "scripts/ollama-batch-servers.sh", os.environ["BENCH_NUM_GPU"], model
    )

    print(f"Variables d'environnement définies pour {num_gpus} GPU(s).")
    # for gpu_id, gpu_info in config["gpus"].items():
    #     print("Préchauffage du modèle Ollama...")
    # try:
    #     requests.post(f"http://localhost:{53100 + int(gpu_id)}/api/generate", json={
    #         "model": model,
    #         "prompt": "warmup: Are you ready to run (yes/no)?",
    #         "stream": False
    #     })
    #     print("Modèle chargé en VRAM !")
    # except Exception as e:
    #     print(f"Erreur lors du préchauffage : {e}")


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
