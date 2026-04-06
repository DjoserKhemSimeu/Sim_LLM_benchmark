#!/usr/bin/env python3
import argparse
import json
from logging import config
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
    num_gpus = sum(int(gpu_info.get("gpu_num", 1)) for gpu_info in config["gpus"].values())
    tegra = 0
    if shutil.which("tegrastats") is not None:
        tegra = 1
    #changer
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
    temp = str(config.get("temperature", 0.0))
    os.environ["BENCH_TEMPERATURE"] = str(temp)
    topK = str(config.get("topK", 5))
    os.environ["BENCH_TOPK"] = str(topK)
    topP = str(config.get("topP", 0.9))
    os.environ["BENCH_TOPP"] = str(topP)
    first_gpu = config["gpus"]["0"]
    os.environ["BENCH_GPU_MODEL"] = first_gpu["gpu_model"]
    os.environ["BENCH_FP32_TFLOPS"] = str(first_gpu["gpu_fp32_tflops"])
    os.environ["BENCH_GPU_MEMORY_GIB"] = str(first_gpu["gpu_memory_gib"])
    model = config["Models"][0]
    os.environ["BENCH_MODEL"] = model
    
    toml_config = {
        "model": model,
        "temperature": temp, #changer
        "system_message": "You are an assistant",
        "ollama_instances": {},
        "topK" : topK,
        "topP" : topP,
    }

    instance_index = 0

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
        os.environ[f"{prefix}_GPU_NUM"] = str(gpu_info.get("gpu_num", 1))

        gpu_count = int(gpu_info.get("gpu_num", 1))

        for _ in range(gpu_count):
            toml_config["ollama_instances"][f"127.0.0.1:{53100 + instance_index}"] = int(gpu_id)
            instance_index += 1

    with open("configs/config.toml", "wb") as f:
        tomli_w.dump(toml_config, f)

    run_front_bash_script(
        "scripts/ollama-batch-servers.sh", os.environ["BENCH_NUM_GPU"], model
    )
    print(f"Variables d'environnement définies pour {num_gpus} GPU(s).")


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
