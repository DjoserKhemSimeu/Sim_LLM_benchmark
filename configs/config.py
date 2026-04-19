#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from itertools import product
import subprocess

import tomli_w

from utils.utils_file import run_front_bash_script

GPU_CHARACTERISTICS = {
    "vercors16": {
        "gpu_model": "NVIDIA L4",
        "gpu_fp32_tflops": 30.3,
        "gpu_memory_gib": 22,
        "gpu_num": 8,
    },
    "hydra": {
        "gpu_model": "NVIDIA GH200",
        "gpu_fp32_tflops": 67,
        "gpu_memory_gib": 96,
        "gpu_num": 1,
    },
    "chuc": {
        "gpu_model": "NVIDIA A100-SXM4-40GB",
        "gpu_fp32_tflops": 19.49,
        "gpu_memory_gib": 40,
        "gpu_num": 4,
    },
    "neowise": {
        "gpu_model": "AMD Radeon Instinct MI50 32GB",
        "gpu_fp32_tflops": 13.41,
        "gpu_memory_gib": 32,
        "gpu_num": 8,
    },
}


def set_env_from_gpu_config(config_path: str) -> None:
    """Lit le fichier JSON et définit les variables d'environnement."""
    with open(config_path, "r") as f:
        config = json.load(f)

    tegra = 1 if shutil.which("tegrastats") is not None else 0

    os.environ["BENCH_TEGRA"] = str(tegra)
    os.environ["BENCH_MANUFACTURE_DATA"] = config["MANUFACTURE_DATA"]
    print(f"Using manufacturing data set: {config['MANUFACTURE_DATA']}")
    os.environ["BENCH_PUE"] = str(config["PUE"])
    os.environ["BENCH_USERS"] = json.dumps(config["Nb_users"])
    os.environ["BENCH_MODELS"] = json.dumps(config["Models"])
    os.environ["BENCH_ITERATION"] = str(config.get("Iteration", 10))
    os.environ["BENCH_GIT_SSH"] = config["GitHub_SSH"]
    os.environ["BENCH_OWNER"] = config["Owner"]
    os.environ["BENCH_REPO_NAME"] = config["Repo_Name"]

    os.environ["BENCH_GPU_MODEL"] = config["gpu_model"]
    os.environ["BENCH_FP32_TFLOPS"] = str(config["gpu_fp32_tflops"])
    os.environ["BENCH_GPU_MEMORY_GIB"] = str(config["gpu_memory_gib"])
    gpu_num = config["gpu_num"]
    os.environ["BENCH_NUM_GPU"] = str(gpu_num)

    temps = config.get("temperature", [0.0])
    topks = config.get("topK", [5])
    topps = config.get("topP", [0.9])

    if not isinstance(temps, list):
        temps = [temps]
    if not isinstance(topks, list):
        topks = [topks]
    if not isinstance(topps, list):
        topps = [topps]

    all_combinations = list(product(temps, topks, topps))
    print(f"Number of combinations: {len(all_combinations)}")

    model = config["Models"][0]
    os.environ["BENCH_MODEL"] = model

    for temp, topK, topP in all_combinations:
        os.environ["BENCH_TEMPERATURE"] = str(temp)
        os.environ["BENCH_TOPK"] = str(topK)
        os.environ["BENCH_TOPP"] = str(topP)
        os.environ["BENCH_NUM_GPU"] = str(gpu_num)

        combo_tag = f"temp{temp}_topk{topK}_topp{topP}_gpu{gpu_num}".replace(".", "_")
        os.environ["BENCH_COMBO_TAG"] = combo_tag

        print(f"Running combination: temp={temp}, topK={topK}, topP={topP}, gpu_num={gpu_num}")

        toml_config = {
            "model": model,
            "temperature": temp,
            "system_message": "You are an assistant",
            "ollama_instances": {},
            "topK": topK,
            "topP": topP,
        }

        instance_index = 0

        for gpu_id, gpu_info in config["gpus"].items():
            prefix = f"BENCH_GPU_{gpu_id}"

            #os.environ[f"{prefix}_NAME"] = gpu_info["nom"]
            os.environ[f"{prefix}_DIE_AREA"] = str(gpu_info["die_area"])
            os.environ[f"{prefix}_TDP"] = str(gpu_info["tdp"])
            os.environ[f"{prefix}_TECH_NODE"] = gpu_info["tech_node"]
            os.environ[f"{prefix}_MEM_TYPE"] = gpu_info["type_memoire"]
            os.environ[f"{prefix}_MEM_SIZE"] = str(gpu_info["taille_memoire"])
            os.environ[f"{prefix}_FOUNDRY"] = gpu_info["foundry"]
            os.environ[f"{prefix}_RELEASE_DATE"] = gpu_info["date_sortie"]
            os.environ[f"{prefix}_FU"] = gpu_info["fu"]
            os.environ[f"{prefix}_DENSITY"] = str(gpu_info["density"])

            gpu_count = gpu_num if gpu_id == "0" else int(gpu_info.get("gpu_num", 1))
            os.environ[f"{prefix}_GPU_NUM"] = str(gpu_count)

            for _ in range(gpu_count):
                toml_config["ollama_instances"][f"127.0.0.1:{53100 + instance_index}"] = int(gpu_id)
                instance_index += 1

        with open("configs/config.toml", "wb") as f:
            tomli_w.dump(toml_config, f)

        run_front_bash_script("scripts/ollama-batch-servers.sh", str(gpu_num), model)

        subprocess.run(
            ["python3", "scripts/multi_gpu_bench.py", "--config", "configs/config.toml"],
            check=True,
            env=os.environ.copy()
        )

    print("All combinations have been prepared and executed.")


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