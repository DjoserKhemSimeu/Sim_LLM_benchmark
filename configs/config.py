#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import tomli_w
from utils.utils_file import run_front_bash_script

def set_env_from_gpu_config(config_path: str) -> None:
    """Lit le fichier JSON et definit les variables d'environnement."""
    with open(config_path, "r") as f:
        config = json.load(f)

    num_gpus = len(config["gpus"])
    tegra = 0
    if shutil.which("tegrastats") is not None:
        tegra = 1
        
    inference_engine = str(config.get("Inference_Engine", "ollama")).lower()

    os.environ["BENCH_TEGRA"] = str(tegra)
    os.environ["BENCH_NUM_GPU"] = str(num_gpus)
    os.environ["BENCH_PUE"] = str(config["PUE"])
    os.environ["BENCH_USERS"] = json.dumps(config["Nb_users"])
    os.environ["BENCH_ITERATION"] = str(config.get("Iteration", 10))
    os.environ["BENCH_INFERENCE_ENGINE"] = inference_engine
    os.environ["BENCH_MODELS"] = json.dumps(config["Models"])
    os.environ["BENCH_ISSUES"] = json.dumps(config["SWEbench_issues"])

    # Extraction du modele et du tokenizer
    base_model_raw = os.environ.get("BENCH_MODEL", "mistral:7b")
    
    if "::" in base_model_raw:
        parts = base_model_raw.split("::")
        base_model = parts[0] # Contient repo_id:quant_type
        tokenizer_repo = parts[1] if len(parts) > 1 else ""
    else:
        base_model = base_model_raw
        tokenizer_repo = ""

    toml_config = {
        "temperature": 0.0,
        "system_message": "You are an assistant",
        "ollama_instances": {}, 
    }

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

    print(f"Variables d'environnement definies pour {num_gpus} GPU(s).")

    if inference_engine == "ollama":
        custom_model = f"{base_model.replace(':', '-')}-swe"
        toml_config["model"] = custom_model
        
        print(f"Creation du Modelfile pour {custom_model} (base sur {base_model})...")
        modelfile_content = f"""FROM {base_model}\nPARAMETER num_gpu 99\nPARAMETER num_thread 4\n"""
        with open("Modelfile", "w", encoding="utf-8") as f:
            f.write(modelfile_content)
            
        with open("configs/config.toml", "wb") as f:
            tomli_w.dump(toml_config, f)

        run_front_bash_script(
            "scripts/ollama-batch-servers.sh", os.environ["BENCH_NUM_GPU"], base_model
        )

        print(f"Compilation en cours du modele {custom_model} dans Ollama...")
        try:
            env_create = os.environ.copy()
            env_create["OLLAMA_HOST"] = "localhost:53100"
            subprocess.run(["ollama", "create", custom_model, "-f", "Modelfile"], env=env_create, check=True)
            print(f"Modele {custom_model} cree avec succes et pret pour l'inference !")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de la compilation du modele : {e}")
        except FileNotFoundError:
            print("L'executable 'ollama' n'a pas ete trouve.")

    elif inference_engine == "vllm":
        toml_config["model"] = base_model
        
        with open("configs/config.toml", "wb") as f:
            tomli_w.dump(toml_config, f)

        print(f"Lancement des serveurs vLLM pour le modele {base_model}...")
        
        # Transmission du tokenizer optionnel au script bash
        if tokenizer_repo:
            run_front_bash_script(
                "scripts/vllm-batch-servers.sh", os.environ["BENCH_NUM_GPU"], base_model, tokenizer_repo
            )
        else:
            run_front_bash_script(
                "scripts/vllm-batch-servers.sh", os.environ["BENCH_NUM_GPU"], base_model
            )
        print("Serveurs vLLM initialises avec succes !")
        
    else:
        print(f"Attention: Moteur d'inference inconnu ('{inference_engine}'). Fin du script.")

def main():
    parser = argparse.ArgumentParser(
        description="Definir des variables d'environnement a partir d'un fichier JSON de configuration GPU."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Chemin vers le fichier JSON de configuration des GPU.",
    )
    args = parser.parse_args()
    set_env_from_gpu_config(args.config)

if __name__ == "__main__":
    main()