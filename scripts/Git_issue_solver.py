import argparse
import json
import os
import subprocess
import threading

import yaml
import uuid
import shutil
from pathlib import Path



# --- CONFIGURATION & ARGUMENTS ---
def parse_args():
    p = argparse.ArgumentParser(description='Run Git issue solver agent')
    p.add_argument('--user-id', type=int, default=None)
    p.add_argument('--host', type=str, default="http://localhost:11434")
    p.add_argument('--n_users', type=int, default=1)
    p.add_argument('--iter', type=int, default=0)
    return p.parse_args()

args = parse_args()
ID = int(args.user_id) if args.user_id is not None else 0
HOST = args.host
ITER = args.iter
NB_USER = args.n_users
MODEL = os.environ.get("BENCH_MODEL", "gemma3:4b")
ENGINE = os.environ.get("BENCH_INFERENCE_ENGINE", "ollama").lower()
ISSUES = json.loads(os.environ.get("BENCH_ISSUES", "[]"))

# --- GESTION DES CHEMINS ABSOLUS ---
ABS_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ABS_ROOT / "logs" / "parsed"
clean_model = MODEL.split("::")[0] if "::" in MODEL else MODEL
SAFE_MODEL = clean_model.replace(':', '-').replace('/', '_')

LOG_FILE = LOG_DIR / f"results_{SAFE_MODEL}.jsonl"
log_lock = threading.Lock()

agent_env_path = os.path.join('agent_env', f'agent_env_user_{SAFE_MODEL}_{NB_USER}_{ID}_{ITER}')
os.makedirs(agent_env_path, exist_ok=True)
os.chdir(agent_env_path)

source_yaml = ABS_ROOT / "data" / "swebench.yaml"
swe_config_yaml = Path("swebench.yaml")

# --- GENERATION DU JOB ID UNIQUE ---
# Permet d'isoler cette exécution pour le nettoyage Docker
job_id = f"job_user_{ID}_iter_{ITER}_{uuid.uuid4().hex[:6]}"
print(f"[{job_id}] Initialisation de la tâche...")

# 2. Copie du fichier
shutil.copy(source_yaml, swe_config_yaml)

# 3. Lecture du fichier YAML copié
with open(swe_config_yaml, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 4. Modification des valeurs avec vos variables dynamiques
if 'model' in config:
    # On utilise le format openai/ pour que LiteLLM/LangChain comprenne
    clean_model = MODEL.split("::")[0] if "::" in MODEL else MODEL
    if ENGINE == "vllm":
        config['model']['model_name'] = f"hosted_vllm/{clean_model}"
    else:
        config['model']['model_name'] = f"openai/{clean_model}"
    config['model']['api_base'] = f"{HOST}/v1"
    config['model']['api_key'] = "sk-dummy-key"

# --- INJECTION DU LABEL DOCKER POUR LE NETTOYAGE ISOLE ---
if 'environment' not in config:
    config['environment'] = {}
if 'run_args' not in config['environment']:
    # S'assure que les args de base sont là si la clé n'existait pas
    config['environment']['run_args'] = ["--rm", "--net=host"]

# Ajoute le label unique à la liste des arguments Docker de SWE-bench
config['environment']['run_args'].extend(["--label", f"run_id={job_id}"])


# 5. Sauvegarde du fichier modifié
with open(swe_config_yaml, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"[{job_id}] Fichier YAML configuré avec le modèle : {MODEL} sur {HOST}")


if __name__ == "__main__":
    
    print(f"[{job_id}] Starting User {ID} Benchmark...")
    filter_string = "|".join(ISSUES)
    # 1. Définir la commande sous forme de liste (recommandé pour subprocess)
    clean_model = MODEL.split("::")[0] if "::" in MODEL else MODEL
    cmd = [
        "mini-extra", "swebench",
        "--subset", "lite",
        "--split", "test",
        "-m", f"openai/{clean_model}",  # Utilise dynamiquement le modèle configuré
        "--filter", filter_string,
        "-c", f"{swe_config_yaml}"
    ]
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = "sk-dummy-key"
    env["OPENAI_API_BASE"] = f"{HOST}/v1"
    env["OPENAI_BASE_URL"] = f"{HOST}/v1"

    env["LITELLM_LOG"] = "DEBUG"
    env["LITELLM_LOCAL_LOGGING"] = "True"
    print(f"[{job_id}] Exécution de la commande : {' '.join(cmd)}")
    print("-" * 40)
    
    try:
        process = subprocess.run(
            cmd,
            check=True,
            env=env
        )
        result = "Exécution terminée avec succès."
        
    except subprocess.CalledProcessError as e:
        print(f"\n[{job_id}] La commande a échoué avec le code erreur {e.returncode}")
        result = "ERREUR : La commande a planté. Consultez les logs ci-dessus."
    except FileNotFoundError:
        result = f"[{job_id}] Erreur : La commande 'mini-extra' est introuvable."
    except KeyboardInterrupt:
        result = f"\n[{job_id}] Processus interrompu manuellement par l'utilisateur."
    finally:
        # Nettoyage exclusif des conteneurs associés à CE run spécifique
        print(f"\n[{job_id}] Nettoyage des conteneurs SWE-bench orphelins...")
        cleanup_cmd = f'docker rm -f $(docker ps -qa --filter "label=run_id={job_id}") 2>/dev/null'
        os.system(cleanup_cmd)
        
    # 3. Afficher le résultat final
    print(f"\n--- Bench result User {ID} ---\n{result}")
