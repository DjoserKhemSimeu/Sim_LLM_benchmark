import asyncio
import time
import random
import json
import toml
import argparse
import subprocess
import sys

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from ollama import AsyncClient
from utils.utils_file import log_message, run_back_bash_script, run_front_bash_script
import threading
import importlib
import json

# Globals to track issue solver subprocesses and handles
ISSUE_SOLVER_PROCS = []
ISSUE_SOLVER_HANDLES = []
LAUNCH_ISSUE_SOLVER = True
ISSUE_SOLVER_PATH = os.path.join('scripts', 'Git_issue_solver.py')
WAIT_FOR_SOLVERS = False
SYNC_ISSUE_SOLVER = True
import functools
# --- CONFIGURABLES ---
T = 2  # Durée du benchmark (en secondes)
lamb = 2  # Taux moyen d'arrivée (req/s)
Max = 1  # Nb max de requêtes par utilisateur
N=int(os.environ.get("BENCH_ITERATION", 10))  # Nombre total d'itérations
MODEL = os.environ.get("BENCH_MODEL", "mistral:7b")
NUM_GPUS=int(os.environ.get("BENCH_NUM_GPU", 1))
print_lock = threading.Lock()
df_prompts = pd.read_csv("data/prompts.csv")

def warmup_gpu(gpu_id, model):
    print(f"Préchauffage du modèle Ollama sur le GPU {gpu_id}...")
    try:
        url = f"http://localhost:{53100 + int(gpu_id)}/api/generate"
        response = requests.post(url, json={
            "model": model,
            "prompt": "warmup: Are you ready to run (yes/no)?",
            "stream": False
        }, timeout=300) # Ajout d'un timeout (5 min) car le chargement VRAM peut être long
        response.raise_for_status() # Vérifie que la requête a réussi (code 200)
        print(f"Modèle chargé en VRAM sur le GPU {gpu_id} !")
        return gpu_id, True
    except Exception as e:
        print(f"Erreur lors du préchauffage sur le GPU {gpu_id} : {e}")
        return gpu_id, False

def load_env_from_directory(env_dir):
    """Charge toutes les variables d'environnement depuis un fichier .env dans un dossier."""
    env = {}

    env_file = os.path.join(env_dir, ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env[key] = value

    return env


# --- FONCTION D'UN UTILISATEUR ---
async def simulate_user(user_id, model, hosts, delta_t_collector,nb_users,n):
    """Simule un utilisateur envoyant des requêtes asynchrones à plusieurs instances Ollama."""
    times = []
    delta_ts = []
    start = time.time()
    i = 0
    clients = [AsyncClient(host=f"http://{h}") for h in hosts]
    nb_hosts = len(clients)

   
    delta_t = np.random.exponential(1 / lamb)
    delta_ts.append(delta_t)
    await asyncio.sleep(delta_t)



    host_idx = user_id % nb_hosts
    client = clients[host_idx]

    delta_t_collector[user_id] = delta_ts

    if LAUNCH_ISSUE_SOLVER:
        try:
            logs_dir = os.path.join('logs', 'issue_solver')
            os.makedirs(logs_dir, exist_ok=True)
            out_path = os.path.join(logs_dir, f'user_{model}_{nb_users}_{user_id}_{n}.stdout.log')
            err_path = os.path.join(logs_dir, f'user_{model}_{nb_users}_{user_id}_{n}.stderr.log')
            agent_env_path = os.path.join('agent_env', f'agent_env_user_{MODEL}_{nb_users}_{user_id}_{n}')

            os.makedirs(agent_env_path, exist_ok=True)
            cmd = [ sys.executable, ISSUE_SOLVER_PATH, '--user-id', str(user_id), '--host', f'http://{hosts[host_idx]}','--n_users', str(nb_users), '--iter', str(n) ]
            if SYNC_ISSUE_SOLVER:
                # run synchronously for this client but offload blocking call to threadpool
                out_log = open(out_path, 'a')
                err_log = open(err_path, 'a')
                run_call = functools.partial(subprocess.run, cmd,stdout=out_log, stderr=err_log, env=os.environ.copy(), close_fds=True)
                loop = asyncio.get_running_loop()
                log_message(f"[User {user_id}] Running issue solver synchronously (cmd={' '.join(cmd)})")
                try:
                    # await completion in threadpool
                    completed = await loop.run_in_executor(None, run_call)
                    log_message(f"[User {user_id}] Issue solver completed with returncode={getattr(completed, 'returncode', None)}")
                except Exception as e:
                    log_message(f"[User {user_id}] Issue solver sync run failed: {e}")
                finally:
                    try:
                        out_log.close()
                    except Exception:
                        pass
                    try:
                        err_log.close()
                    except Exception:
                        pass
            else:
                out_log = open(out_path, 'a')
                err_log = open(err_path, 'a')
                proc = subprocess.Popen(cmd, stdout=out_log, stderr=err_log, env=os.environ.copy(), close_fds=True)
                # keep references to proc and file handles for optional waiting/cleanup
                ISSUE_SOLVER_PROCS.append(proc)
                ISSUE_SOLVER_HANDLES.append((out_log, err_log))
                log_message(f"[User {user_id}] Started issue solver subprocess (pid={proc.pid}), logs -> {logs_dir}")
        except Exception as e:
            log_message(f"[User {user_id}] Failed to start issue solver subprocess: {e}")
    end=time.time()
    times.append(end - start)
    return times


# --- BENCHMARK ---
async def benchmark(config_path, users_list):
    # Charger la config
    config = toml.load(config_path)
    model = config.get("model", "mistral:7b")
    gpus = config["ollama_instances"]
    hosts = list(gpus.keys())

    log_message(f"Configuration chargée : {len(hosts)} hôtes, modèle = {model}")

    results = {}
    delta_t_data = {}
    print("--- Début du préchauffage en parallèle ---")

    # max_workers définit le nombre de threads simultanés (un par GPU)
    with ThreadPoolExecutor(max_workers=NUM_GPUS) as executor:
        # On soumet toutes les tâches à l'exécuteur
        futures = [executor.submit(warmup_gpu, gpu_id, model) for gpu_id in range(NUM_GPUS)]
        
        # La boucle 'as_completed' attend et traite les réponses au fur et à mesure de leur arrivée
        for future in as_completed(futures):
            gpu_id, success = future.result()


    print("--- Préchauffage de tous les GPU terminé ---")

    for n_users in users_list:
    
        save_times = {}
        save_delta_t = {}
        for i in range(N):

            log_message(f"\n=== Benchmark avec {n_users} utilisateurs for iteration {i} ===")
            run_back_bash_script("measure/scripts/script_start_tx.sh", str(n_users), MODEL, str(i))

            tasks = []
            delta_t_collector = {}
            for u in range(n_users):
                tasks.append(simulate_user(u, model, hosts, delta_t_collector,n_users,i))

            all_times_nested = await asyncio.gather(*tasks)
            all_times = [t for sublist in all_times_nested for t in sublist]

            run_front_bash_script("measure/scripts/script_stop_tx.sh")
            save_times[i] = all_times
            save_delta_t[i] = [
                dt for user in delta_t_collector for dt in delta_t_collector[user]
            ]

        results[n_users] = save_times
        delta_t_data[n_users] = save_delta_t

        log_message(f"Completed {n_users} users, total requests: {len(all_times)}")
        await asyncio.sleep(5)

    return results, delta_t_data


# --- VISUALISATION ---
def plot_latency_and_efficiency(results):
    # `results` can be either:
    # - {nb_users: [latency_list]} (old format)
    # - {nb_users: {iteration: [latency_list], ...}} (new format)
    users = sorted(results.keys())

    def flatten_times(item):
        # item may be a list of latencies or a dict of iterations -> lists
        if isinstance(item, dict):
            all_times = []
            for v in item.values():
                if isinstance(v, list):
                    all_times.extend(v)
                elif isinstance(v, (int, float)):
                    all_times.append(v)
            return all_times
        elif isinstance(item, list):
            return item
        else:
            return []

    latencies_lists = [flatten_times(results[u]) for u in users]
    # For plotting/aggregation, replace empty lists with [np.nan] so matplotlib
    # can accept the dataset while keeping the slot for that user count.
    latencies_for_plot = [l if len(l) > 0 else [np.nan] for l in latencies_lists]

    # Average latency per user-count (mean over all iterations and observations)
    avg_latencies = [float(np.nanmean(l)) if len(l) > 0 else float('nan') for l in latencies_lists]
    avg_req_user = [ (len([x for x in l if not (isinstance(x, float) and np.isnan(x))]) / users[i]) if users[i] > 0 else 0 for i, l in enumerate(latencies_lists) ]

    # Calcul de M = (lamb * nb_users) / latence_moyenne (guard against div-by-zero/nan)
    M = []
    for i, u in enumerate(users):
        avg = avg_latencies[i]
        if np.isnan(avg) or avg == 0:
            M.append(0.0)
        else:
            M.append((lamb * u) / avg)

    # --- Enregistrement des données dans un fichier CSV ---
    # Création d'un DataFrame pour les latences et M
    data = {
        "nb_users": users,
        "avg_latency": avg_latencies,
        "efficiency_M": M,
        "avg_req_user": avg_req_user,
    }
    print(data)

    df = pd.DataFrame(data)

    # Création du répertoire 'data' s'il n'existe pas
    os.makedirs("./measure/data", exist_ok=True)

    # Enregistrement du fichier CSV
    csv_path = f"./measure/data/latency_efficiency_data_{MODEL}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Données enregistrées sous {csv_path}")
    raw_data = []
    for u, lats in zip(users, latencies_lists):
        for lat in lats:
            raw_data.append({"nb_users": u, "latency": lat})

    df_raw = pd.DataFrame(raw_data)
    print(df_raw)
    raw_csv_path = f"./measure/data/raw_latencies_{MODEL}.csv"
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Données brutes enregistrées sous {raw_csv_path}")
    # --- Création des graphiques ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Graphique 1 : Boxplot de la latence
    positions = np.arange(len(users))
    bp = ax1.boxplot(
        latencies_for_plot, positions=positions, widths=0.6, patch_artist=True, showfliers=False
    )
    colors = plt.cm.viridis(np.linspace(0, 1, len(users)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(users)
    ax1.set_xlabel("Number of concurrents users")
    ax1.set_ylabel("latencies per queries")
    ax1.set_title(f"Latency distribution{MODEL}")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)

    # Graphique 2 : Efficacité (M)
    ax2.plot(users, M, "bo-", label="Efficacité (M)")
    ax2.set_xlabel("Number of concurrents users")
    ax2.set_ylabel("Efficacity (queries/s / latency)")
    ax2.set_title(f"Model efficacity {MODEL} (λ = {lamb})")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"./measure/data/latency_efficiency_{MODEL}.png")
    print(f"Graphiques enregistrés sous ./measure/data/latency_efficiency_{MODEL}.png")


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronous Ollama Benchmark")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.toml",
        help="Path to TOML config file",
    )
    parser.add_argument('--no-issue-solver', action='store_true', help='Do not launch Git_issue_solver subprocesses per user')
    parser.add_argument('--issue-solver-path', type=str, default=os.path.join('scripts', 'Git_issue_solver.py'), help='Path to the Git_issue_solver script')
    parser.add_argument('--wait-for-solvers', action='store_true', help='Wait for launched issue solver subprocesses to finish before exiting')
    parser.add_argument('--no-sync-issue-solver', action='store_true', help='Do not wait for issue solver per client (run in background)')
    users_list = json.loads(os.environ.get("BENCH_USERS", "[1,10,100]"))

    args = parser.parse_args()

    # configure issue solver behavior

    if args.no_issue_solver:
        LAUNCH_ISSUE_SOLVER = False
    ISSUE_SOLVER_PATH = args.issue_solver_path
    WAIT_FOR_SOLVERS = args.wait_for_solvers
    # By default clients wait for the solver; use --no-sync-issue-solver to disable
    SYNC_ISSUE_SOLVER = not getattr(args, 'no_sync_issue_solver', False)

    try:
        results, delta_t = asyncio.run(benchmark(args.config, users_list))
        plot_latency_and_efficiency(results)
        log_message("Benchmark terminé avec succès.")
        # After benchmark: parse Ollama server logs for each host and save parsed JSONL
        try:
            # load parse_ollama_log module from local scripts (same directory)
            try:
                import parse_ollama_log
            except Exception:
                # fallback: import by path
                spec = importlib.util.spec_from_file_location("parse_ollama_log", os.path.join(os.path.dirname(__file__), "parse_ollama_log.py"))
                parse_ollama_log = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(parse_ollama_log)

            # reload config to get hosts and model
            cfg = toml.load(args.config)
            model = cfg.get("model", "mistral:7b")
            instances = cfg.get("ollama_instances", {})
            hosts = list(instances.keys())
            LOG_DIR = "ollama-server-logs"
            parsed_out_dir = os.path.join("logs", "parsed")
            os.makedirs(parsed_out_dir, exist_ok=True)

            for host in hosts:
                # construct expected log path: ${LOG_DIR}/${PORT}_${MODEL}.log
                # Hosts may be like '127.0.0.1:53100', 'localhost:53100' or include scheme 'http://127.0.0.1:53100'.
                try:
                    h = host
                    # strip scheme if present
                    if h.startswith("http://") or h.startswith("https://"):
                        h = h.split("//", 1)[1]
                    port = h.split(":")[-1]
                except Exception:
                    port = host.replace(":", "_")
                log_path = os.path.join(LOG_DIR, f"{port}_{model}.log")
                # if os.path.exists(log_path):
                #     # try:
                #     #     # items = parse_ollama_log.parse_log(log_path)
                #     #     # out_path = os.path.join(parsed_out_dir, f"parsed_{port}_{model}.jsonl")
                #     #     # with open(out_path, "w", encoding="utf-8") as outf:
                #     #     #     for it in items:
                #     #     #         outf.write(json.dumps(it, ensure_ascii=False) + "\n")
                #     #     # log_message(f"Parsed {log_path} -> {out_path} ({len(items)} items)")
                #     # except Exception as e:
                #     #     log_message(f"Failed parsing {log_path}: {e}")
                # else:
                #     log_message(f"Log file not found: {log_path}")
        except Exception as e:
            log_message(f"Post-benchmark parsing failed: {e}")
    except KeyboardInterrupt:
        log_message("Interrompu par l’utilisateur.")
