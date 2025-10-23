import asyncio
import time
import random
import json
import toml
import argparse
import subprocess
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from ollama import AsyncClient
from utils.utils_file import log_message, run_back_bash_script
import threading

# --- CONFIGURABLES ---
T = 1800      # Durée du benchmark (en secondes)
lamb = 0.025 # Taux moyen d'arrivée (req/s)
Max = 16     # Nb max de requêtes par utilisateur
MODEL=os.environ.get("BENCH_MODEL","mistral:7b")
print_lock = threading.Lock()
df_prompts=pd.read_csv("data/prompts.csv")

# --- FONCTION D'UN UTILISATEUR ---
async def simulate_user(user_id, model, hosts, delta_t_collector):
    """Simule un utilisateur envoyant des requêtes asynchrones à plusieurs instances Ollama."""
    times = []
    delta_ts = []
    start = time.time()
    i = 0
    clients = [AsyncClient(host=f"http://{h}") for h in hosts]
    nb_hosts = len(clients)

    while time.time() < start + T and i < Max:
        delta_t = np.random.exponential(1 / lamb)
        delta_ts.append(delta_t)
        await asyncio.sleep(delta_t)

        if time.time() >= start + T:
            break

        host_idx = user_id % nb_hosts
        client = clients[host_idx]
        prompt=df_prompts["content"].sample(n=1).iloc[0]
        msg = {"role": "user", "content": f"{prompt}"}

        try:
            start_req = time.time()
            #response = await client.chat(model=model, messages=[msg])
            time.sleep(0.5)#TEST
            elapsed = time.time() - start_req
            times.append(elapsed)

            log_message(f"[User {user_id}] Req {i} on {hosts[host_idx]}: {elapsed:.3f}s")
        except Exception as e:
            log_message(f"[User {user_id}] Error on request {i}: {e}")
        i += 1

    delta_t_collector[user_id] = delta_ts
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

    for n_users in users_list:
        log_message(f"\n=== Benchmark avec {n_users} utilisateurs ===")
        run_back_bash_script('measure/scripts/script_start_tx.sh', str(n_users),MODEL)

        tasks = []
        delta_t_collector = {}
        for u in range(n_users):
            tasks.append(simulate_user(u, model, hosts, delta_t_collector))

        all_times_nested = await asyncio.gather(*tasks)
        all_times = [t for sublist in all_times_nested for t in sublist]

        run_back_bash_script('measure/scripts/script_stop_tx.sh')

        results[n_users] = all_times
        delta_t_data[n_users] = [dt for user in delta_t_collector for dt in delta_t_collector[user]]

        log_message(f"Completed {n_users} users, total requests: {len(all_times)}")
        await asyncio.sleep(5)

    return results, delta_t_data

# --- VISUALISATION ---
def plot_latency_and_efficiency(results):
    users = sorted(results.keys())
    latencies = [results[u] for u in users]
    avg_latencies = [np.mean(lats) for lats in latencies]

    # Calcul de M = (lamb * nb_users) / latence_moyenne
    M = [(lamb * u) / avg_latencies[i] for i, u in enumerate(users)]

    # --- Enregistrement des données dans un fichier CSV ---
    # Création d'un DataFrame pour les latences et M
    data = {
        "nb_users": users,
        "avg_latency": avg_latencies,
        "efficiency_M": M,
    }


    df = pd.DataFrame(data)

    # Création du répertoire 'data' s'il n'existe pas
    os.makedirs("./measure/data", exist_ok=True)

    # Enregistrement du fichier CSV
    csv_path = f"./measure/data/latency_efficiency_data_{MODEL}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Données enregistrées sous {csv_path}")

    # --- Création des graphiques ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Graphique 1 : Boxplot de la latence
    positions = np.arange(len(users))
    bp = ax1.boxplot(latencies, positions=positions, widths=0.6, patch_artist=True, showfliers=False)
    colors = plt.cm.viridis(np.linspace(0, 1, len(users)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(users)
    ax1.set_xlabel("Number of concurrents users")
    ax1.set_ylabel("latencies per queries")
    ax1.set_title(f"Latency distribution{MODEL}")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Graphique 2 : Efficacité (M)
    ax2.plot(users, M, 'bo-', label='Efficacité (M)')
    ax2.set_xlabel("Number of concurrents users")
    ax2.set_ylabel("Efficacity (queries/s / latency)")
    ax2.set_title(f"Model efficacity {MODEL} (λ = {lamb})")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"./measure/data/latency_efficiency_{MODEL}.png")
    print(f"Graphiques enregistrés sous ./measure/data/latency_efficiency_{MODEL}.png")
# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronous Ollama Benchmark")
    parser.add_argument("--config", type=str, default="configs/config.toml", help="Path to TOML config file")
    users_list= json.loads(os.environ.get("BENCH_USERS","[1,10,100]"))

    args = parser.parse_args()

    try:
        results, delta_t = asyncio.run(benchmark(args.config, users_list))
        plot_latency_and_efficiency(results)
        log_message("Benchmark terminé avec succès.")
    except KeyboardInterrupt:
        log_message("Interrompu par l’utilisateur.")

