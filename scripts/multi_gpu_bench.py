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
T = 10      # Durée du benchmark (en secondes)
lamb = 1
# Taux moyen d'arrivée (req/s)
Max = 32     # Nb max de requêtes par utilisateur
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
            response = await client.chat(model=model, messages=[msg])
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
def plot_results(results, delta_t_data):
    users = sorted(results.keys())
    latencies = [results[u] for u in users]
    avg_requests_per_user = [len(results[u]) / u for u in users]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    bp = ax1.boxplot(latencies, positions=users, widths=0.6, patch_artist=True, showfliers=False)
    ax1.set_xticks(users)
    ax1.set_xlabel("Number of concurrent users")
    ax1.set_ylabel("Latency per request (s)")
    ax1.set_title(f"LLM Latency and Requests per User for {MODEL}")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    ax1b = ax1.twinx()
    ax1b.plot(users, avg_requests_per_user, 'r-o', label='Avg requests per user')
    ax1b.set_ylabel("Avg requests per user", color='tab:red')
    ax1b.tick_params(axis='y', labelcolor='tab:red')
    ax1b.yaxis.set_major_locator(MaxNLocator(integer=True))

    colors = plt.cm.viridis(np.linspace(0, 1, len(users)))
    for i, u in enumerate(users):
        ax2.hist(delta_t_data[u], bins=30, alpha=0.5, density=True,
                 label=f'{u} users', color=colors[i], edgecolor='black')
    ax2.set_xlabel("Delta_t (s)")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution of delta_t per number of users")
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"./measure/data/results_bench_{MODEL}.png")
    log_message("Graph saved to ./measure/data/results_bench.png")

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asynchronous Ollama Benchmark")
    parser.add_argument("--config", type=str, default="configs/config.toml", help="Path to TOML config file")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 10, 100], help="List of concurrent user counts")

    args = parser.parse_args()

    try:
        results, delta_t = asyncio.run(benchmark(args.config, args.users))
        plot_results(results, delta_t)
        log_message("Benchmark terminé avec succès.")
    except KeyboardInterrupt:
        log_message("Interrompu par l’utilisateur.")

